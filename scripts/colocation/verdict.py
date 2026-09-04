#!/usr/bin/env python3
"""Stage-5 verdict: one judgement per (mix, arm) cell from a measure_mixes.sh run.

  verdict.py --run results/coloc_<tag>_<stamp>   -> verdict.json + report.md in the run

Per cell (from <mix>/<arm>/concurrent_mix.json, period_makespan.json, clock_drift.json,
paced_solo/, composed/<mix>.json):
  valid   the number can be trusted: the arm ran (status OK), every frame row came from
          the row_loop driver and finished OK, every trace is present, the rows launched
          on one shared grid (aligned is not False), the clock lock held (drift PASS/WARN)
          and, under mps, the daemon was verified. Anything else -> the cell is INVALID,
          a field, not a warning.
  fits    every frame row: miss_frac < 1 % AND done before its own next trigger >= 99 %;
          every side row: its harness reported OK (ASR additionally RTF < 1).
  per frame row: contended p99, paced-solo p99 (same driver, same rate, alone),
          contention_factor = contended / paced solo, the composed prediction
          p99_contended_pred and its error, deadline misses.
  per side row: the harness summary against its solo run (ttft / decode / tok/s ratios).
  makespan p99 vs the period, free time = period - sum(paced-solo p99),
  N_measured = min(L_contended, C_composed) with L_contended = min(deadline / contended p99)
  over the frame rows, cause tag as in stage 4.
The report carries the matrix (mixes x arms), one row table per cell and the
composed budget per mix. Nothing here re-measures; every number is read.
"""
import argparse
import datetime
import glob
import json
import os

MISS_MAX = 0.01
DONE_MIN = 0.99
MAKESPAN_KEYS = ('period_ms', 'periods', 'aligned', 'launch_phase_ms', 'makespan_p50_ms',
                 'makespan_p99_ms', 'makespan_max_ms', 'all_rows_done_in_period_frac',
                 'overlap_frac_of_period_p50', 'rows_missing', 'rows_slid', 'makespan_basis')
SIDE_RATIO_KEYS = ('ttft_ms_median', 'decode_ms_p99', 'decode_ms_median', 'rtf_wall_median',
                   'tokens_per_second', 'step_ms_p99', 'gpu_total_ms_p99')
# (contended key, solo key, description) in preference order for the side row's contended/solo ratio
SIDE_RATIO_BASES = (
    ('window_request_ms_mean', 'request_ms_mean',
     'per-request wall mean (the battery is a fixed prompt set, so the mean is its stable summary)'),
    ('request_ms_mean', 'request_ms_mean', 'per-request wall mean (whole battery)'),
    ('window_decode_ms_p99_max', 'decode_ms_p99_max', 'decode p99 max'),
    ('decode_ms_p99_max', 'decode_ms_p99_max', 'decode p99 max (whole pass)'))


def load_json(path, default=None):
    try:
        return json.load(open(path))
    except Exception:
        return default


def fmt_pct(value, digits=1):
    return '-' if value is None else f'{100 * value:.{digits}f}%'


def fmt(value, digits=1):
    return '-' if value is None else f'{value:.{digits}f}'


def side_ratios(result, solo):
    """side-row contention: harness numbers vs the same harness alone. The ASR harness also reports
    window_* statistics over the clips that started inside the frame window; those are the contended
    numbers when present (the whole pass outlasts the rows by design), ratioed against the solo pass."""
    result = result or {}
    solo = solo or {}
    out = {}
    for key in SIDE_RATIO_KEYS:
        contended = result.get('window_' + key, result.get(key))
        alone = solo.get(key)
        if contended is None or not alone:
            continue
        basis = 'frame window' if ('window_' + key) in result else 'whole pass'
        out[key] = {'contended': contended, 'solo': alone, 'ratio': round(contended / alone, 3),
                    'basis': basis}
    return out


def _frame_row(frame, result, paced_solo, makespan_row, predicted_p99):
    p99 = result.get('p99_ms')
    solo_p99 = paced_solo.get('p99_ms')
    row = {'status': result.get('status'), 'driver': result.get('driver'), 'why': result.get('why'),
           'n': result.get('n'), 'p50_ms': result.get('p50_ms'), 'p99_ms': p99,
           'max_ms': result.get('max_ms'),
           'deadline_ms': frame['deadline_ms'], 'hz': frame['hz'], 'achieved_hz': result.get('achieved_hz'),
           'miss_frac': result.get('miss_frac'), 'overrun_frac': result.get('overrun_frac'),
           'stream_priority': result.get('stream_priority'), 'mps_pct': result.get('mps_pct'),
           'device': result.get('device'),
           'paced_solo_p99_ms': solo_p99, 'paced_solo_p50_ms': paced_solo.get('p50_ms'),
           'paced_solo_miss_frac': paced_solo.get('miss_frac'),
           'stage4_solo_p99_ms': frame['solo'].get('latency_ms'),
           'contention_factor': round(p99 / solo_p99, 3) if (p99 and solo_p99) else None,
           'p99_contended_pred': predicted_p99,
           'pred_error_frac': round((predicted_p99 - p99) / p99, 3) if (predicted_p99 and p99) else None,
           'done_before_next_trigger_frac': makespan_row.get('done_before_next_trigger_frac'),
           'gpu_ms_p99_trace': makespan_row.get('gpu_ms_p99'),
           'fits': None, 'unfit': []}
    miss_frac = row['miss_frac']
    done_frac = row['done_before_next_trigger_frac']
    if miss_frac is not None and miss_frac >= MISS_MAX:
        row['unfit'].append(f"deadline misses {fmt_pct(miss_frac)}")
    if done_frac is not None and done_frac < DONE_MIN:
        row['unfit'].append(f"done before next trigger {fmt_pct(done_frac)}")
    if done_frac is None and miss_frac is None:
        row['unfit'].append('no frame statistics')
    row['fits'] = not row['unfit']
    return row


def _side_overlap_reason(name, result):
    if not result:
        return None
    if result.get('window_clips') == 0:
        return f'{name}: no side clip started inside the frame window (side load did not overlap the rows)'
    if result.get('status') == 'OK' and not any(key.startswith('window_') for key in result):
        return (f'{name}: side overlap unverified - the side result carries no window statistics '
                f'(harness predates windowing; re-run the arm)')
    if result.get('window_requests') == 0:
        return f'{name}: no side request started inside the frame window (side load did not overlap the rows)'
    return None


def _side_row(name, side, result, solo, composed):
    result_or_empty = result or {}
    row = {'runtime': side.get('runtime'), 'status': result_or_empty.get('status', 'missing'),
           'why': result_or_empty.get('why'), 'summary': result_or_empty.get('summary'),
           'solo_summary': (solo or {}).get('summary'), 'vs_solo': side_ratios(result, solo),
           'result': result, 'fits': None, 'unfit': []}
    if row['status'] != 'OK':
        row['unfit'].append(f"side harness {row['status']}: {row['why']}")
    rtf = result_or_empty.get('window_rtf_wall_median', result_or_empty.get('rtf_wall_median'))
    if rtf is not None and rtf >= 1:
        row['unfit'].append(f'RTF {rtf:.2f} >= 1 (falls behind real time)')
    # the side row's contended L: composed L (deadline / stage-4 solo figure) scaled by the measured
    # contended/solo ratio of its per-request latency (generative) or decode step (ASR) - the same row set
    # N_predicted is built from
    composed_row = next((item for item in (composed or {}).get('rows', [])
                         if item.get('row') == name or item.get('name') == name), None)
    composed_l = ((composed_row or {}).get('metrics') or {}).get('L')
    ratio = None
    basis = None
    if result and solo:
        for contended_key, solo_key, what in SIDE_RATIO_BASES:
            if result.get(contended_key) and solo.get(solo_key):
                ratio = result[contended_key] / solo[solo_key]
                basis = what
                break
    if composed_l and ratio:
        row['L_contended'] = composed_l / ratio
        row['L_basis'] = f'composed L {composed_l:.2f} / contended-over-solo {ratio:.2f} ({basis})'
    row['fits'] = not row['unfit']
    return row


def _cell_validity_reasons(arm, concurrent, makespan, drift):
    reasons = []
    if concurrent.get('status') != 'OK':
        reasons.append(f"arm status {concurrent.get('status')}")
    if drift.get('verdict') not in ('PASS', 'WARN'):
        reasons.append(f"clock drift {drift.get('verdict') or 'unrecorded'}")
    if makespan.get('aligned') is False:
        reasons.append('rows launched on different phases (aligned:false) - not a shared-trigger measurement')
    if makespan.get('rows_missing'):
        reasons.append(f"no trace for {makespan['rows_missing']}")
    if arm == 'mps' and not concurrent.get('arm', {}).get('mps_verified'):
        reasons.append('MPS daemon not verified')
    return reasons


def _cause(n_measured, l_contended, c_composed):
    if n_measured >= 1:
        return None
    return 'latency-limited' if l_contended <= c_composed else 'throughput-limited'


def judge_cell(run, mix, arm, resolved, composed):
    cell_dir = os.path.join(run, mix, arm)
    concurrent = load_json(os.path.join(cell_dir, 'concurrent_mix.json'))
    cell = {'mix': mix, 'arm': arm, 'dir': cell_dir, 'status': None, 'valid': False, 'fits': None,
            'invalid_reasons': [], 'unfit_reasons': [], 'rows': {}, 'side_rows': {}, 'makespan': None,
            'drift': None, 'N_measured': None, 'N_predicted': (composed or {}).get('composed', {}).get('N')}
    if concurrent is None:
        cell['status'] = 'missing'
        cell['invalid_reasons'].append('no concurrent_mix.json (arm never ran)')
        return cell
    cell['status'] = concurrent.get('status')
    if concurrent.get('status') == 'unsupported':
        cell['invalid_reasons'].append(f"unsupported: {concurrent.get('why')}")
        return cell
    makespan = load_json(os.path.join(cell_dir, 'period_makespan.json')) or {}
    drift = load_json(os.path.join(cell_dir, 'clock_drift.json')) or {}
    cell['drift'] = drift.get('verdict')
    cell['makespan'] = {key: makespan.get(key) for key in MAKESPAN_KEYS}
    cell['invalid_reasons'] += _cell_validity_reasons(arm, concurrent, makespan, drift)

    frames = {row['name']: row for row in resolved['rows'] if row['role'] == 'frame'}
    sides = {row['name']: row for row in resolved['rows'] if row['role'] == 'side'}
    composed_metrics = {row['row']: row['metrics'] for row in (composed or {}).get('rows', [])}
    solo_dir = os.path.join(run, 'paced_solo')
    results = {row['name']: row for row in concurrent.get('rows', []) if row.get('name') in frames}
    l_values = []
    for name, frame in frames.items():
        result = results.get(name)
        if not result:
            cell['invalid_reasons'].append(f'{name}: no row result')
            continue
        paced_solo = load_json(os.path.join(solo_dir, f"{name}@{frame['hz']:g}.json")) or {}
        makespan_row = (makespan.get('rows') or {}).get(name) or {}
        row = _frame_row(frame, result, paced_solo, makespan_row,
                         composed_metrics.get(name, {}).get('p99_contended_pred'))
        if result.get('driver') != 'row_loop_cpp' or result.get('status') != 'OK':
            cell['invalid_reasons'].append(f"{name}: driver {result.get('driver')} "
                                           f"status {result.get('status')}: {result.get('why')}")
        if not row['paced_solo_p99_ms']:
            cell['invalid_reasons'].append(f"{name}: paced solo missing ({name}@{frame['hz']:g}.json) - "
                                           f"contention factor undefined")
        if row['p99_ms']:
            l_values.append(frame['deadline_ms'] / row['p99_ms'])
        cell['rows'][name] = row
    for name in concurrent.get('side_exited_before_start') or []:
        cell['invalid_reasons'].append(f'{name}: side load exited before the frame rows started '
                                       f'(never overlapped - check its readiness signal)')
    for name, side in sides.items():
        result = load_json(os.path.join(cell_dir, 'side', name, 'side_result.json'))
        overlap_reason = _side_overlap_reason(name, result)
        if overlap_reason:
            cell['invalid_reasons'].append(overlap_reason)
        solo = load_json(os.path.join(solo_dir, f'side_{name}', 'side_result.json'))
        row = _side_row(name, side, result, solo, composed)
        if 'L_contended' in row:
            l_values.append(row['L_contended'])
        cell['side_rows'][name] = row

    unfit = ([f'{name}: {reason}' for name, row in cell['rows'].items() for reason in row['unfit']]
             + [f'{name}: {reason}' for name, row in cell['side_rows'].items() for reason in row['unfit']])
    cell['unfit_reasons'] = unfit
    cell['valid'] = not cell['invalid_reasons']
    cell['fits'] = (not unfit) if cell['valid'] else None
    composed_summary = (composed or {}).get('composed', {})
    period_ms = composed_summary.get('period_ms')
    solo_p99s = [row['paced_solo_p99_ms'] for row in cell['rows'].values() if row['paced_solo_p99_ms']]
    cell['period_ms'] = period_ms
    every_row_has_solo = len(solo_p99s) == len(cell['rows']) and bool(solo_p99s)
    cell['sum_paced_solo_p99_ms'] = sum(solo_p99s) if every_row_has_solo else None
    cell['free_time_ms'] = ((period_ms - cell['sum_paced_solo_p99_ms'])
                            if (period_ms and cell['sum_paced_solo_p99_ms'] is not None) else None)
    cell['makespan_p99_over_period'] = ((makespan['makespan_p99_ms'] / period_ms)
                                        if (makespan.get('makespan_p99_ms') and period_ms) else None)
    cell['L_contended'] = min(l_values) if l_values else None
    cell['C_composed'] = composed_summary.get('C')
    if cell['valid'] and cell['L_contended'] and cell['C_composed']:
        cell['N_measured'] = min(cell['L_contended'], cell['C_composed'])
        cell['cause'] = _cause(cell['N_measured'], cell['L_contended'], cell['C_composed'])
    worst_name, worst_row = max(cell['rows'].items(), key=lambda item: (item[1]['contention_factor'] or 0),
                                default=(None, None))
    cell['worst_row'] = worst_name
    cell['worst_contention_factor'] = worst_row['contention_factor'] if worst_row else None
    cell['stream_priority_range'] = concurrent.get('arm', {}).get('stream_priority_range')
    cell['seconds'] = concurrent.get('seconds')
    return cell


def _matrix_entry(cell):
    if not cell:
        return '-'
    if cell['status'] == 'unsupported':
        return 'unsupported'
    if not cell['valid']:
        return f"INVALID ({cell['invalid_reasons'][0][:60]})"
    return (f"{'fits' if cell['fits'] else 'MISSES'} · "
            f"makespan p99 {fmt(cell['makespan']['makespan_p99_ms'])} / {fmt(cell['period_ms'])} ms · "
            f"N {fmt(cell['N_measured'], 2)} (pred {fmt(cell['N_predicted'], 2)}) · "
            f"worst {cell['worst_row']} x{fmt(cell['worst_contention_factor'], 2)}")


def _composed_section(mix, composed):
    summary = composed.get('composed', {})
    cause = ' ' + summary['cause'] if summary.get('cause') else ''
    incomplete = ('' if summary.get('budget_complete')
                  else f"; budgets not measured: {summary.get('budgets_not_measured')}")
    lines = [f"## {mix}", "",
             f"Composed from stage-4 solo numbers: U_time {fmt(summary.get('U_time'), 3)}  "
             f"U_bw {fmt(summary.get('U_bw'), 3)}  U_vram {fmt(summary.get('U_vram'), 4)}  "
             f"-> U_max {fmt(summary.get('U_max'), 3)} ({summary.get('binding_budget')}), "
             f"C {fmt(summary.get('C'), 2)}, L {fmt(summary.get('L'), 2)}, "
             f"**N_predicted {fmt(summary.get('N'), 2)}**{cause}; "
             f"units at 65 % headroom: {summary.get('units_at_0.65')}" + incomplete, "",
             "| row | role | hz | deadline | solo p99 | time share | bw share | contended p99 pred |",
             "|---|---|---|---|---|---|---|---|"]
    for row in composed.get('rows', []):
        metrics = row['metrics']
        hz = metrics.get('hz') if metrics.get('hz') is not None else 'X'
        solo_p99 = metrics.get('p99_ms') or metrics.get('step_p99_ms') or metrics.get('decode_ms_p99')
        lines.append(f"| {row['row']} | {row['role']} | {hz} | {fmt(metrics.get('deadline_ms'))} | "
                     f"{fmt(solo_p99)} | {fmt(metrics.get('time_share'), 3)} | "
                     f"{fmt(metrics.get('bw_share'), 3)} | {fmt(metrics.get('p99_contended_pred'))} |")
    lines.append("")
    return lines


def _cell_heading(cell):
    if not cell['valid']:
        return 'INVALID - ' + '; '.join(cell['invalid_reasons'])
    return 'fits' if cell['fits'] else 'MISSES - ' + '; '.join(cell['unfit_reasons'])


def _frame_row_line(name, row):
    prio = row['stream_priority'] if row.get('stream_priority') is not None else ''
    mps = '/' + str(row['mps_pct']) + '%' if row.get('mps_pct') else ''
    lever = f"{prio}{mps}"
    fits = 'yes' if row['fits'] else 'NO: ' + '; '.join(row['unfit'])
    return (f"| {name} | {row['hz']:g} | {lever or '-'} | {fmt(row['p50_ms'])} / {fmt(row['p99_ms'])} / "
            f"{fmt(row['max_ms'])} | {fmt(row['paced_solo_p99_ms'])} | {fmt(row['contention_factor'], 2)} | "
            f"{fmt(row['p99_contended_pred'])} ({fmt_pct(row['pred_error_frac'], 0)}) | "
            f"{fmt_pct(row['miss_frac'], 2)} | {fmt_pct(row['done_before_next_trigger_frac'])} | {fits} |")


def _side_row_line(name, row):
    versus = ', '.join(f"{key} x{ratio['ratio']}" for key, ratio in row['vs_solo'].items())
    unfit = '' if row['fits'] else ' - ' + '; '.join(row['unfit'])
    return (f"| {name} | {row['runtime']} | {row['status']}{unfit} | {row['summary'] or row['why'] or '-'} | "
            f"{row['solo_summary'] or '-'} | {versus or '-'} |")


def _cell_section(mix, arm, cell):
    makespan = cell['makespan'] or {}
    cause = ' ' + cell['cause'] if cell.get('cause') else ''
    basis = f"; makespan basis: {makespan['makespan_basis']}" if makespan.get('rows_slid') else ""
    prio_range = (f"; stream priority range {cell['stream_priority_range']}"
                  if cell.get('stream_priority_range') else '')
    lines = [f"### {mix} / {arm}: {_cell_heading(cell)}", "",
             f"makespan p50 / p99 / max {fmt(makespan.get('makespan_p50_ms'))} / "
             f"{fmt(makespan.get('makespan_p99_ms'))} / {fmt(makespan.get('makespan_max_ms'))} ms "
             f"over {makespan.get('periods')} periods of {fmt(cell['period_ms'])} ms "
             f"(all rows done in period {fmt_pct(makespan.get('all_rows_done_in_period_frac'))}, "
             f"aligned {makespan.get('aligned')}, "
             f"overlap {fmt_pct(makespan.get('overlap_frac_of_period_p50'))} of the period); "
             f"free time {fmt(cell['free_time_ms'])} ms = period - sum of paced-solo p99 "
             f"({fmt(cell['sum_paced_solo_p99_ms'])} ms); drift {cell['drift']}" + basis + prio_range,
             f"N_measured {fmt(cell['N_measured'], 2)} (L_contended {fmt(cell['L_contended'], 2)}, "
             f"C_composed {fmt(cell['C_composed'], 2)}){cause}", "",
             "| frame row | hz | prio/mps | contended p50 / p99 / max | paced solo p99 | x contention | "
             "predicted p99 (err) | misses | done before next trigger | fits |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    lines += [_frame_row_line(name, row) for name, row in cell['rows'].items()]
    if cell['side_rows']:
        lines += ["", "| side row | runtime | status | contended | alone | vs solo |",
                  "|---|---|---|---|---|---|"]
        lines += [_side_row_line(name, row) for name, row in cell['side_rows'].items()]
    lines.append("")
    return lines


def report(run, provenance, mixes, cells, arms):
    lock_verdict = load_json(os.path.join(run, 'provenance', 'lock_verified.json'), {}).get('verdict')
    preflight_verdict = load_json(os.path.join(run, 'provenance', 'preflight.json'), {}).get('verdict')
    lines = [f"# Co-location report (this device)\n",
             f"- device: **{provenance.get('device')}** ({provenance.get('platform')}, "
             f"tag `{provenance.get('device_tag')}`)  date {provenance.get('date')}",
             f"- stage-4 solo results: `{provenance.get('stage4_results')}`",
             f"- run length per paced solo / arm: {provenance.get('run_seconds')} s;  "
             f"arms: {', '.join(arms)}",
             f"- clock lock: {lock_verdict};  preflight: {preflight_verdict}",
             "",
             "Vocabulary: **fits** = every frame row < 1 % deadline misses and >= 99 % done before its own "
             "next trigger, every side load OK; "
             "**invalid** = the cell cannot be read as a shared-trigger measurement (driver, alignment, "
             "trace, drift or MPS evidence missing) - reason listed. "
             "contention factor = contended p99 / paced-solo p99 (same driver, same rate, alone). "
             "N_measured = min(L_contended, C_composed).", ""]
    lines += ["## Matrix", "", "| mix | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for mix in mixes:
        entries = [_matrix_entry(cells.get((mix, arm))) for arm in arms]
        lines.append(f"| {mix} | " + " | ".join(entries) + " |")
    lines.append("")
    for mix in mixes:
        lines += _composed_section(mix, load_json(os.path.join(run, 'composed', f'{mix}.json')) or {})
        for arm in arms:
            cell = cells.get((mix, arm))
            if not cell or cell['status'] in ('unsupported', 'missing'):
                continue
            lines += _cell_section(mix, arm, cell)
    return '\n'.join(lines) + '\n'


def _cell_summary_line(cell):
    if cell['status'] in ('unsupported', 'missing'):
        tag = cell['status']
    elif not cell['valid']:
        tag = 'INVALID: ' + cell['invalid_reasons'][0]
    elif cell['fits']:
        tag = 'fits'
    else:
        tag = 'MISSES: ' + '; '.join(cell['unfit_reasons'])[:120]
    numbers = ''
    if cell['valid']:
        makespan_p99 = (cell['makespan'] or {}).get('makespan_p99_ms')
        numbers = (f"   makespan p99 {fmt(makespan_p99)}/{fmt(cell['period_ms'])} ms"
                   f"  N {fmt(cell['N_measured'], 2)} (pred {fmt(cell['N_predicted'], 2)})")
    return f"  {cell['mix']:22} {cell['arm']:8} {tag}" + numbers


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    run = os.path.abspath(args.run)
    provenance = load_json(os.path.join(run, 'provenance', 'provenance.json'), {})
    concurrent_files = glob.glob(os.path.join(run, '*', '*', 'concurrent_mix.json'))
    arms = provenance.get('arms') or sorted({os.path.basename(os.path.dirname(path)) for path in concurrent_files})
    resolved_files = glob.glob(os.path.join(run, 'resolved', '*.json'))
    mixes = provenance.get('mixes') or sorted(os.path.basename(path)[:-5] for path in resolved_files)
    cells = {}
    for mix in mixes:
        resolved = load_json(os.path.join(run, 'resolved', f'{mix}.json'))
        if not resolved:
            continue
        composed = load_json(os.path.join(run, 'composed', f'{mix}.json'))
        for arm in arms:
            cells[(mix, arm)] = judge_cell(run, mix, arm, resolved, composed)
    ordered = sorted(cells, key=lambda key: (mixes.index(key[0]), arms.index(key[1])))
    lock_verdict = load_json(os.path.join(run, 'provenance', 'lock_verified.json'), {}).get('verdict')
    out = {'run': run, 'date': datetime.datetime.now().isoformat(timespec='seconds'),
           'device': provenance.get('device'), 'device_tag': provenance.get('device_tag'),
           'platform': provenance.get('platform'), 'stage4_results': provenance.get('stage4_results'),
           'run_seconds': provenance.get('run_seconds'), 'arms': arms, 'mixes': mixes,
           'lock_verified': lock_verdict,
           'rules': {'miss_frac_max': MISS_MAX, 'done_before_next_trigger_min': DONE_MIN},
           'cells': [cells[key] for key in ordered]}
    json.dump(out, open(os.path.join(run, 'verdict.json'), 'w'), indent=1)
    open(os.path.join(run, 'report.md'), 'w').write(report(run, provenance, mixes, cells, arms))
    for cell in out['cells']:
        print(_cell_summary_line(cell))
    print(f"verdict: {os.path.join(run, 'verdict.json')}  report: {os.path.join(run, 'report.md')}")


if __name__ == '__main__':
    main()
