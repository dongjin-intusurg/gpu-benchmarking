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
import argparse, glob, json, os, sys, datetime

MISS_MAX = 0.01; DONE_MIN = 0.99


def jload(p, default=None):
    try:
        return json.load(open(p))
    except Exception:
        return default


def pct(v, d=1):
    return '-' if v is None else f'{100 * v:.{d}f}%'


def f1(v, d=1):
    return '-' if v is None else f'{v:.{d}f}'


def side_ratios(res, solo):
    """side-row contention: harness numbers vs the same harness alone. The ASR harness also reports
    window_* statistics over the clips that started inside the frame window; those are the contended
    numbers when present (the whole pass outlasts the rows by design), ratioed against the solo pass."""
    out = {}
    for k in ('ttft_ms_median', 'decode_ms_p99', 'decode_ms_median', 'rtf_wall_median', 'tokens_per_second', 'step_ms_p99', 'gpu_total_ms_p99'):
        a = (res or {}).get('window_' + k, (res or {}).get(k)); b = (solo or {}).get(k)
        if a is not None and b:
            out[k] = {'contended': a, 'solo': b, 'ratio': round(a / b, 3), 'basis': 'frame window' if ('window_' + k) in (res or {}) else 'whole pass'}
    return out


def judge_cell(run, mix, arm, resolved, composed):
    d = os.path.join(run, mix, arm)
    cm = jload(os.path.join(d, 'concurrent_mix.json'))
    cell = {'mix': mix, 'arm': arm, 'dir': d, 'status': None, 'valid': False, 'fits': None, 'invalid_reasons': [], 'unfit_reasons': [],
            'rows': {}, 'side_rows': {}, 'makespan': None, 'drift': None, 'N_measured': None, 'N_predicted': (composed or {}).get('composed', {}).get('N')}
    if cm is None:
        cell['status'] = 'missing'; cell['invalid_reasons'].append('no concurrent_mix.json (arm never ran)'); return cell
    cell['status'] = cm.get('status')
    if cm.get('status') == 'unsupported':
        cell['invalid_reasons'].append(f"unsupported: {cm.get('why')}"); return cell
    if cm.get('status') != 'OK':
        cell['invalid_reasons'].append(f"arm status {cm.get('status')}")
    mk = jload(os.path.join(d, 'period_makespan.json')) or {}
    dr = jload(os.path.join(d, 'clock_drift.json')) or {}
    cell['drift'] = dr.get('verdict'); cell['makespan'] = {k: mk.get(k) for k in ('period_ms', 'periods', 'aligned', 'launch_phase_ms', 'makespan_p50_ms', 'makespan_p99_ms', 'makespan_max_ms',
                                                                                'all_rows_done_in_period_frac', 'overlap_frac_of_period_p50', 'rows_missing', 'rows_slid', 'makespan_basis')}
    if dr.get('verdict') not in ('PASS', 'WARN'):
        cell['invalid_reasons'].append(f"clock drift {dr.get('verdict') or 'unrecorded'}")
    if mk.get('aligned') is False:
        cell['invalid_reasons'].append('rows launched on different phases (aligned:false) - not a shared-trigger measurement')
    if mk.get('rows_missing'):
        cell['invalid_reasons'].append(f"no trace for {mk['rows_missing']}")
    if arm == 'mps' and not cm.get('arm', {}).get('mps_verified'):
        cell['invalid_reasons'].append('MPS daemon not verified')
    frames = {r['name']: r for r in resolved['rows'] if r['role'] == 'frame'}
    sides = {r['name']: r for r in resolved['rows'] if r['role'] == 'side'}
    cmet = {r['row']: r['metrics'] for r in (composed or {}).get('rows', [])}
    solo_dir = os.path.join(run, 'paced_solo')
    got = {r['name']: r for r in cm.get('rows', []) if r.get('name') in frames}
    Ls = []
    for n, fr in frames.items():
        r = got.get(n)
        if not r:
            cell['invalid_reasons'].append(f'{n}: no row result'); continue
        ps = jload(os.path.join(solo_dir, f"{n}@{fr['hz']:g}.json")) or {}
        mrow = (mk.get('rows') or {}).get(n) or {}
        p99 = r.get('p99_ms'); sp = ps.get('p99_ms'); pred = cmet.get(n, {}).get('p99_contended_pred')
        row = {'status': r.get('status'), 'driver': r.get('driver'), 'why': r.get('why'), 'n': r.get('n'), 'p50_ms': r.get('p50_ms'), 'p99_ms': p99, 'max_ms': r.get('max_ms'),
               'deadline_ms': fr['deadline_ms'], 'hz': fr['hz'], 'achieved_hz': r.get('achieved_hz'), 'miss_frac': r.get('miss_frac'), 'overrun_frac': r.get('overrun_frac'),
               'stream_priority': r.get('stream_priority'), 'mps_pct': r.get('mps_pct'), 'device': r.get('device'),
               'paced_solo_p99_ms': sp, 'paced_solo_p50_ms': ps.get('p50_ms'), 'paced_solo_miss_frac': ps.get('miss_frac'), 'stage4_solo_p99_ms': fr['solo'].get('latency_ms'),
               'contention_factor': round(p99 / sp, 3) if (p99 and sp) else None,
               'p99_contended_pred': pred, 'pred_error_frac': round((pred - p99) / p99, 3) if (pred and p99) else None,
               'done_before_next_trigger_frac': mrow.get('done_before_next_trigger_frac'), 'gpu_ms_p99_trace': mrow.get('gpu_ms_p99'),
               'fits': None, 'unfit': []}
        if r.get('driver') != 'row_loop_cpp' or r.get('status') != 'OK':
            cell['invalid_reasons'].append(f"{n}: driver {r.get('driver')} status {r.get('status')}: {r.get('why')}")
        if not sp:
            cell['invalid_reasons'].append(f"{n}: paced solo missing ({n}@{fr['hz']:g}.json) - contention factor undefined")
        if row['miss_frac'] is not None and row['miss_frac'] >= MISS_MAX: row['unfit'].append(f"deadline misses {pct(row['miss_frac'])}")
        if row['done_before_next_trigger_frac'] is not None and row['done_before_next_trigger_frac'] < DONE_MIN: row['unfit'].append(f"done before next trigger {pct(row['done_before_next_trigger_frac'])}")
        if row['done_before_next_trigger_frac'] is None and row['miss_frac'] is None: row['unfit'].append('no frame statistics')
        row['fits'] = not row['unfit']
        if p99: Ls.append(fr['deadline_ms'] / p99)
        cell['rows'][n] = row
    for n in (cm or {}).get('side_exited_before_start') or []:
        cell['invalid_reasons'].append(f'{n}: side load exited before the frame rows started (never overlapped - check its readiness signal)')
    for n, sr in sides.items():
        res = jload(os.path.join(d, 'side', n, 'side_result.json'))
        if res and res.get('window_clips') == 0:
            cell['invalid_reasons'].append(f'{n}: no side clip started inside the frame window (side load did not overlap the rows)')
        elif res and res.get('status') == 'OK' and not any(k.startswith('window_') for k in res):
            cell['invalid_reasons'].append(f'{n}: side overlap unverified - the side result carries no window statistics (harness predates windowing; re-run the arm)')
        elif res and res.get('window_requests') == 0:
            cell['invalid_reasons'].append(f'{n}: no side request started inside the frame window (side load did not overlap the rows)')
        solo = jload(os.path.join(solo_dir, f'side_{n}', 'side_result.json'))
        row = {'runtime': sr.get('runtime'), 'status': (res or {}).get('status', 'missing'), 'why': (res or {}).get('why'), 'summary': (res or {}).get('summary'),
               'solo_summary': (solo or {}).get('summary'), 'vs_solo': side_ratios(res, solo), 'result': res, 'fits': None, 'unfit': []}
        if row['status'] != 'OK': row['unfit'].append(f"side harness {row['status']}: {row['why']}")
        rtf = (res or {}).get('window_rtf_wall_median', (res or {}).get('rtf_wall_median'))
        if rtf is not None and rtf >= 1: row['unfit'].append(f'RTF {rtf:.2f} >= 1 (falls behind real time)')
        # the side row's contended L: composed L (deadline / stage-4 solo figure) scaled by the measured contended/solo ratio of
        # its per-request latency (generative) or decode step (ASR) - the same row set N_predicted is built from
        cm_row = next((x for x in (composed or {}).get('rows', []) if x.get('row') == n or x.get('name') == n), None)
        L0 = ((cm_row or {}).get('metrics') or {}).get('L'); ratio = None; basis = None
        if res and solo:
            for a_key, b_key, what in (('window_request_ms_mean', 'request_ms_mean', 'per-request wall mean (the battery is a fixed prompt set, so the mean is its stable summary)'), ('request_ms_mean', 'request_ms_mean', 'per-request wall mean (whole battery)'),
                                       ('window_decode_ms_p99_max', 'decode_ms_p99_max', 'decode p99 max'), ('decode_ms_p99_max', 'decode_ms_p99_max', 'decode p99 max (whole pass)')):
                if res.get(a_key) and solo.get(b_key):
                    ratio = res[a_key] / solo[b_key]; basis = what; break
        if L0 and ratio:
            row['L_contended'] = L0 / ratio; row['L_basis'] = f'composed L {L0:.2f} / contended-over-solo {ratio:.2f} ({basis})'; Ls.append(row['L_contended'])
        row['fits'] = not row['unfit']
        cell['side_rows'][n] = row
    unfit = [f'{n}: {u}' for n, r in cell['rows'].items() for u in r['unfit']] + [f'{n}: {u}' for n, r in cell['side_rows'].items() for u in r['unfit']]
    cell['unfit_reasons'] = unfit
    cell['valid'] = not cell['invalid_reasons']
    cell['fits'] = (not unfit) if cell['valid'] else None
    comp = (composed or {}).get('composed', {})
    per = comp.get('period_ms'); sps = [cell['rows'][n]['paced_solo_p99_ms'] for n in cell['rows'] if cell['rows'][n]['paced_solo_p99_ms']]
    cell['period_ms'] = per
    cell['sum_paced_solo_p99_ms'] = sum(sps) if len(sps) == len(cell['rows']) and sps else None
    cell['free_time_ms'] = (per - cell['sum_paced_solo_p99_ms']) if (per and cell['sum_paced_solo_p99_ms'] is not None) else None
    cell['makespan_p99_over_period'] = (mk['makespan_p99_ms'] / per) if (mk.get('makespan_p99_ms') and per) else None
    cell['L_contended'] = min(Ls) if Ls else None; cell['C_composed'] = comp.get('C')
    if cell['valid'] and cell['L_contended'] and cell['C_composed']:
        cell['N_measured'] = min(cell['L_contended'], cell['C_composed'])
        cell['cause'] = None if cell['N_measured'] >= 1 else ('latency-limited' if cell['L_contended'] <= cell['C_composed'] else 'throughput-limited')
    worst = max(cell['rows'].items(), key=lambda kv: (kv[1]['contention_factor'] or 0), default=(None, None))
    cell['worst_row'] = worst[0]; cell['worst_contention_factor'] = worst[1]['contention_factor'] if worst[1] else None
    cell['stream_priority_range'] = cm.get('arm', {}).get('stream_priority_range'); cell['seconds'] = cm.get('seconds')
    return cell


def report(run, prov, mixes, cells, arms):
    L = [f"# Co-location report (this device)\n", f"- device: **{prov.get('device')}** ({prov.get('platform')}, tag `{prov.get('device_tag')}`)  date {prov.get('date')}",
         f"- stage-4 solo results: `{prov.get('stage4_results')}`", f"- run length per paced solo / arm: {prov.get('run_seconds')} s;  arms: {', '.join(arms)}",
         f"- clock lock: {jload(os.path.join(run, 'provenance', 'lock_verified.json'), {}).get('verdict')};  preflight: {jload(os.path.join(run, 'provenance', 'preflight.json'), {}).get('verdict')}",
         "", "Vocabulary: **fits** = every frame row < 1 % deadline misses and >= 99 % done before its own next trigger, every side load OK; "
         "**invalid** = the cell cannot be read as a shared-trigger measurement (driver, alignment, trace, drift or MPS evidence missing) - reason listed. "
         "contention factor = contended p99 / paced-solo p99 (same driver, same rate, alone). N_measured = min(L_contended, C_composed).", ""]
    L += ["## Matrix", "", "| mix | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for m in mixes:
        cs = []
        for a in arms:
            c = cells.get((m, a))
            if not c: cs.append('-'); continue
            if c['status'] == 'unsupported': cs.append('unsupported'); continue
            if not c['valid']: cs.append(f"INVALID ({c['invalid_reasons'][0][:60]})"); continue
            cs.append(f"{'fits' if c['fits'] else 'MISSES'} · makespan p99 {f1(c['makespan']['makespan_p99_ms'])} / {f1(c['period_ms'])} ms · N {f1(c['N_measured'], 2)} (pred {f1(c['N_predicted'], 2)}) · worst {c['worst_row']} x{f1(c['worst_contention_factor'], 2)}")
        L.append(f"| {m} | " + " | ".join(cs) + " |")
    L.append("")
    for m in mixes:
        comp = jload(os.path.join(run, 'composed', f'{m}.json')) or {}; c = comp.get('composed', {})
        L += [f"## {m}", "", f"Composed from stage-4 solo numbers: U_time {f1(c.get('U_time'), 3)}  U_bw {f1(c.get('U_bw'), 3)}  U_vram {f1(c.get('U_vram'), 4)}  -> U_max {f1(c.get('U_max'), 3)} ({c.get('binding_budget')}), "
              f"C {f1(c.get('C'), 2)}, L {f1(c.get('L'), 2)}, **N_predicted {f1(c.get('N'), 2)}**{' ' + c['cause'] if c.get('cause') else ''}; units at 65 % headroom: {c.get('units_at_0.65')}"
              + ('' if c.get('budget_complete') else f"; budgets not measured: {c.get('budgets_not_measured')}"), "",
              "| row | role | hz | deadline | solo p99 | time share | bw share | contended p99 pred |", "|---|---|---|---|---|---|---|---|"]
        for r in comp.get('rows', []):
            x = r['metrics']
            L.append(f"| {r['row']} | {r['role']} | {x.get('hz') if x.get('hz') is not None else 'X'} | {f1(x.get('deadline_ms'))} | {f1(x.get('p99_ms') or x.get('step_p99_ms') or x.get('decode_ms_p99'))} | {f1(x.get('time_share'), 3)} | {f1(x.get('bw_share'), 3)} | {f1(x.get('p99_contended_pred'))} |")
        L.append("")
        for a in arms:
            c = cells.get((m, a))
            if not c or c['status'] in ('unsupported', 'missing'): continue
            head = 'INVALID - ' + '; '.join(c['invalid_reasons']) if not c['valid'] else ('fits' if c['fits'] else 'MISSES - ' + '; '.join(c['unfit_reasons']))
            mk = c['makespan'] or {}
            L += [f"### {m} / {a}: {head}", "",
                  f"makespan p50 / p99 / max {f1(mk.get('makespan_p50_ms'))} / {f1(mk.get('makespan_p99_ms'))} / {f1(mk.get('makespan_max_ms'))} ms over {mk.get('periods')} periods of {f1(c['period_ms'])} ms "
                  f"(all rows done in period {pct(mk.get('all_rows_done_in_period_frac'))}, aligned {mk.get('aligned')}, overlap {pct(mk.get('overlap_frac_of_period_p50'))} of the period); "
                  f"free time {f1(c['free_time_ms'])} ms = period - sum of paced-solo p99 ({f1(c['sum_paced_solo_p99_ms'])} ms); drift {c['drift']}"
                  + (f"; makespan basis: {mk['makespan_basis']}" if mk.get('rows_slid') else "")
                  + (f"; stream priority range {c['stream_priority_range']}" if c.get('stream_priority_range') else ''),
                  f"N_measured {f1(c['N_measured'], 2)} (L_contended {f1(c['L_contended'], 2)}, C_composed {f1(c['C_composed'], 2)}){' ' + c['cause'] if c.get('cause') else ''}", "",
                  "| frame row | hz | prio/mps | contended p50 / p99 / max | paced solo p99 | x contention | predicted p99 (err) | misses | done before next trigger | fits |", "|---|---|---|---|---|---|---|---|---|---|"]
            for n, r in c['rows'].items():
                lever = f"{r['stream_priority'] if r.get('stream_priority') is not None else ''}{'/' + str(r['mps_pct']) + '%' if r.get('mps_pct') else ''}"
                L.append(f"| {n} | {r['hz']:g} | {lever or '-'} | {f1(r['p50_ms'])} / {f1(r['p99_ms'])} / {f1(r['max_ms'])} | {f1(r['paced_solo_p99_ms'])} | {f1(r['contention_factor'], 2)} | {f1(r['p99_contended_pred'])} ({pct(r['pred_error_frac'], 0)}) | {pct(r['miss_frac'], 2)} | {pct(r['done_before_next_trigger_frac'])} | {'yes' if r['fits'] else 'NO: ' + '; '.join(r['unfit'])} |")
            if c['side_rows']:
                L += ["", "| side row | runtime | status | contended | alone | vs solo |", "|---|---|---|---|---|---|"]
                for n, r in c['side_rows'].items():
                    vs = ', '.join(f"{k} x{v['ratio']}" for k, v in r['vs_solo'].items())
                    L.append(f"| {n} | {r['runtime']} | {r['status']}{'' if r['fits'] else ' - ' + '; '.join(r['unfit'])} | {r['summary'] or r['why'] or '-'} | {r['solo_summary'] or '-'} | {vs or '-'} |")
            L.append("")
    return '\n'.join(L) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True); a = ap.parse_args()
    run = os.path.abspath(a.run)
    prov = jload(os.path.join(run, 'provenance', 'provenance.json'), {})
    arms = prov.get('arms') or sorted({os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(run, '*', '*', 'concurrent_mix.json'))})
    mixes = prov.get('mixes') or sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(run, 'resolved', '*.json')))
    cells = {}
    for m in mixes:
        resolved = jload(os.path.join(run, 'resolved', f'{m}.json'))
        if not resolved: continue
        composed = jload(os.path.join(run, 'composed', f'{m}.json'))
        for arm in arms:
            cells[(m, arm)] = judge_cell(run, m, arm, resolved, composed)
    out = {'run': run, 'date': datetime.datetime.now().isoformat(timespec='seconds'), 'device': prov.get('device'), 'device_tag': prov.get('device_tag'), 'platform': prov.get('platform'),
           'stage4_results': prov.get('stage4_results'), 'run_seconds': prov.get('run_seconds'), 'arms': arms, 'mixes': mixes,
           'lock_verified': jload(os.path.join(run, 'provenance', 'lock_verified.json'), {}).get('verdict'), 'rules': {'miss_frac_max': MISS_MAX, 'done_before_next_trigger_min': DONE_MIN},
           'cells': [cells[k] for k in sorted(cells, key=lambda k: (mixes.index(k[0]), arms.index(k[1])))]}
    json.dump(out, open(os.path.join(run, 'verdict.json'), 'w'), indent=1)
    open(os.path.join(run, 'report.md'), 'w').write(report(run, prov, mixes, cells, arms))
    for c in out['cells']:
        tag = c['status'] if c['status'] in ('unsupported', 'missing') else ('INVALID: ' + c['invalid_reasons'][0] if not c['valid'] else ('fits' if c['fits'] else 'MISSES: ' + '; '.join(c['unfit_reasons'])[:120]))
        print(f"  {c['mix']:22} {c['arm']:8} {tag}" + (f"   makespan p99 {f1((c['makespan'] or {}).get('makespan_p99_ms'))}/{f1(c['period_ms'])} ms  N {f1(c['N_measured'], 2)} (pred {f1(c['N_predicted'], 2)})" if c['valid'] else ''))
    print(f"verdict: {os.path.join(run, 'verdict.json')}  report: {os.path.join(run, 'report.md')}")


if __name__ == '__main__':
    main()
