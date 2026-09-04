#!/usr/bin/env python3
"""Stage-6 verdict: one row per operating point, every number measured at that
point, judged against the baseline point of the same run.

  power_verdict.py --run <results/power_<tag>_<stamp>> [--out verdict.json] [--report report.md]

Per point (<run>/<point>/):
  readback.json                      the knob as read back (mode name / power limit / caps)
  device_config.json                 the derived config the point was measured against
  coloc/paced_solo/<row>@<hz>.json   paced solo per frame row (+ <row>@<hz>/clock_samples.csv)
                                     -> p99, miss, mean W, J/frame = mean W x wall / frames,
                                        marginal J/frame = (mean W - idle W) x wall / frames
  coloc/verdict.json                 the stage-5 cells at this point (fits, makespan, N_measured)
                                     + <mix>/<arm>/clock_samples.csv -> mean W, J per period
  per_watt/power_tops_sweep.json     W-vs-throughput fits per precision, 50 % / 100 % points
Ratios vs baseline: p99, N_measured, mean W, J/frame, throughput per W.

Checks (recorded, never silently applied):
  C1 the baseline point exists and completed;  C2 every point's read-back matched
  its request;  C3 lock verified at the point's own caps;  C4 per-watt fits R2 >= 0.95
  over the fitted points (else WARN: the point is not linear in that precision);
  C5 idle power at the point within 1.5 x the config expectation (preflight W3).
Writes verdict.json + report.md, prints the report and a final `verdict <PASS|WARN|FAIL>: <path>` line.
"""
import argparse
import csv
import glob
import json
import os
from datetime import datetime

POWER_COLS = ('module_w', 'power_w')
CLOCK_COLS = ('gpu_mhz', 'sm_mhz')
THROTTLE_REASON_MASK = 0xEC   # SW power cap | HW slowdown | SW thermal | HW thermal | HW power brake
FIT_MIN_R2 = 0.95
IDLE_POWER_TOLERANCE = 1.5    # x the config's idle_power_w_expected (preflight W3)


def load_json(path, default=None):
    try:
        return json.load(open(path))
    except Exception:
        return default


def mean(values):
    return (sum(values) / len(values)) if values else None


def sampler_stats(csv_path):
    """Mean power / clock / temperature / throttle events over a sampler CSV (Jetson or discrete header)."""
    if not os.path.isfile(csv_path):
        return None
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return None

    def column(names):
        """First of `names` present in the header, with its parseable float values."""
        for name in names:
            if name not in rows[0]:
                continue
            values = []
            for row in rows:
                try:
                    values.append(float(row[name]))
                except (TypeError, ValueError):
                    pass
            return name, values
        return None, []

    power_field, power = column(POWER_COLS)
    _, clock = column(CLOCK_COLS)
    _, gpu_rail = column(('vdd_gpu_w',))
    _, temperature = column(('tj_c', 'temp_c'))
    _, oc_events = column(('oc_event_count',))
    _, throttle_reasons = column(('throttle_reasons_hex',))
    t_s = [float(row['t_s']) for row in rows if row.get('t_s')]
    stats = dict(n=len(rows), duration_s=(t_s[-1] - t_s[0]) if len(t_s) > 1 else 0.0, power_field=power_field,
                 mean_w=mean(power), max_w=max(power) if power else None,
                 vdd_gpu_mean_w=mean(gpu_rail),
                 clock_mhz_mean=mean(clock), clock_mhz_min=min(clock) if clock else None,
                 temp_c_mean=mean(temperature), temp_c_max=max(temperature) if temperature else None,
                 oc_events=(oc_events[-1] - oc_events[0]) if len(oc_events) > 1 else None)
    if throttle_reasons:
        throttled = sum(1 for row in rows
                        if int((row.get('throttle_reasons_hex') or '0x0'), 16) & THROTTLE_REASON_MASK)
        stats['throttled_frac'] = throttled / len(rows)
    return stats


def paced_solo_rows(point_dir, idle_w):
    """<row>@<hz> -> latency, power and J/frame of the paced solo measured at the point."""
    solo_dir = os.path.join(point_dir, 'coloc', 'paced_solo')
    out = {}
    for path in sorted(glob.glob(os.path.join(solo_dir, '*@*.json'))):
        key = os.path.basename(path)[:-5]
        solo = load_json(path)
        if not solo:
            continue
        stats = sampler_stats(os.path.join(solo_dir, key, 'clock_samples.csv')) or {}
        frames = solo.get('n') or 0
        wall_s = solo.get('seconds') or stats.get('duration_s') or 0
        mean_w = stats.get('mean_w')
        measured = bool(mean_w and frames and wall_s)
        drift = load_json(os.path.join(solo_dir, key, 'clock_drift.json')) or {}
        out[key] = dict(row=solo.get('name'), hz=solo.get('target_hz'), p50_ms=solo.get('p50_ms'),
                        p99_ms=solo.get('p99_ms'), miss_frac=solo.get('miss_frac'),
                        frames=frames, wall_s=wall_s, mean_w=mean_w,
                        vdd_gpu_mean_w=stats.get('vdd_gpu_mean_w'),
                        clock_mhz_min=stats.get('clock_mhz_min'), oc_events=stats.get('oc_events'),
                        throttled_frac=stats.get('throttled_frac'), temp_c_max=stats.get('temp_c_max'),
                        j_per_frame=(mean_w * wall_s / frames) if measured else None,
                        j_per_frame_marginal=((mean_w - idle_w) * wall_s / frames)
                        if (measured and idle_w is not None) else None,
                        drift=drift.get('verdict'))
    return out


def colocation_cells(point_dir):
    """<mix>/<arm> -> the stage-5 cell at the point plus the power sampled over its arm."""
    verdict = load_json(os.path.join(point_dir, 'coloc', 'verdict.json')) or {}
    out = {}
    for cell in verdict.get('cells') or []:
        samples = os.path.join(point_dir, 'coloc', cell['mix'], cell['arm'], 'clock_samples.csv')
        stats = sampler_stats(samples) or {}
        makespan = cell.get('makespan') or {}
        period_ms = cell.get('period_ms')
        out[f"{cell['mix']}/{cell['arm']}"] = dict(
            mix=cell['mix'], arm=cell['arm'], valid=cell.get('valid'), fits=cell.get('fits'),
            invalid_reasons=cell.get('invalid_reasons'),
            makespan_p99_ms=makespan.get('makespan_p99_ms'),
            all_done_frac=makespan.get('all_rows_done_in_period_frac'),
            N_measured=cell.get('N_measured'), N_predicted=cell.get('N_predicted'),
            worst_row=cell.get('worst_row'),
            rows={name: dict(p99_ms=row.get('p99_ms'), miss_frac=row.get('miss_frac'))
                  for name, row in (cell.get('rows') or {}).items()},
            mean_w=stats.get('mean_w'), vdd_gpu_mean_w=stats.get('vdd_gpu_mean_w'),
            oc_events=stats.get('oc_events'), throttled_frac=stats.get('throttled_frac'),
            j_per_period=(stats['mean_w'] * period_ms / 1e3) if (stats.get('mean_w') and period_ms) else None,
            drift=cell.get('drift'))
    return out


def per_watt_summary(point_dir):
    """Per precision: the fit to report, the knee, and the points nearest 50 % and 100 % target."""
    sweep = load_json(os.path.join(point_dir, 'per_watt', 'power_tops_sweep.json'))
    if not sweep:
        return None
    out = dict(idle_module_w=sweep.get('idle_module_w'), n=sweep.get('n'), precisions={})
    for precision, result in (sweep.get('precisions') or {}).items():
        points = result.get('points') or []

        def nearest(target_pct):
            if not points:
                return None
            return min(points, key=lambda point: abs(point['target_pct'] - target_pct), default=None)

        at50, at100 = nearest(50), nearest(100)
        out['precisions'][precision] = dict(
            unit=result.get('unit'),
            fit=result.get('fit_w_vs_tops_ge50') or result.get('fit_w_vs_tops_unsaturated'),
            fit_unsaturated=result.get('fit_w_vs_tops_unsaturated'), knee=result.get('saturation_knee'),
            at50=dict(delivered=at50['delivered_tops'], w=at50['module_w'],
                      per_w=at50['delivered_tops'] / at50['module_w']) if at50 else None,
            at100=dict(delivered=at100['delivered_tops'], w=at100['module_w'],
                       per_w=at100['delivered_tops'] / at100['module_w'], gpu_mhz=at100.get('gpu_mhz'),
                       oc=at100.get('oc_events'), j_per_unit=at100.get('j_per_unit')) if at100 else None)
    return out


def ratio(numerator, denominator):
    return (numerator / denominator) if (numerator is not None and denominator not in (None, 0)) else None


def load_point(run, name):
    """Everything measured at one point, with its C2-C5 checks."""
    point_dir = os.path.join(run, name)
    if not os.path.isdir(point_dir):
        return dict(status='MISSING'), []
    readback = load_json(os.path.join(point_dir, 'readback.json')) or {}
    status = load_json(os.path.join(point_dir, 'status.json')) or {}
    preflight = load_json(os.path.join(point_dir, 'coloc', 'provenance', 'preflight.json')) or {}
    lock = load_json(os.path.join(point_dir, 'coloc', 'provenance', 'lock_verified.json')) or {}
    config = load_json(os.path.join(point_dir, 'device_config.json')) or {}
    per_watt = per_watt_summary(point_dir)
    idle_w = preflight.get('idle_power_w_mean')
    if idle_w is None and per_watt:
        idle_w = per_watt.get('idle_module_w')
    point = dict(status=status.get('status', 'INCOMPLETE'), why=status.get('why'), readback=readback,
                 applied_mode=readback.get('mode'), lock_targets_mhz=config.get('lock_targets_mhz'),
                 lock_verdict=lock.get('verdict'), reference_clock_mhz=lock.get('reference_clock_mhz'),
                 preflight_verdict=preflight.get('verdict'), idle_w=idle_w,
                 idle_w_expected=config.get('idle_power_w_expected'),
                 paced_solo=paced_solo_rows(point_dir, idle_w), cells=colocation_cells(point_dir),
                 per_watt=per_watt)

    checks = []

    def check(check_id, level, msg):
        checks.append(dict(id=check_id, point=name, level=level, msg=msg))

    if status.get('status') not in (None, 'OK'):
        check('C2', 'FAIL', f"point did not complete: {status.get('why')}")
    applied_mode = readback.get('mode') if readback else None
    if applied_mode and config.get('platform') == 'jetson' and applied_mode != name:
        check('C2', 'FAIL', f"read-back mode {applied_mode} != point {name}")
    if lock and lock.get('verdict') not in ('PASS', 'WARN'):
        check('C3', 'FAIL', f"lock verification {lock.get('verdict')}")
    if per_watt:
        for precision, summary in per_watt['precisions'].items():
            r2 = (summary.get('fit') or {}).get('r2')
            if r2 is not None and r2 < FIT_MIN_R2:
                check('C4', 'WARN',
                      f"{precision}: W-vs-throughput fit R2 {r2:.3f} < 0.95 - not linear at this point")
    idle_expected = config.get('idle_power_w_expected')
    if idle_w is not None and idle_expected and idle_w > IDLE_POWER_TOLERANCE * idle_expected:
        check('C5', 'WARN', f"idle power {idle_w:.1f} W > 1.5 x expected {idle_expected} W")
    return point, checks


def ratios_vs_baseline(point, base):
    """Point / baseline for every paced solo, cell and precision the baseline also has."""
    base_solo, base_cells = base['paced_solo'], base['cells']
    base_per_watt = ((base['per_watt'] or {}).get('precisions') or {})

    def base_fit(precision, field):
        return (base_per_watt.get(precision, {}).get(field) or {})

    return dict(
        paced_solo={key: dict(p99_ratio=ratio(solo.get('p99_ms'), base_solo[key].get('p99_ms')),
                              mean_w_ratio=ratio(solo.get('mean_w'), base_solo[key].get('mean_w')),
                              j_per_frame_ratio=ratio(solo.get('j_per_frame'),
                                                      base_solo[key].get('j_per_frame')),
                              j_marginal_ratio=ratio(solo.get('j_per_frame_marginal'),
                                                     base_solo[key].get('j_per_frame_marginal')))
                    for key, solo in point['paced_solo'].items() if key in base_solo},
        cells={key: dict(makespan_p99_ratio=ratio(cell.get('makespan_p99_ms'),
                                                  base_cells[key].get('makespan_p99_ms')),
                         N_ratio=ratio(cell.get('N_measured'), base_cells[key].get('N_measured')),
                         mean_w_ratio=ratio(cell.get('mean_w'), base_cells[key].get('mean_w')),
                         fits_changed=(cell.get('fits') != base_cells[key].get('fits')))
               for key, cell in point['cells'].items() if key in base_cells},
        per_watt={precision: dict(
            slope_ratio=ratio((summary.get('fit') or {}).get('slope'),
                              base_fit(precision, 'fit').get('slope')),
            per_w_at100_ratio=ratio((summary.get('at100') or {}).get('per_w'),
                                    base_fit(precision, 'at100').get('per_w')),
            delivered_at100_ratio=ratio((summary.get('at100') or {}).get('delivered'),
                                        base_fit(precision, 'at100').get('delivered')))
            for precision, summary in ((point.get('per_watt') or {}).get('precisions') or {}).items()})


def point_order(run, provenance):
    if provenance.get('points'):
        return provenance['points']
    return sorted(os.path.basename(path) for path in glob.glob(os.path.join(run, '*'))
                  if os.path.isfile(os.path.join(path, 'readback.json')))


def build_verdict(run):
    provenance = load_json(os.path.join(run, 'provenance', 'provenance.json')) or {}
    order = point_order(run, provenance)
    baseline = provenance.get('baseline')
    points = {}
    checks = []
    for name in order:
        points[name], point_checks = load_point(run, name)
        checks += point_checks
    base = points.get(baseline)
    if not base or base.get('status') != 'OK':
        checks.append(dict(id='C1', point=baseline, level='FAIL',
                           msg='baseline point missing or incomplete - ratios are undefined'))
        base = None
    for name, point in points.items():
        if not base or name == baseline or point.get('status') != 'OK':
            continue
        point['vs_baseline'] = ratios_vs_baseline(point, base)
    worst = 'FAIL' if any(check['level'] == 'FAIL' for check in checks) else ('WARN' if checks else 'PASS')
    return dict(schema='power-verdict/v1', run=run, date=datetime.now().isoformat(timespec='seconds'),
                device_tag=provenance.get('device_tag'), platform=provenance.get('platform'),
                baseline=baseline, points_order=order, verdict=worst, checks=checks, points=points,
                rules=dict(j_per_frame='mean sampled power over the paced solo x wall / frames',
                           j_per_frame_marginal='(mean W - preflight idle W) x wall / frames',
                           per_watt_fit='module W vs delivered throughput, >= 45 % busy points '
                                        '(unsaturated fit when a cap pins the board)',
                           ratios='point / baseline, same row / cell / precision'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--out')
    parser.add_argument('--report')
    args = parser.parse_args()
    run = os.path.abspath(args.run)
    verdict = build_verdict(run)
    out_path = args.out or os.path.join(run, 'verdict.json')
    json.dump(verdict, open(out_path, 'w'), indent=1)
    report_path = args.report or os.path.join(run, 'report.md')
    open(report_path, 'w').write(report(verdict))
    print(open(report_path).read())
    print(f"verdict {verdict['verdict']}: {out_path}")


def fmt(x, pattern='%.2f', na='-'):
    return (pattern % x) if isinstance(x, (int, float)) and x == x else na


def table_header(first_column, point_names):
    return ['| ' + first_column + ' | ' + ' | '.join(point_names) + ' |', '|---|' + '---|' * len(point_names)]


def points_table(verdict):
    lines = ['## Points', '', '| point | applied | lock targets | lock | preflight | idle W | status |',
             '|---|---|---|---|---|---|---|']
    for name in verdict['points_order']:
        point = verdict['points'].get(name, {})
        targets = json.dumps(point.get('lock_targets_mhz')) if point.get('lock_targets_mhz') else '-'
        status = f"{point.get('status')}{(' - ' + point['why']) if point.get('why') else ''}"
        lines.append(f"| {name} | {point.get('applied_mode') or '-'} | {targets} | "
                     f"{point.get('lock_verdict') or '-'} | {point.get('preflight_verdict') or '-'} | "
                     f"{fmt(point.get('idle_w'), '%.1f')} | {status} |")
    return lines


def checks_section(checks):
    if not checks:
        return ['', 'Checks C1-C5: all PASS.']
    return ['', '## Checks', ''] + [f"- **{check['level']}** {check['id']} [{check['point']}]: {check['msg']}"
                                     for check in checks]


def paced_solo_cell(point, key):
    solo = point['paced_solo'].get(key)
    if not solo:
        return '-'
    events = solo.get('oc_events')
    if events is not None:
        events = fmt(events, '%.0f')
    elif solo.get('throttled_frac') is not None:
        events = fmt(solo.get('throttled_frac', None), '%.0f%%')
    else:
        events = '-'
    ratios = (point.get('vs_baseline') or {}).get('paced_solo', {}).get(key, {})
    text = (f"**{fmt(solo['p99_ms'])} ms** / {fmt(100 * (solo.get('miss_frac') or 0), '%.1f')}% / "
            f"{fmt(solo.get('mean_w'), '%.0f')} W [{fmt(solo.get('vdd_gpu_mean_w'), '%.1f')}] / "
            f"{fmt(solo.get('j_per_frame'), '%.3f')} J ({fmt(solo.get('j_per_frame_marginal'), '%.3f')}) / "
            f"{fmt(solo.get('clock_mhz_min'), '%.0f')} MHz / {events}")
    if ratios.get('p99_ratio'):
        text += f" · x{fmt(ratios.get('p99_ratio'))} p99, x{fmt(ratios.get('j_per_frame_ratio'))} J"
    return text


def paced_solo_table(verdict, point_names):
    keys = sorted({key for name in point_names for key in verdict['points'][name]['paced_solo']})
    if not keys:
        return []
    lines = ['', '## Paced solo per point: p99 / miss % / mean W [GPU rail W] / J per frame (marginal) / '
                 'clock min / events', ''] + table_header('row @ Hz', point_names)
    for key in keys:
        cells = [paced_solo_cell(verdict['points'][name], key) for name in point_names]
        lines.append(f"| {key} | " + ' | '.join(cells) + ' |')
    return lines


def colocation_cell(point, key):
    cell = point['cells'].get(key)
    if not cell:
        return '-'
    if not cell.get('valid'):
        return f"INVALID ({'; '.join(cell.get('invalid_reasons') or [])})"
    rows = ', '.join(f"{name} {fmt(row.get('p99_ms'), '%.1f')}"
                     for name, row in (cell.get('rows') or {}).items())
    events = cell.get('oc_events')
    events = fmt(events, '%.0f') if events is not None else '-'
    ratios = (point.get('vs_baseline') or {}).get('cells', {}).get(key, {})
    text = (f"**{fmt(cell.get('makespan_p99_ms'), '%.1f')} ms** / {'fits' if cell.get('fits') else 'MISS'} / "
            f"N {fmt(cell.get('N_measured'))} / {rows} / "
            f"{fmt(cell.get('mean_w'), '%.0f')} W [{fmt(cell.get('vdd_gpu_mean_w'), '%.1f')}] / "
            f"{fmt(cell.get('j_per_period'), '%.2f')} J / {events}")
    if ratios.get('makespan_p99_ratio'):
        text += f" · x{fmt(ratios.get('makespan_p99_ratio'))} makespan, x{fmt(ratios.get('N_ratio'))} N"
    return text


def colocation_table(verdict, point_names):
    keys = sorted({key for name in point_names for key in verdict['points'][name]['cells']})
    if not keys:
        return []
    lines = ['', '## Co-location cells per point: makespan p99 / fits / N_measured / per-row p99 / '
                 'mean W [GPU rail W] / J per period / events', ''] + table_header('mix / arm', point_names)
    for key in keys:
        cells = [colocation_cell(verdict['points'][name], key) for name in point_names]
        lines.append(f"| {key} | " + ' | '.join(cells) + ' |')
    return lines


def per_watt_cell(point, precision):
    summary = ((point.get('per_watt') or {}).get('precisions') or {}).get(precision)
    if not summary:
        return '-'
    fit = summary.get('fit') or {}
    unit = summary.get('unit') or 'TOPS'
    at50 = summary.get('at50') or {}
    at100 = summary.get('at100') or {}
    text = (f"W = {fmt(fit.get('intercept'), '%.0f')} + {fmt(fit.get('slope'))} x {unit} "
            f"(R2 {fmt(fit.get('r2'), '%.3f')}); "
            f"50 %: {fmt(at50.get('delivered'), '%.0f')} {unit} @ {fmt(at50.get('w'), '%.0f')} W = "
            f"{fmt(at50.get('per_w'))}/W; "
            f"100 %: {fmt(at100.get('delivered'), '%.0f')} {unit} @ {fmt(at100.get('w'), '%.0f')} W = "
            f"{fmt(at100.get('per_w'))}/W, {fmt(at100.get('gpu_mhz'), '%.0f')} MHz, "
            f"events {fmt(at100.get('oc'), '%.0f')}")
    knee = summary.get('knee')
    if knee:
        text += (f"; cap knee at {fmt(knee.get('busy_pct'), '%.0f')} % busy "
                 f"({fmt(knee.get('module_w'), '%.0f')} W)")
    ratios = (point.get('vs_baseline') or {}).get('per_watt', {}).get(precision, {})
    if ratios.get('per_w_at100_ratio'):
        text += (f" · x{fmt(ratios['per_w_at100_ratio'])} per W, "
                 f"x{fmt(ratios['delivered_at100_ratio'])} delivered")
    return text


def per_watt_table(verdict, point_names):
    precisions = []
    for name in point_names:
        for precision in ((verdict['points'][name].get('per_watt') or {}).get('precisions') or {}):
            if precision not in precisions:
                precisions.append(precision)
    if not precisions:
        return []
    lines = ['', '## Per-watt sweeps per point: fit over the >= 45 % busy points, the 50 % and 100 % points',
             ''] + table_header('precision', point_names)
    for precision in precisions:
        cells = [per_watt_cell(verdict['points'][name], precision) for name in point_names]
        lines.append(f"| {precision} | " + ' | '.join(cells) + ' |')
    return lines


READING_NOTES = [
    '', '## Reading the table', '',
    "- Every number is measured at its own point under a clock lock verified against that point's caps "
    "(the derived config); nothing is scaled from the baseline.",
    '- A mode or cap that lowers the clock ceiling shows up as a p99 ratio > 1 with a mean-W ratio < 1; '
    'J per frame decides whether the excursion pays: a lower J at a p99 that still fits the deadline is a '
    'real saving, a lower W at a higher J is not.',
    '- The per-watt slope is the marginal cost of throughput (W per TOPS or per GB/s); the intercept is the '
    'standing cost. Compare slopes across points: a point that changes only the intercept moves idle, '
    'not efficiency.',
    '- Cells INVALID at the baseline stay INVALID at every point for the same reason (e.g. a runtime that '
    'cannot start under an MPS thread cap); compare cells only where both points are valid.', '']


def report(verdict):
    completed = [name for name in verdict['points_order']
                 if verdict['points'].get(name, {}).get('status') == 'OK']
    lines = [f"# Stage 6 - power: {verdict.get('device_tag')} ({verdict.get('platform')}), "
             f"baseline {verdict['baseline']}", '',
             f"Run: `{verdict['run']}`  ·  verdict **{verdict['verdict']}**  ·  {verdict['date']}", '']
    lines += points_table(verdict)
    lines += checks_section(verdict['checks'])
    lines += paced_solo_table(verdict, completed)
    lines += colocation_table(verdict, completed)
    lines += per_watt_table(verdict, completed)
    lines += READING_NOTES
    return '\n'.join(lines)


if __name__ == '__main__':
    main()
