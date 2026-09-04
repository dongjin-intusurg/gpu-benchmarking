#!/usr/bin/env python3
"""Stage 7.2: write report.md from data.json.

    report.py data.json --out report.md

One section per stage present in data.json; an absent stage leaves a note
naming the command that produces it. Every table is read from the data -
nothing here names a row, a mix, an arm or a power point - and the only
dates in the output are the ones under 'Sources', so a regeneration from the
same runs is byte-identical.
"""
import argparse
import json
import sys

from tables import flag, fmt, num, pct, table

PRECISION_ORDER = ('fp16', 'int8', 'fp8', 'fp32')


def h(level, text):
    return ['', '#' * level + ' ' + text, '']


def value(x):
    return x is not None and isinstance(x, (int, float)) and x == x


def ordered(keys, first=PRECISION_ORDER):
    """Known precisions in their usual order, then anything else alphabetically."""
    keys = list(keys)
    return [k for k in first if k in keys] + sorted(k for k in keys if k not in first)


# ----------------------------------------------------------------------------- header
def header(d):
    dev = d['device']
    lines = [f"# Measurement report: {dev.get('name') or dev.get('tag')} (`{dev.get('tag')}`)", '',
             f"Stage 7 output, regenerated from the runs below (`data.json` schema `{d.get('schema')}`). "
             'Every number is measured or derived from a measurement; the datasheet values appear only as '
             'the denominators of attainment ratios.', '']
    lines += h(2, 'Sources')
    rows = []
    for stage in ('ceilings', 'solo', 'coloc', 'power'):
        s = (d.get('sources') or {}).get(stage)
        if s:
            rows.append([stage, f"`{s.get('run')}`", s.get('date') or '-', s.get('kind') or '-', f"`{s.get('path')}`"])
        else:
            rows.append([stage, '-', '-', 'absent', '-'])
    lines += table(['stage', 'run', 'date', 'kind', 'path'], rows, align=['l'] * 5)
    if d.get('checks'):
        lines += h(3, 'Provenance checks')
        lines += [f'- {c}' for c in d['checks']]
    if d.get('missing'):
        lines += h(3, 'Missing stages')
        lines += [f'- {m}' for m in d['missing']]
    lines += h(2, 'Device')
    ds = dev.get('datasheet') or {}
    rows = [['config', f"`{dev.get('config')}`"], ['module', dev.get('module') or '-'],
            ['platform', dev.get('platform') or '-'],
            ['datasheet dense peaks', f"fp16 {fmt(ds.get('fp16'), '%g')} TFLOPS, int8 {fmt(ds.get('int8'), '%g')} TOPS, "
                                      f"fp8 {fmt(ds.get('fp8'), '%g')} TFLOPS, fp32 {fmt(ds.get('fp32'), '%g')} TFLOPS, "
                                      f"DRAM {fmt(ds.get('bw_gbps'), '%g')} GB/s"],
            ['datasheet source', f"{ds.get('source_url') or '-'} (retrieved {ds.get('retrieved') or '-'})"],
            ['power envelope', f"{fmt(dev.get('power_envelope_w'), '%g')} W"],
            ['lock targets', ', '.join(f'{k} {v} MHz' for k, v in sorted((dev.get('lock_targets_mhz') or {}).items())) or '-'],
            ['VRAM budget cap', f"{fmt(dev.get('vram_budget_cap_mb'), '%g')} MB" if value(dev.get('vram_budget_cap_mb')) else 'none (unified memory)'],
            ['headroom line', f"{d.get('headroom'):.0%} of every budget (C at headroom = {1 / d['headroom']:.3g} / U_max)"]]
    lines += table(['', ''], rows, align=['l', 'l'])
    return lines


# ----------------------------------------------------------------------------- ceilings
def ceilings_section(d):
    c = d.get('ceilings')
    lines = h(2, 'Device ceilings (stage 3)')
    if not c:
        return lines + ['Not available: run the ceilings stage first.']
    lines += ['Figures: `fig_ceilings_attain.png`, `fig_bandwidth.png`, `fig_sustained.png`.', '',
              'The tensor ceilings of record are the kernel-isolated GEMM peaks (best n over the sweep); the torch '
              'burst values are what a framework GEMM reaches at the same n, and the sustained median is the '
              'planning number for a thermally settled device.', '']
    rows = []
    for p in ordered((c.get('precisions') or {}).keys()):
        v = c['precisions'][p]
        rows.append([p, fmt(v.get('datasheet'), '%g'),
                     f"{fmt(v.get('trt_best'), '%.1f')} @ n={fmt(v.get('trt_at_n'), '%d')}", pct(v.get('attain_trt')),
                     fmt(v.get('torch_burst_same_n'), '%.1f'), pct(v.get('attain_burst')),
                     fmt(v.get('torch_sustained_median'), '%.1f'), pct(v.get('attain_sustained')), v.get('unit') or '-'])
    lines += table(['precision', 'datasheet', 'kernel-isolated GEMM', '% datasheet', 'torch burst (same n)', '% datasheet',
                    'torch sustained (median)', '% datasheet', 'unit'], rows)
    shaped = c.get('shaped_gemm_tflops') or {}
    if shaped:
        lines += ['', 'Shaped GEMMs (fp16 TFLOPS, torch burst):', '']
        lines += table(['M x N x K', 'TFLOPS', '% fp16 datasheet'],
                       [[k, fmt(v, '%.1f'), pct(v / c['precisions']['fp16']['datasheet']) if value(v) and c['precisions'].get('fp16', {}).get('datasheet') else '-']
                        for k, v in sorted(shaped.items())])
    lines += h(3, 'DRAM bandwidth')
    rows = [[k, fmt(v.get('best_gbps'), '%.1f'), v.get('at_buffer') or '-', pct(v.get('frac_datasheet'))]
            for k, v in sorted((c.get('bandwidth') or {}).items())]
    lines += table(['kernel', 'best GB/s', 'at buffer', '% datasheet'], rows)
    lines += ['', f"`bw_eff` = {fmt(c.get('bw_eff_gbps'), '%.1f')} GB/s ({c.get('bw_eff_source')}) is the denominator of every "
              'bandwidth budget in the stages below.']
    haircut = c.get('haircut_gbps_by_workers') or {}
    if haircut:
        lines += ['', 'CPU-load haircut (informational; budgets use the idle figure):', '']
        lines += table(['CPU workers', 'copy GB/s'], [[k, fmt(v, '%.1f')] for k, v in sorted(haircut.items(), key=lambda kv: int(kv[0]))])
    s = c.get('sustained') or {}
    if s:
        lines += h(3, 'Sustained throughput')
        lines += [f"{s.get('precision')} GEMM for {fmt(s.get('duration_s'), '%g')} s: median {fmt(s.get('median'), '%.1f')} "
                  f"{s.get('unit')} (min {fmt(s.get('min'), '%.1f')}, max {fmt(s.get('max'), '%.1f')}, {s.get('n_samples')} samples) "
                  f"against a burst of {fmt(s.get('burst_same_n'), '%.1f')} at the same n: sustained / burst = "
                  f"{fmt(s.get('sustained_over_burst'), '%.2f')}. GPU clock never below {s.get('gpu_mhz_min')} MHz, "
                  f"temperature peaked at {fmt(s.get('temp_c_max'), '%.0f')} degC."]
    r = c.get('regime') or {}
    if r:
        lines += ['', f"Regime: preflight {r.get('preflight')}, lock {r.get('lock')}, drift {r.get('drift')}"
                  + (' - SMOKE ONLY, not certified' if r.get('smoke_only') else '') + '.']
    return lines


# ----------------------------------------------------------------------------- solo
def solo_section(d):
    s = d.get('solo')
    lines = h(2, 'Solo rows (stage 4)')
    if not s:
        return lines + ['Not available: run the model bench stage first.']
    rows = s.get('rows') or []
    lines += ['Figures: `fig_solo_latency.png`, `fig_solo_N.png`, `fig_solo_budgets.png`, `fig_bound_mix.png`, '
              '`fig_roofline.png`, `fig_floors.png`, `fig_rate_sweep.png`.', '',
              f"Budgets divide by `bw_eff` {fmt(s.get('bw_eff_gbps'), '%.1f')} GB/s ({s.get('bw_ceiling_source')}) and a VRAM "
              f"capacity of {fmt(s.get('vram_capacity_mb'), '%.0f')} MB ({s.get('vram_capacity_source')}). "
              f"U_max is the fullest of the three gauges, C = 1 / U_max, L = deadline / p99, N = min(L, C).", '',
              f"Run valid: {flag(s.get('run_valid'))} (lock {s.get('lock_verify')}, worst clock verdict "
              f"{s.get('worst_model_verdict')}" + (f", excluded: {', '.join(s['excluded_models'])}" if s.get('excluded_models') else '') + ').', '']
    body = []
    for r in rows:
        b = r.get('budgets') or {}
        deadline = f"{fmt(r.get('deadline_ms'), '%.4g')}" + ('' if r.get('deadline_kind') == 'spec' else f" ({r.get('deadline_kind')})")
        body.append([r['name'], r.get('kind') or '-', r.get('precision') or '-', fmt(r.get('hz'), '%.4g'), deadline,
                     num(r.get('p99_ms')), num(r.get('mean_ms')), num(r.get('bytes_MB')), num(r.get('vram_mb')), pct(b.get('time')), pct(b.get('bw')), pct(b.get('vram')),
                     fmt(r.get('C'), '%.3g'), fmt(r.get('L'), '%.3g'), fmt(r.get('N'), '%.3g'),
                     (r.get('cause') or '-').replace('-limited', ''),
                     'modal' if r.get('modal') else flag(r.get('meets_deadline'))])
    lines += table(['row', 'kind', 'precision', 'Hz', 'deadline ms', 'p99 ms', 'mean ms', 'bytes/frame MB', 'VRAM MB',
                    'U_time', 'U_bw', 'U_vram', 'C', 'L', 'N', 'limited by', 'meets'], body)
    incomplete = [r['name'] for r in rows if r.get('budgets_not_measured')]
    if incomplete:
        lines += ['', 'Rows with an unmeasured budget (U_max is a lower bound there): ' + ', '.join(incomplete) + '.']
    placeholder = [r['name'] for r in rows if r.get('deadline_kind') == 'placeholder']
    if placeholder:
        lines += ['', 'Placeholder deadlines (no product rate yet; their L and N are provisional): ' + ', '.join(placeholder) + '.']
    errors = [(r['name'], r['error']) for r in rows if r.get('error')]
    if errors:
        lines += ['', 'Unscored rows:'] + [f'- {n}: {e}' for n, e in errors]
    ru = s.get('rollup') or {}
    if ru:
        lines += h(3, 'Engine-row rollup')
        lines += [f"Scope: {ru.get('device_rollup_scope')}. Summed budgets: time {pct((ru.get('budgets') or {}).get('time_occupancy'))}, "
                  f"bandwidth {pct((ru.get('budgets') or {}).get('dram_bandwidth'))}, VRAM {pct((ru.get('budgets') or {}).get('vram_footprint'))} "
                  f"-> U_max {fmt(ru.get('U_max'), '%.3g')} ({ru.get('binding_budget')}), C {fmt(ru.get('C'), '%.3g')}, "
                  f"L {fmt(ru.get('L'), '%.3g')}, N {fmt(ru.get('N'), '%.3g')}"
                  + (f" ({ru.get('shortfall_cause_if_N_lt_1')})" if value(ru.get('N')) and ru['N'] < 1 else '')
                  + f"; workload constant {fmt(ru.get('workload_constant_gflops_per_s'), '%.4g')} GFLOP/s, "
                  f"score at the deadline {fmt(ru.get('score_tflops_at_deadline'), '%.3g')} TFLOP/s."]
    lines += floors_table(rows, d.get('headroom'))
    lines += sweep_table(rows)
    lines += roofline_table(rows)
    return lines


def floors_table(rows, headroom):
    lines = h(3, 'Latency floors and N ceilings')
    lines += ['t_floor = max(bytes / bw_eff, arch GFLOPs / compute ceiling); N_ceiling is N with p99 replaced by the floor '
              '- a roofline bound, never a promise. The compute ceiling is the sustained tensor median of the row precision '
              'where measured, the fp32 CUDA-core ceiling otherwise.', '']
    body = []
    for r in rows:
        f = r.get('floor')
        if not f:
            continue
        body.append([r['name'], num(r.get('p99_ms')), num(f.get('t_mem_ms')), num(f.get('t_comp_ms')), num(f.get('t_floor_ms')), f.get('bound_by') or '-',
                     num(r['p99_ms'] / f['t_floor_ms']) + 'x' if value(f.get('t_floor_ms')) and f['t_floor_ms'] > 0 and value(r.get('p99_ms')) else '-',
                     fmt(r.get('N'), '%.3g'), fmt(f.get('N_ceiling'), '%.3g'),
                     f"{f.get('comp_ceiling_key') or '-'} = {fmt(f.get('comp_ceiling'), '%.4g')}" if value(f.get('comp_ceiling')) else 'memory only'])
    if not body:
        return lines + ['No row carries a floor (no bytes/frame or architectural FLOPs recorded).']
    lines += table(['row', 'p99 ms', 't_mem ms', 't_comp ms', 't_floor ms', 'bound by', 'p99 / floor', 'N', 'N_ceiling', 'ceiling used'], body)
    return lines


def sweep_table(rows):
    body = [[r['name'], fmt(r.get('hz'), '%.4g'), fmt(r.get('max_hz_at_N1'), '%.4g'), r.get('max_hz_bound_by') or '-',
             ', '.join(f"{fmt(hz, '%g')} Hz: {fmt(n, '%.2f')}" for hz, n in (r.get('N_vs_hz') or []))]
            for r in rows if r.get('N_vs_hz') or value(r.get('max_hz_at_N1'))]
    if not body:
        return []
    lines = h(3, 'Rate sweep')
    lines += ['N at each candidate rate with the period as the deadline; `max Hz at N = 1` is the highest rate one device '
              'sustains solo.', '']
    lines += table(['row', 'declared Hz', 'max Hz at N = 1', 'bound by', 'N vs Hz'], body, align=['l', 'r', 'r', 'l', 'l'])
    return lines


def roofline_table(rows):
    body = []
    for r in rows:
        rf = r.get('roofline')
        if not rf:
            continue
        bound = rf.get('time_pct_by_bound') or {}
        pipe = rf.get('time_pct_by_pipe') or {}
        top = (rf.get('top_kernels') or [None])[0] or {}
        body.append([r['name'], fmt(bound.get('memory'), '%.0f%%'), fmt(bound.get('compute'), '%.0f%%'),
                     fmt(bound.get('latency'), '%.0f%%'), fmt(bound.get('unclassified'), '%.0f%%'),
                     ', '.join(f"{k} {fmt(v, '%.0f')}%" for k, v in sorted(pipe.items(), key=lambda kv: -kv[1])),
                     rf.get('byte_source') or '-',
                     f"{top.get('label')} ({fmt(top.get('time_pct'), '%.1f')}% of time)" if top else '-'])
    if not body:
        return []
    lines = h(3, 'Kernel roofline classification')
    lines += ['Share of engine time by bound class (memory: near the DRAM roof; compute: near a pipe ceiling; latency: '
              'far from both) and by issuing pipe. See `fig_bound_mix.png` and `fig_roofline.png`.', '']
    lines += table(['row', 'memory', 'compute', 'latency', 'unclassified', 'time by pipe', 'bytes from', 'largest kernel'],
                   body, align=['l', 'r', 'r', 'r', 'r', 'l', 'l', 'l'])
    return lines


# ----------------------------------------------------------------------------- co-location
def coloc_section(d):
    c = d.get('coloc')
    lines = h(2, 'Co-location (stage 5)')
    if not c:
        return lines + ['Not available: run the co-location stage first.']
    mixes, arms, cells = c.get('mixes') or [], c.get('arms') or [], c.get('cells') or []
    rules = c.get('rules') or {}
    by = {(x['mix'], x['arm']): x for x in cells}
    lines += ['Figures: `fig_coloc_matrix.png`, `fig_coloc_contention.png`, `fig_coloc_bounds.png`.', '',
              f"Source: {c.get('kind')} run, {fmt(c.get('run_seconds'), '%g')} s per cell, lock {c.get('lock_verified')}. "
              f"A cell fits when every frame row misses its deadline in at most {pct(rules.get('miss_frac_max'))} of frames "
              f"and at least {pct(rules.get('done_before_next_trigger_min'))} of frames finish before the next trigger. "
              'N_measured = min(L contended, C composed); an unfit cell keeps its N for comparison but is not a passing configuration.', '']
    lines += h(3, 'N_measured per mix and arm')
    body = []
    for m in mixes:
        row = [m]
        for a in arms:
            x = by.get((m, a))
            if not x:
                row.append('-')
            elif not x.get('valid'):
                row.append(x.get('status') or 'invalid')
            else:
                row.append(fmt(x.get('N_measured'), '%.2f') + ('' if x.get('fits') else ' (unfit)'))
        body.append(row)
    lines += table(['mix'] + arms, body)
    lines += h(3, 'Composed budgets per mix')
    lines += ['The frame rows\' solo budgets summed at the mix rates: the composed capacity the contended measurement is '
              'checked against.', '']
    body = []
    for m in mixes:
        cp = (c.get('composed') or {}).get(m)
        if not cp:
            continue
        body.append([m, fmt(cp.get('period_ms'), '%.4g'), pct(cp.get('U_time')), pct(cp.get('U_bw')), pct(cp.get('U_vram')),
                     cp.get('binding_budget') or '-', fmt(cp.get('C'), '%.3g'), fmt(cp.get('L'), '%.3g'), fmt(cp.get('N'), '%.3g'),
                     fmt(cp.get('units_at_0.65'), '%d'), ', '.join(f"{r.get('row')} ({r.get('role')})" for r in cp.get('rows') or [])])
    if body:
        lines += table(['mix', 'period ms', 'U_time', 'U_bw', 'U_vram', 'binding', 'C', 'L (predicted)', 'N (predicted)',
                        'units at headroom', 'rows'], body, align=['l'] + ['r'] * 9 + ['l'])
    lines += h(3, 'Contention per cell')
    body = []
    for x in cells:
        if not x.get('valid'):
            continue
        worst = []
        for rn, r in sorted((x.get('rows') or {}).items()):
            worst.append(f"{rn} {fmt(r.get('paced_solo_p99_ms'), '%.3g')} -> {fmt(r.get('p99_ms'), '%.3g')} ms"
                         + (f" (misses {pct(r.get('miss_frac'), 1)})" if value(r.get('miss_frac')) and r['miss_frac'] > 0 else ''))
        body.append([f"{x['mix']} / {x['arm']}", fmt(x.get('L_contended'), '%.3g'), fmt(x.get('C_composed'), '%.3g'),
                     fmt(x.get('N_measured'), '%.3g'), fmt(x.get('N_predicted'), '%.3g'),
                     fmt(x.get('makespan_p99_over_period'), '%.2f'), flag(x.get('fits')), x.get('worst_row') or '-',
                     '; '.join(worst)])
    if body:
        lines += table(['cell', 'L contended', 'C composed', 'N measured', 'N predicted', 'makespan p99 / period', 'fits',
                        'worst row', 'paced solo -> contended p99'], body, align=['l'] + ['r'] * 6 + ['l', 'l'])
    invalid = [x for x in cells if not x.get('valid')]
    if invalid:
        lines += ['', 'Invalid cells:'] + [f"- {x['mix']} / {x['arm']}: {'; '.join(x.get('invalid_reasons') or [x.get('status') or 'invalid'])}"
                                            for x in invalid]
    side = [(x, n, sr) for x in cells if x.get('valid') for n, sr in sorted((x.get('side_rows') or {}).items())]
    if side:
        lines += h(3, 'Side rows under contention')
        lines += ['Generative and streaming rows that share the device with the frame rows; their L is the composed L '
                  'scaled by the measured contended-over-solo ratio.', '']
        body = [[f"{x['mix']} / {x['arm']}", n, sr.get('runtime') or '-', fmt(sr.get('L_contended'), '%.3g'), flag(sr.get('fits')),
                 ', '.join(f"{k} x{fmt(v.get('ratio'), '%.2f')}" for k, v in sorted((sr.get('vs_solo') or {}).items()))]
                for x, n, sr in side]
        lines += table(['cell', 'side row', 'runtime', 'L contended', 'fits', 'contended / solo'], body,
                       align=['l', 'l', 'l', 'r', 'r', 'l'])
    return lines


# ----------------------------------------------------------------------------- power
def power_section(d):
    p = d.get('power')
    lines = h(2, 'Power operating points (stage 6)')
    if not p:
        return lines + ['Not available: run the power stage first.']
    points = p.get('points_order') or []
    P = p.get('points') or {}
    lines += ['Figures: `fig_power_solo.png`, `fig_power_cells.png`, `fig_power_per_watt.png`.', '',
              f"Baseline point `{p.get('baseline')}`; {fmt(p.get('run_seconds'), '%g')} s per paced solo row and per cell; "
              f"verdict {p.get('verdict')}.", '']
    body = [[pt, P[pt].get('applied_mode') or '-', P[pt].get('status') or '-', P[pt].get('preflight_verdict') or '-',
             P[pt].get('lock_verdict') or '-',
             ', '.join(f'{k} {v}' for k, v in sorted((P[pt].get('lock_targets_mhz') or {}).items())) or '-',
             fmt(P[pt].get('idle_w'), '%.1f'), P[pt].get('why') or '']
            for pt in points]
    lines += table(['point', 'applied mode', 'status', 'preflight', 'lock', 'lock targets MHz', 'idle W', 'note'], body,
                   align=['l'] * 8)
    keys = sorted({k for pt in points for k in (P[pt].get('paced_solo') or {})})
    if keys:
        lines += h(3, 'Paced solo rows per point')
        lines += ['p99 / module W / energy per frame above idle, per point (`row@Hz`).', '']
        body = []
        for k in keys:
            row = [k]
            for pt in points:
                v = (P[pt].get('paced_solo') or {}).get(k)
                row.append('-' if not v else f"{fmt(v.get('p99_ms'), '%.3g')} ms / {fmt(v.get('mean_w'), '%.1f')} W / "
                                             f"{fmt(v.get('j_per_frame_marginal'), '%.3g')} J"
                                             + (f" (misses {pct(v.get('miss_frac'), 1)})" if value(v.get('miss_frac')) and v['miss_frac'] > 0 else ''))
            body.append(row)
        lines += table(['row'] + points, body, align=['l'] * (len(points) + 1))
    keys = sorted({k for pt in points for k, v in (P[pt].get('cells') or {}).items() if v.get('valid')})
    if keys:
        lines += h(3, 'Co-location cells per point')
        lines += ['N_measured / mean module W / energy per period, per point; `unfit` where a row misses its rule.', '']
        body = []
        for k in keys:
            row = [k]
            for pt in points:
                v = (P[pt].get('cells') or {}).get(k)
                row.append('-' if not (v and v.get('valid')) else
                           f"N {fmt(v.get('N_measured'), '%.2f')} / {fmt(v.get('mean_w'), '%.1f')} W / {fmt(v.get('j_per_period'), '%.3g')} J"
                           + ('' if v.get('fits') else ' (unfit)'))
            body.append(row)
        lines += table(['cell'] + points, body, align=['l'] * (len(points) + 1))
    lines += h(3, 'Throughput per watt')
    lines += ['Linear fit of module W against delivered throughput over the duty sweep; `at 100%` is the saturated point.', '']
    body = []
    for pt in points:
        pw = (P[pt].get('per_watt') or {}).get('precisions') or {}
        for prec in ordered(pw.keys(), PRECISION_ORDER + ('copy',)):
            v = pw[prec]
            fit, a100, knee = v.get('fit') or {}, v.get('at100') or {}, v.get('knee') or {}
            body.append([pt, prec, v.get('unit') or '-', fmt(a100.get('delivered'), '%.1f'), fmt(a100.get('w'), '%.1f'),
                         fmt(a100.get('per_w'), '%.3g'), fmt(a100.get('gpu_mhz'), '%.0f'),
                         f"{fmt(fit.get('slope'), '%.3g')} W per unit + {fmt(fit.get('intercept'), '%.1f')} W (R2 {fmt(fit.get('r2'), '%.3f')}, n={fmt(fit.get('n'), '%d')})",
                         fmt(knee.get('busy_pct'), '%.0f%%')])
    if body:
        lines += table(['point', 'precision', 'unit', 'delivered at 100%', 'module W', 'per W', 'GPU MHz', 'fit', 'knee busy'],
                       body, align=['l', 'l', 'l', 'r', 'r', 'r', 'r', 'l', 'r'])
    vs = [(pt, P[pt].get('vs_baseline')) for pt in points if P[pt].get('vs_baseline')]
    if vs:
        lines += h(3, 'Against the baseline point')
        lines += ['Ratios point / baseline for the same row, cell or precision.', '']
        for pt, v in vs:
            sol = v.get('paced_solo') or {}
            cel = v.get('cells') or {}
            pwr = v.get('per_watt') or {}
            lines += [f'- `{pt}`: paced solo p99 x' + '..'.join(fmt(x, '%.2f') for x in span(r.get('p99_ratio') for r in sol.values()))
                      + ', module W x' + '..'.join(fmt(x, '%.2f') for x in span(r.get('mean_w_ratio') for r in sol.values()))
                      + '; cell N x' + '..'.join(fmt(x, '%.2f') for x in span(r.get('N_ratio') for r in cel.values()))
                      + ('; fit changes: ' + ', '.join(k for k, r in sorted(cel.items()) if r.get('fits_changed')) if any(r.get('fits_changed') for r in cel.values()) else '')
                      + '; per W at 100%: ' + ', '.join(f"{k} x{fmt(r.get('per_w_at100_ratio'), '%.2f')}" for k, r in sorted(pwr.items())) + '.']
    return lines


def span(values):
    vals = [v for v in values if value(v)]
    return (min(vals), max(vals)) if vals else (None, None)


# ----------------------------------------------------------------------------- notes
def reading_notes(d):
    lines = h(2, 'Reading notes')
    lines += ['- N is the number of devices one copy of the workload needs; N >= 1 means one device carries it. '
              'C is what the summed budgets allow, L what the deadline allows; the smaller one is the verdict and its '
              'name (throughput- or latency-limited) says whether more capacity or faster silicon would move it.',
              f"- The {d.get('headroom'):.0%} headroom line is a planning margin on C, not a measurement: a row or mix whose "
              'fullest gauge exceeds it fits on one device only without margin.',
              '- Kernel-isolated GEMM ceilings are the physics of the tensor pipes; the sustained median is what a '
              'thermally settled device delivers and is the compute ceiling the floors use.',
              '- A latency floor is a roofline bound. N_ceiling says how far optimization could take a row on this '
              'device; it never says that it will.',
              '- Co-location N is measured under contention; the composed prediction is the solo budgets summed. '
              'Where the two disagree the contended measurement wins, and the gap is the interference the isolation '
              'arms are meant to reduce.',
              '- Power points change clocks, not models: compare the same row or cell across points, never rows '
              'across points.']
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    d = json.load(open(args.data))
    lines = header(d) + ceilings_section(d) + solo_section(d) + coloc_section(d) + power_section(d) + reading_notes(d)
    text = '\n'.join(lines).rstrip() + '\n'
    with open(args.out, 'w') as f:
        f.write(text)
    print(f'  wrote   {args.out} ({len(lines)} lines)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
