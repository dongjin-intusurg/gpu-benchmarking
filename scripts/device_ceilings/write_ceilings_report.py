#!/usr/bin/env python3
"""Render ceilings_report.txt from a thorough-ceilings results.json.

Usage:
  write_ceilings_report.py <ceilings_results.json> --device <cfg.json>
      --provenance <dir> --out <ceilings_report.txt> [--drift <drift.json>]

Seven fixed sections: [1] device constants, [2] per-precision compute ceilings,
[3] bandwidth suite + bw_eff, [4] shaped GEMMs, [5] sustained vs burst,
[6] clock drift verdict, [7] sanity verdicts. A missing suite is printed as
'SKIP: not measured' — never fabricated, never a crash.

Pre-declared policies (declared here, not chosen after seeing the data):
- Attainable ceiling per precision = the TRT probe's opt-level-5 peak
  (measure_compute_trt.py, passed via --trt) when present — the run's
  definition of attainable and the same instrument the model half uses; the
  torch/cuBLAS sweep is then printed as a labeled cross-check only, and the
  sanity band (section [7]) is judged against the TRT peak. Without --trt the
  torch sweep is the ceiling (legacy behavior, unchanged).
- Burst ceiling per precision (torch cross-check) = best over the GEMM size
  sweep; the winning N is reported because the peak size is itself a device
  characteristic.
- Sustained ceiling = median of the 3-minute sustained[] samples. Samples are
  attributed to fp16 (the suite-6 default precision) unless meta.sustain_prec
  says otherwise; every other precision shows '-' in the sustained column.
- Sustained/burst is reported BOTH ways: vs the same-size burst (n=4096,
  fp16 fp16-acc — the sustained kernel's own configuration) and vs the global
  burst (best over the whole sweep). The two ratios answer different
  questions (kernel thermal fade vs planning margin) and quoting only one
  has caused ambiguity before.
- Sustained clamp incidence = samples below 0.9x the best sustained sample —
  the same 10% clean-rep window the measurement suite uses for rep spread.
- bw_eff = best idle copy_RW over the buffer-size sweep. This is the run
  budget basis (solo-exclusive regime measures with an otherwise-idle system);
  the CPU-load haircut curve is printed as informational context only.
- Sanity bands come from the device config sanity_bands block; defaults if
  absent: GEMM burst 45-95% of the dense datasheet peak, idle copy 75-95% of
  the datasheet bandwidth (the calibrated bands from the ceiling guide).
  Out-of-band low = FAIL (probe likely off the intended unit/precision path);
  out-of-band high = FAIL (suspect sparsity/precision mixup in the datasheet
  comparison). In-band = PASS.
- headless_verified is taken from an explicit headless field in preflight.json
  when present; otherwise inferred true from a passing preflight verdict
  (preflight refuses to run under a desktop), and forced false if smoke_only.
- Old-schema tolerance: meta.gpu_freq_hz (Hz string) is normalized to MHz;
  sweep points without spread_clean_pct/clamped_reps print '-'; fp8 may be a
  scalar, an 'unavailable: ...' string, or absent; haircut may be a string
  (e.g. a missing-tool note) or contain string entries.

stdlib only.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

GEMM_BAND_DEFAULT = (0.45, 0.95)   # fraction of dense datasheet peak
BW_BAND_DEFAULT = (0.75, 0.95)     # idle copy fraction of datasheet bandwidth
CLEAN_FACTOR = 0.9                 # sample is clean iff >= 0.9x best (suite policy)
RULE = '=' * 78


def load_json(path):
    if not path:
        return None
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def fmt(v, nd=1, dash='-'):
    n = num(v)
    return f'{n:.{nd}f}' if n is not None else dash


def pct(a, b, nd=1, dash='-'):
    a, b = num(a), num(b)
    if a is None or b is None or b == 0:
        return dash
    return f'{100.0 * a / b:.{nd}f}%'


def parse_env_txt(path):
    """environment.txt is strict 'key: value' lines; first value per key wins."""
    d = {}
    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            for line in f:
                if ':' in line:
                    k, v = line.split(':', 1)
                    d.setdefault(k.strip(), v.strip())
    except OSError:
        pass
    return d


def sweep_points(gemm, metric):
    """(n, point-dict) pairs for one precision metric; sizes only, sorted."""
    pts = []
    for k, row in gemm.items():
        try:
            n = int(k)
        except (TypeError, ValueError):
            continue
        if not isinstance(row, dict):
            continue
        m = row.get(metric)
        if isinstance(m, dict) and num(m.get('best')) is not None:
            pts.append((n, m))
    return sorted(pts)


def find_headless(obj):
    """Depth-first search for any key containing 'headless' (schema-tolerant)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if 'headless' in str(k).lower() and isinstance(v, bool):
                return v
            r = find_headless(v)
            if r is not None:
                return r
    return None


def kv(lines, key, value):
    lines.append(f'{key}: {value}')


# ---------------------------------------------------------------- section 1
def section_device_constants(L, cfg, env, meta, preflight, lockv):
    L.append('[1] DEVICE CONSTANTS')
    kv(L, 'device_id', cfg.get('device_id', '-'))
    kv(L, 'device_tag', env.get('device_tag', cfg.get('device_tag', '-')))
    kv(L, 'platform', env.get('platform', cfg.get('platform', '-')))
    kv(L, 'arch', cfg.get('arch', '-'))
    kv(L, 'module', cfg.get('module', '-'))
    kv(L, 'gpu_name', env.get('gpu_name', cfg.get('device_name_match', '-')))
    kv(L, 'driver', env.get('driver', '-'))
    kv(L, 'vram_total', env.get('vram_total', '-'))
    vp = cfg.get('vram_physical_mb')
    kv(L, 'vram_physical_mb', vp if vp is not None else 'unified (MemTotal at run time)')
    vc = cfg.get('vram_budget_cap_mb')
    kv(L, 'vram_budget_cap_mb', vc if vc is not None else 'auto (no cap configured)')
    kv(L, 'host', env.get('host', '-'))
    kv(L, 'kernel', env.get('kernel', '-'))
    if env.get('board'):
        kv(L, 'board', env['board'])
    if env.get('l4t'):
        kv(L, 'l4t', env['l4t'])
    kv(L, 'run_date', env.get('run_date', '-'))
    kv(L, 'trtexec', env.get('trtexec', '-'))
    kv(L, 'tensorrt_py', env.get('tensorrt_py', '-'))
    kv(L, 'nvcc', env.get('nvcc', '-'))
    kv(L, 'torch', env.get('torch', meta.get('torch', '-')))
    kv(L, 'lock_method', cfg.get('lock_method', '-'))
    lt = cfg.get('lock_targets_mhz')
    if isinstance(lt, dict):
        kv(L, 'lock_targets_mhz', ' '.join(f'{k}={v}' for k, v in lt.items()))
    else:
        kv(L, 'lock_targets_mhz', lt if lt is not None else '-')
    ref = (lockv or {}).get('reference_clock_mhz')
    if isinstance(ref, dict):
        kv(L, 'realized_reference_clock_mhz', ' '.join(f'{k}={v}' for k, v in ref.items()))
    if lockv is not None and 'verdict' in lockv:
        kv(L, 'lock_verify_verdict', lockv.get('verdict'))
    # gpu clock at measurement start: new schema MHz, old schema Hz string
    mhz = num(meta.get('gpu_freq_mhz_start'))
    if mhz is None:
        try:
            mhz = int(float(meta.get('gpu_freq_hz'))) // 1_000_000
        except (TypeError, ValueError):
            mhz = None
    kv(L, 'gpu_clock_at_measure_start_mhz', mhz if mhz is not None else '-')
    headless = find_headless(preflight) if preflight else None
    smoke = bool((preflight or {}).get('smoke_only', False))
    if headless is None and preflight is not None:
        verdict = str((preflight or {}).get('verdict', '')).upper()
        headless = verdict in ('PASS', 'OK', 'WARN') and not smoke
    kv(L, 'headless_verified', str(headless).lower() if headless is not None else 'unknown')
    if preflight is not None:
        kv(L, 'preflight_verdict', preflight.get('verdict', '-'))
        kv(L, 'smoke_only', str(smoke).lower())
    kv(L, 'measure_start', meta.get('start', '-'))
    kv(L, 'measure_end', meta.get('end', '-'))
    if meta.get('incomplete'):
        L.append(f"WARN: results flagged incomplete — {meta['incomplete']}")
    ds = cfg.get('datasheet') or {}
    for k in sorted(ds):
        kv(L, f'datasheet_{k}', ds[k])
    kv(L, 'power_envelope_w', cfg.get('power_envelope_w', '-'))


# ---------------------------------------------------------------- section 2
ROW = '  {:<22}{:>9}{:>8}{:>10}{:>10}{:>8}{:>8}{:>9}{:>8}{:>9}'


def trt_ok(trt):
    """True iff the TRT compute JSON has at least one measured precision peak."""
    if not isinstance(trt, dict):
        return False
    precs = trt.get('precisions')
    return isinstance(precs, dict) and any(
        isinstance(v, dict) and num(v.get('best')) is not None for v in precs.values())


# TRT precision name -> (label unit, datasheet key, the bests-dict key section
# [7] sanity already looks for). fp16 maps to the fp16-acc slot so the existing
# max()-over-fp16-keys sanity logic picks it up unchanged.
TRT_ROWS = [
    ('fp16', 'TFLOPS', 'peak_fp16_dense_tflops', 'fp16_fp16acc_tflops'),
    ('int8', 'TOPS', 'peak_int8_dense_tops', 'int8_tops'),
    ('fp8', 'TFLOPS', 'peak_fp8_dense_tflops', 'fp8'),
    # fp32 rides the same instrument (--noTF32 pins the CUDA cores); its bests
    # key is display-only — section [7] sanity bands stay tensor-only.
    ('fp32', 'TFLOPS', 'peak_fp32_cuda_tflops', 'fp32_trt'),
]
TRT_ROW = '  {:<20}{:>10}{:>8}{:>10}{:>9}   {}'


def section_compute_trt(L, trt, cfg):
    """Authoritative per-precision compute ceiling from the TRT probe. Returns
    the bests dict section [7] consumes."""
    ds = cfg.get('datasheet') or {}
    meta = trt.get('meta') or {}
    precs = trt.get('precisions') or {}
    L.append('[2] ATTAINABLE COMPUTE CEILINGS  (TensorRT, --builderOptimizationLevel=5; '
             'best over size sweep)')
    L.append(f'  instrument: trtexec {meta.get("trt_version", "?")}  '
             f'opt_level={meta.get("opt_level", "?")}  sizes={meta.get("sizes", "?")}  '
             f'(attainable = 2*n^3 / GEMM-kernel time; conversion kernels excluded by design)')
    L.append(TRT_ROW.format('precision', 'attain', '@N', 'dsheet', 'a/ds', 'flags'))
    bests = {}
    for name, unit, ds_key, best_key in TRT_ROWS:
        p = precs.get(name)
        peak = ds.get(ds_key)
        if isinstance(p, dict) and num(p.get('best')) is not None:
            bests[best_key] = (p.get('at_n', '-'), p['best'])
            L.append(TRT_ROW.format(f'{name} {unit}', fmt(p['best']), p.get('at_n', '-'),
                                    fmt(peak, 0), pct(p['best'], peak, 0), p.get('flags', '-')))
        else:
            why = (p.get('error') if isinstance(p, dict) else p) or 'not measured'
            L.append(f'  {name + " " + unit:<20}SKIP: {why}')
    return bests


def section_compute(L, R, cfg, sust_prec, sust_median, trt):
    """Emit section [2]. When a TRT compute probe is present it is the
    authoritative ceiling (and drives sanity); the torch sweep is printed below
    as a cross-check. Without it, the torch sweep is the ceiling (legacy
    behavior). Returns (bests_for_sanity, torch_bests)."""
    if trt_ok(trt):
        trt_bests = section_compute_trt(L, trt, cfg)
        L.append('')
        torch_bests = section_compute_torch(
            L, R, cfg, sust_prec, sust_median,
            '[2b] COMPUTE CROSS-CHECK  (torch/cuBLAS default dispatch — NOT the attainable '
            'ceiling; typed int8 underperforms on some arches)')
        # A precision the TRT probe failed on still gets a sanity verdict from
        # the torch sweep rather than silently vanishing from section [7] —
        # but ONLY for precisions with no TRT result at all: a present-but-poor
        # TRT value must FAIL loudly, never be masked by a torch sibling key
        # (measured trap: TRT fp16 at 12% of datasheet hid behind the torch
        # fp32-acc key and sanity read 52% PASS).
        groups = {'fp16': ('fp16_fp32acc_tflops', 'fp16_fp16acc_tflops'),
                  'int8': ('int8_tops',), 'fp8': ('fp8',)}
        for keys in groups.values():
            if any(k in trt_bests for k in keys):
                continue
            for k in keys:
                if k in torch_bests:
                    trt_bests[k] = torch_bests[k]
        return trt_bests, torch_bests
    torch_bests = section_compute_torch(
        L, R, cfg, sust_prec, sust_median,
        '[2] COMPUTE CEILINGS  (burst = best over size sweep; plan against sustained)')
    return torch_bests, torch_bests


def section_compute_torch(L, R, cfg, sust_prec, sust_median, header):
    L.append(header)
    gemm = R.get('gemm_sweep')
    ds = cfg.get('datasheet') or {}
    if not isinstance(gemm, dict) or not gemm:
        L.append('  SKIP: not measured')
        return {}
    L.append(ROW.format('precision', 'burst', '@N', 'sustain', 'dsheet',
                        'b/ds', 's/ds', 'spread%', 'clean%', 'clamped'))
    bests = {}
    rows = [
        ('fp16 fp32-acc TFLOPS', 'fp16_fp32acc_tflops', ds.get('peak_fp16_dense_tflops')),
        ('fp16 fp16-acc TFLOPS', 'fp16_fp16acc_tflops', ds.get('peak_fp16_dense_tflops')),
        ('int8 TOPS', 'int8_tops', ds.get('peak_int8_dense_tops')),
    ]
    sust_row_metric = 'int8_tops' if str(sust_prec).startswith('int8') else 'fp16_fp16acc_tflops'
    for label, metric, peak in rows:
        pts = sweep_points(gemm, metric)
        if not pts:
            L.append(f'  {label:<22}SKIP: not measured')
            continue
        n_best, p = max(pts, key=lambda x: x[1]['best'])
        bests[metric] = (n_best, p['best'])
        clamped = p.get('clamped_reps')
        sust = sust_median if metric == sust_row_metric else None
        L.append(ROW.format(
            label, fmt(p['best']), n_best, fmt(sust),
            fmt(peak, 0), pct(p['best'], peak, 0), pct(sust, peak, 0),
            fmt(p.get('spread_pct')), fmt(p.get('spread_clean_pct')),
            clamped if num(clamped) is not None else '-'))
    fp8 = gemm.get('fp8_n4096_tflops')
    if num(fp8) is not None:
        peak = ds.get('peak_fp8_dense_tflops')
        bests['fp8'] = (4096, fp8)
        L.append(ROW.format('fp8 TFLOPS', fmt(fp8), 4096, '-',
                            fmt(peak, 0), pct(fp8, peak, 0), '-', '-', '-', '-'))
    elif isinstance(fp8, str):
        L.append(f'  {"fp8 TFLOPS":<22}SKIP: {fp8}')
    else:
        L.append(f'  {"fp8 TFLOPS":<22}SKIP: not measured')
    cuda = num(R.get('cuda_fp32'))
    if cuda is not None:
        peak = ds.get('peak_fp32_cuda_tflops')
        bests['cuda_fp32'] = (8192, cuda)
        L.append(ROW.format('cuda fp32 TFLOPS', fmt(cuda, 2), 8192, '-',
                            fmt(peak, 1), pct(cuda, peak, 0), '-', '-', '-', '-'))
    else:
        L.append(f'  {"cuda fp32 TFLOPS":<22}SKIP: not measured')
    # clamp incidence across the whole sweep: power-envelope data, not error
    total_clamped = 0
    counted = 0
    for _, metric, _ in rows:
        for _, p in sweep_points(gemm, metric):
            c = num(p.get('clamped_reps'))
            if c is not None:
                total_clamped += int(c)
                counted += 1
    if counted:
        L.append(f'  clamped reps across sweep: {total_clamped} '
                 f'(over {counted} points x 5 reps; overcurrent clamps are data, not lock failure)')
    else:
        L.append('  clamped reps across sweep: - (per-rep clamp accounting absent in this schema)')
    return bests


# ---------------------------------------------------------------- section 3
def section_bandwidth(L, R, cfg):
    L.append('[3] BANDWIDTH  (GB/s; best per kernel over buffer sizes)')
    bw = R.get('bandwidth')
    ds_bw = (cfg.get('datasheet') or {}).get('peak_bw_gbps')
    bw_eff = None
    if not isinstance(bw, dict) or not bw:
        L.append('  SKIP: not measured')
    else:
        L.append('  {:<14}{:>9}{:>11}{:>9}'.format('kernel', 'best', '@buffer', 'frac-ds'))
        for kernel in ('copy_RW', 'read_only', 'write_only', 'triad_2R1W'):
            best_v, best_sz = None, None
            for sz, row in bw.items():
                v = num(row.get(kernel)) if isinstance(row, dict) else None
                if v is not None and (best_v is None or v > best_v):
                    best_v, best_sz = v, sz
            if best_v is None:
                L.append(f'  {kernel:<14}SKIP: not measured')
                continue
            L.append('  {:<14}{:>9}{:>11}{:>9}'.format(
                kernel, fmt(best_v), str(best_sz), pct(best_v, ds_bw, 0)))
            if kernel == 'copy_RW':
                bw_eff = best_v
        if bw_eff is not None:
            L.append(f'  bw_eff (budget basis): {fmt(bw_eff)} GB/s = best idle copy_RW — '
                     'the solo-exclusive regime budgets against idle bandwidth')
        L.append(f'  datasheet bandwidth: {fmt(ds_bw, 0)} GB/s')
    hc = R.get('haircut')
    if isinstance(hc, dict) and hc:
        def worker_key(k):
            try:
                return int(str(k).rstrip('w'))
            except ValueError:
                return 1 << 30
        parts = []
        for k in sorted(hc, key=worker_key):
            v = hc[k]
            parts.append(f'{k}: {fmt(v) if num(v) is not None else v}')
        L.append('  CPU-load haircut (informational — shared-memory contention curve, '
                 'NOT the budget basis):')
        L.append('    ' + '   '.join(parts))
    elif isinstance(hc, str):
        L.append(f'  CPU-load haircut: SKIP: {hc}')
    else:
        L.append('  CPU-load haircut: SKIP: not measured')
    return bw_eff


# ---------------------------------------------------------------- section 4
def section_shaped(L, R, bests):
    L.append('[4] SHAPED GEMMs  (fp16; non-square transformer-like shapes)')
    shaped = R.get('shaped')
    if not isinstance(shaped, dict) or not shaped:
        L.append('  SKIP: not measured')
        return
    fp16_peaks = [bests[m][1] for m in ('fp16_fp32acc_tflops', 'fp16_fp16acc_tflops') if m in bests]
    square = max(fp16_peaks) if fp16_peaks else None
    ref = f'{fmt(square)} TFLOPS' if square is not None else 'unavailable'
    L.append(f'  square fp16 burst peak (reference): {ref}')
    for shape in shaped:
        v = num(shaped[shape])
        if v is None:
            L.append(f'  {shape:<18}SKIP: {shaped[shape]}')
        else:
            L.append(f'  {shape:<18}{fmt(v):>8} TFLOPS   ({pct(v, square, 0)} of square peak)')


# ---------------------------------------------------------------- section 5
def section_sustained(L, R, bests, sust_prec, sust_median):
    L.append(f'[5] SUSTAINED vs BURST  (3-min continuous {sust_prec} GEMM at n=4096)')
    samples = R.get('sustained')
    if not isinstance(samples, list) or not samples:
        L.append('  SKIP: not measured')
        return
    tf = [num(s.get('tflops')) for s in samples if isinstance(s, dict)]
    tf = [v for v in tf if v is not None]
    if not tf:
        L.append('  SKIP: samples carry no throughput values')
        return
    best_s = max(tf)
    clamped = sum(1 for v in tf if v < CLEAN_FACTOR * best_s)
    L.append(f'  samples: {len(tf)}   median: {fmt(sust_median)}   '
             f'min: {fmt(min(tf))}   max: {fmt(best_s)}')
    same_metric = 'int8_tops' if str(sust_prec).startswith('int8') else 'fp16_fp16acc_tflops'
    gemm = R.get('gemm_sweep') or {}
    same = None
    if isinstance(gemm, dict):
        p4096 = gemm.get('4096') or gemm.get(4096)
        if isinstance(p4096, dict):
            m = p4096.get(same_metric)
            if isinstance(m, dict):
                same = num(m.get('best'))
    if same is not None:
        L.append(f'  vs same-size burst (n=4096 {same_metric} best {fmt(same)}): '
                 f'{fmt(sust_median / same, 2) if num(sust_median) else "-"}')
    else:
        L.append('  vs same-size burst: - (no n=4096 sweep point for the sustained precision)')
    if same_metric in bests:
        n_g, glob = bests[same_metric]
        L.append(f'  vs global burst    (sweep best {fmt(glob)} @ n={n_g}): '
                 f'{fmt(sust_median / glob, 2) if num(sust_median) else "-"}')
    else:
        L.append('  vs global burst: - (no sweep points for the sustained precision)')
    temps = [num(s.get('max_temp_c')) for s in samples if isinstance(s, dict)]
    temps = [t for t in temps if t is not None and t >= 0]
    clocks = [num(s.get('gpu_mhz')) for s in samples if isinstance(s, dict)]
    clocks = [c for c in clocks if c is not None and c >= 0]
    L.append(f'  max temp: {fmt(max(temps)) if temps else "-"} C   '
             f'gpu clock min/max: '
             f'{fmt(min(clocks), 0) if clocks else "-"}/{fmt(max(clocks), 0) if clocks else "-"} MHz')
    L.append(f'  clamp incidence: {clamped}/{len(tf)} samples below '
             f'{CLEAN_FACTOR:.1f}x the best sustained sample')


# ---------------------------------------------------------------- section 6
def section_drift(L, drift):
    L.append('[6] CLOCK DRIFT VERDICT  (under-load samples during the ceiling suites)')
    if drift is None:
        L.append('  not sampled')
        return
    L.append(f'  verdict: {drift.get("verdict", "unknown")}')
    for k in sorted(drift):
        if k == 'verdict':
            continue
        v = drift[k]
        if isinstance(v, (int, float, str, bool)) or v is None:
            L.append(f'  {k}: {v}')
        elif isinstance(v, dict):
            flat = ' '.join(f'{sk}={sv}' for sk, sv in v.items()
                            if isinstance(sv, (int, float, str, bool)))
            if flat:
                L.append(f'  {k}: {flat}')
        elif isinstance(v, list) and len(v) <= 8:
            L.append(f'  {k}: {v}')


# ---------------------------------------------------------------- section 7
def section_sanity(L, cfg, bests, bw_eff, compute_source='torch'):
    L.append('[7] SANITY VERDICTS  (bands from device config; out-of-band = FAIL)')
    src = 'TensorRT attainable (opt level 5)' if compute_source == 'TRT' else 'torch/cuBLAS sweep'
    L.append(f'  compute ceiling source: {src}')
    bands = cfg.get('sanity_bands') or {}
    gemm_band = bands.get('gemm_frac_of_datasheet', list(GEMM_BAND_DEFAULT))
    bw_band = bands.get('bw_idle_frac_of_datasheet', list(BW_BAND_DEFAULT))
    ds = cfg.get('datasheet') or {}
    verdicts = []

    def judge(label, measured, peak, band, low_meaning, high_meaning, in_meaning):
        measured, peak = num(measured), num(peak)
        if measured is None or peak is None or not peak:
            L.append(f'  {label:<26}SKIP: not measured (or no datasheet value)')
            return
        frac = measured / peak
        lo, hi = float(band[0]), float(band[1])
        if frac < lo:
            verdict, meaning = 'FAIL', low_meaning
        elif frac > hi:
            verdict, meaning = 'FAIL', high_meaning
        else:
            verdict, meaning = 'PASS', in_meaning
        verdicts.append((label, verdict))
        L.append(f'  {label:<26}{100 * frac:5.1f}%  band {100 * lo:.0f}-{100 * hi:.0f}%  '
                 f'{verdict} — {meaning}')

    fp16_peaks = [bests[m][1] for m in ('fp16_fp32acc_tflops', 'fp16_fp16acc_tflops') if m in bests]
    judge('fp16 burst / datasheet', max(fp16_peaks) if fp16_peaks else None,
          ds.get('peak_fp16_dense_tflops'), gemm_band,
          'below band — probe likely not on the tensor-core fp16 path, or device throttled hard',
          'above band — suspect a sparsity/precision mixup in the datasheet comparison',
          'measured fraction sits in the expected achievable window')
    judge('int8 burst / datasheet', bests.get('int8_tops', (None, None))[1],
          ds.get('peak_int8_dense_tops'), gemm_band,
          'below band — int8 tensor-core path not engaged (no QDQ/calibration, or a slow tactic)',
          'above band — suspect a sparsity/precision mixup in the datasheet comparison',
          'measured fraction sits in the expected achievable window')
    judge('fp8 burst / datasheet', bests.get('fp8', (None, None))[1],
          ds.get('peak_fp8_dense_tflops'), gemm_band,
          'below band — fp8 scaled-mm path probably not hitting the fp8 tensor pipe',
          'above band — suspect a sparsity/precision mixup in the datasheet comparison',
          'measured fraction sits in the expected achievable window')
    judge('idle copy / datasheet', bw_eff, ds.get('peak_bw_gbps'), bw_band,
          'below band — contention or wrong kernel mix; idle copy should approach datasheet',
          'above band — check the byte accounting (copy counts read+write bytes)',
          'streaming probe is DRAM-limited as intended')
    return verdicts


def main():
    ap = argparse.ArgumentParser(description='Render the 7-section ceilings TXT report.')
    ap.add_argument('results', help='ceilings results.json from measure_ceilings_thorough.py')
    ap.add_argument('--device', required=True, help='device config json')
    ap.add_argument('--provenance', required=True, help='provenance directory')
    ap.add_argument('--out', required=True, help='output ceilings_report.txt path')
    ap.add_argument('--drift', default=None, help='clock drift json (driftmon/v1)')
    ap.add_argument('--trt', default=None,
                    help='trt_compute.json from measure_compute_trt.py — when present it is '
                         'the authoritative section [2] compute ceiling and drives sanity')
    args = ap.parse_args()

    R = load_json(args.results)
    if R is None:
        print(f'FATAL: cannot read results json: {args.results}', file=sys.stderr)
        sys.exit(1)
    cfg = load_json(args.device)
    if cfg is None:
        print(f'FATAL: cannot read device config: {args.device}', file=sys.stderr)
        sys.exit(1)
    prov = Path(args.provenance)
    env = parse_env_txt(prov / 'environment.txt')
    if not env:
        print(f'WARN: no readable environment.txt under {prov} — identity lines fall back to config',
              file=sys.stderr)
    preflight = load_json(prov / 'preflight.json')
    lockv = load_json(prov / 'lock_verified.json')
    drift = load_json(args.drift)
    if args.drift and drift is None:
        print(f'WARN: drift json unreadable: {args.drift} — section [6] says not sampled',
              file=sys.stderr)
    trt = load_json(args.trt)
    if args.trt and trt is None:
        print(f'WARN: trt json unreadable: {args.trt} — section [2] falls back to torch cross-check',
              file=sys.stderr)
    meta = R.get('meta') or {}

    sust_prec = str(meta.get('sustain_prec', 'fp16'))
    tf = [num(s.get('tflops')) for s in (R.get('sustained') or []) if isinstance(s, dict)]
    tf = [v for v in tf if v is not None]
    sust_median = round(statistics.median(tf), 1) if tf else None

    L = []
    L.append(RULE)
    L.append(f'MEASURED DEVICE CEILINGS — {cfg.get("device_id", "unknown device")}')
    L.append(RULE)
    section_device_constants(L, cfg, env, meta, preflight, lockv)
    L.append('')
    # sanity_bests: the attainable (TRT) peaks when present, else torch.
    # torch_bests: always the torch sweep — the self-consistent reference for
    # the torch-measured shaped and sustained comparisons below.
    sanity_bests, torch_bests = section_compute(L, R, cfg, sust_prec, sust_median, trt)
    L.append('')
    bw_eff = section_bandwidth(L, R, cfg)
    L.append('')
    section_shaped(L, R, torch_bests)
    L.append('')
    section_sustained(L, R, torch_bests, sust_prec, sust_median)
    L.append('')
    section_drift(L, drift)
    L.append('')
    verdicts = section_sanity(L, cfg, sanity_bests, bw_eff, compute_source='TRT' if trt_ok(trt) else 'torch') or []
    L.append(RULE)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(L) + '\n', encoding='utf-8')
    summary = '  '.join(f'{lbl.split(" /")[0]}={v}' for lbl, v in verdicts) or 'no sanity verdicts'
    print(f'ceilings report -> {out}')
    print(f'sanity: {summary}')


if __name__ == '__main__':
    main()
