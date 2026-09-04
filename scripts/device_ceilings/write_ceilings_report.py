#!/usr/bin/env python3
"""Render ceilings_report.txt from a thorough-ceilings results.json.

Seven fixed sections: [1] device constants, [2] per-precision compute ceilings
(the TRT probe when --trt is given, else the torch sweep), [3] bandwidth + bw_eff,
[4] shaped GEMMs, [5] sustained vs burst, [6] clock drift verdict, [7] sanity
verdicts. A missing suite prints 'SKIP: ...', never a crash. Prints
'ceilings report -> <path>' and 'sanity: <label>=<PASS|FAIL> ...'. stdlib only.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

# Sanity bands used when the device config has no sanity_bands block: the
# calibrated achievable windows from the ceiling guide. Below = probe off the
# intended unit/precision path; above = sparsity/precision mixup vs datasheet.
GEMM_BAND_DEFAULT = (0.45, 0.95)
BW_BAND_DEFAULT = (0.75, 0.95)
# A sustained sample is clean iff >= 0.9x the best one — the same 10% window the
# measurement suite uses for rep spread, so clamp incidence is comparable.
CLEAN_FACTOR = 0.9
RULE = '=' * 78

TORCH_ROW = '  {:<22}{:>9}{:>8}{:>10}{:>10}{:>8}{:>8}{:>9}{:>8}{:>9}'
TORCH_ROWS = [
    ('fp16 fp32-acc TFLOPS', 'fp16_fp32acc_tflops', 'peak_fp16_dense_tflops'),
    ('fp16 fp16-acc TFLOPS', 'fp16_fp16acc_tflops', 'peak_fp16_dense_tflops'),
    ('int8 TOPS', 'int8_tops', 'peak_int8_dense_tops'),
]
FP16_METRICS = ('fp16_fp32acc_tflops', 'fp16_fp16acc_tflops')

# TRT precision -> (unit, datasheet key, bests key). fp16 lands on the fp16-acc
# slot so section [7]'s max-over-fp16-keys logic picks it up; fp32 rides the same
# instrument but its key is display-only — sanity bands stay tensor-only.
TRT_ROW = '  {:<20}{:>10}{:>8}{:>10}{:>9}   {}'
TRT_ROWS = [
    ('fp16', 'TFLOPS', 'peak_fp16_dense_tflops', 'fp16_fp16acc_tflops'),
    ('int8', 'TOPS', 'peak_int8_dense_tops', 'int8_tops'),
    ('fp8', 'TFLOPS', 'peak_fp8_dense_tflops', 'fp8'),
    ('fp32', 'TFLOPS', 'peak_fp32_cuda_tflops', 'fp32_trt'),
]
# Precisions the TRT probe can fail on wholesale; only then does the torch
# sweep supply the sanity value. A present-but-poor TRT value must FAIL loudly,
# never be masked by a torch sibling key.
SANITY_GROUPS = {'fp16': FP16_METRICS, 'int8': ('int8_tops',), 'fp8': ('fp8',)}

BANDWIDTH_ROW = '  {:<14}{:>9}{:>11}{:>9}'
BANDWIDTH_KERNELS = ('copy_RW', 'read_only', 'write_only', 'triad_2R1W')


def load_json(path):
    if not path:
        return None
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def num(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def fmt(value, decimals=1, dash='-'):
    n = num(value)
    return f'{n:.{decimals}f}' if n is not None else dash


def pct(numerator, denominator, decimals=1, dash='-'):
    numerator, denominator = num(numerator), num(denominator)
    if numerator is None or denominator is None or denominator == 0:
        return dash
    return f'{100.0 * numerator / denominator:.{decimals}f}%'


def parse_env_txt(path):
    """environment.txt is strict 'key: value' lines; first value per key wins."""
    env = {}
    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            for line in f:
                if ':' in line:
                    key, value = line.split(':', 1)
                    env.setdefault(key.strip(), value.strip())
    except OSError:
        pass
    return env


def sweep_points(gemm_sweep, metric):
    """Sorted (n, point) pairs for one precision metric; non-size keys are skipped."""
    points = []
    for key, row in gemm_sweep.items():
        try:
            n = int(key)
        except (TypeError, ValueError):
            continue
        if not isinstance(row, dict):
            continue
        point = row.get(metric)
        if isinstance(point, dict) and num(point.get('best')) is not None:
            points.append((n, point))
    return sorted(points)


def find_headless(obj):
    """Depth-first search for any boolean key containing 'headless' (schema-tolerant)."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if 'headless' in str(key).lower() and isinstance(value, bool):
                return value
            found = find_headless(value)
            if found is not None:
                return found
    return None


def sustained_throughputs(results):
    samples = results.get('sustained') or []
    throughputs = [num(sample.get('tflops')) for sample in samples if isinstance(sample, dict)]
    return [value for value in throughputs if value is not None]


def sustained_metric(sustain_prec):
    return 'int8_tops' if str(sustain_prec).startswith('int8') else 'fp16_fp16acc_tflops'


def kv(lines, key, value):
    lines.append(f'{key}: {value}')


def flat_dict(value):
    return ' '.join(f'{key}={item}' for key, item in value.items())


def start_clock_mhz(meta):
    """Clock at measurement start: new schema is MHz, old schema an Hz string."""
    mhz = num(meta.get('gpu_freq_mhz_start'))
    if mhz is not None:
        return mhz
    try:
        return int(float(meta.get('gpu_freq_hz'))) // 1_000_000
    except (TypeError, ValueError):
        return None


def headless_verdict(preflight):
    """Explicit headless field when present; otherwise inferred from a passing
    preflight (it refuses to run under a desktop), forced false if smoke_only."""
    headless = find_headless(preflight) if preflight else None
    smoke = bool((preflight or {}).get('smoke_only', False))
    if headless is None and preflight is not None:
        verdict = str((preflight or {}).get('verdict', '')).upper()
        headless = verdict in ('PASS', 'OK', 'WARN') and not smoke
    return headless, smoke


def section_device_constants(lines, device_config, env, meta, preflight, lock_verified):
    lines.append('[1] DEVICE CONSTANTS')
    kv(lines, 'device_id', device_config.get('device_id', '-'))
    kv(lines, 'device_tag', env.get('device_tag', device_config.get('device_tag', '-')))
    kv(lines, 'platform', env.get('platform', device_config.get('platform', '-')))
    kv(lines, 'arch', device_config.get('arch', '-'))
    kv(lines, 'module', device_config.get('module', '-'))
    kv(lines, 'gpu_name', env.get('gpu_name', device_config.get('device_name_match', '-')))
    kv(lines, 'driver', env.get('driver', '-'))
    kv(lines, 'vram_total', env.get('vram_total', '-'))
    vram_physical = device_config.get('vram_physical_mb')
    kv(lines, 'vram_physical_mb',
       vram_physical if vram_physical is not None else 'unified (MemTotal at run time)')
    vram_cap = device_config.get('vram_budget_cap_mb')
    kv(lines, 'vram_budget_cap_mb', vram_cap if vram_cap is not None else 'auto (no cap configured)')
    kv(lines, 'host', env.get('host', '-'))
    kv(lines, 'kernel', env.get('kernel', '-'))
    if env.get('board'):
        kv(lines, 'board', env['board'])
    if env.get('l4t'):
        kv(lines, 'l4t', env['l4t'])
    kv(lines, 'run_date', env.get('run_date', '-'))
    kv(lines, 'trtexec', env.get('trtexec', '-'))
    kv(lines, 'tensorrt_py', env.get('tensorrt_py', '-'))
    kv(lines, 'nvcc', env.get('nvcc', '-'))
    kv(lines, 'torch', env.get('torch', meta.get('torch', '-')))
    kv(lines, 'lock_method', device_config.get('lock_method', '-'))
    lock_targets = device_config.get('lock_targets_mhz')
    if isinstance(lock_targets, dict):
        kv(lines, 'lock_targets_mhz', flat_dict(lock_targets))
    else:
        kv(lines, 'lock_targets_mhz', lock_targets if lock_targets is not None else '-')
    reference_clock = (lock_verified or {}).get('reference_clock_mhz')
    if isinstance(reference_clock, dict):
        kv(lines, 'realized_reference_clock_mhz', flat_dict(reference_clock))
    if lock_verified is not None and 'verdict' in lock_verified:
        kv(lines, 'lock_verify_verdict', lock_verified.get('verdict'))
    mhz = start_clock_mhz(meta)
    kv(lines, 'gpu_clock_at_measure_start_mhz', mhz if mhz is not None else '-')
    headless, smoke = headless_verdict(preflight)
    kv(lines, 'headless_verified', str(headless).lower() if headless is not None else 'unknown')
    if preflight is not None:
        kv(lines, 'preflight_verdict', preflight.get('verdict', '-'))
        kv(lines, 'smoke_only', str(smoke).lower())
    kv(lines, 'measure_start', meta.get('start', '-'))
    kv(lines, 'measure_end', meta.get('end', '-'))
    if meta.get('incomplete'):
        lines.append(f"WARN: results flagged incomplete — {meta['incomplete']}")
    datasheet = device_config.get('datasheet') or {}
    for key in sorted(datasheet):
        kv(lines, f'datasheet_{key}', datasheet[key])
    kv(lines, 'power_envelope_w', device_config.get('power_envelope_w', '-'))


def trt_ok(trt):
    """True iff the TRT compute JSON has at least one measured precision peak."""
    if not isinstance(trt, dict):
        return False
    precisions = trt.get('precisions')
    return isinstance(precisions, dict) and any(
        isinstance(entry, dict) and num(entry.get('best')) is not None
        for entry in precisions.values())


def section_compute_trt(lines, trt, device_config):
    """Authoritative per-precision ceiling from the TRT probe; returns the bests
    dict section [7] consumes: metric -> (n, value)."""
    datasheet = device_config.get('datasheet') or {}
    meta = trt.get('meta') or {}
    precisions = trt.get('precisions') or {}
    lines.append('[2] ATTAINABLE COMPUTE CEILINGS  (TensorRT, --builderOptimizationLevel=5; '
                 'best over size sweep)')
    lines.append(f'  instrument: trtexec {meta.get("trt_version", "?")}  '
                 f'opt_level={meta.get("opt_level", "?")}  sizes={meta.get("sizes", "?")}  '
                 f'(attainable = 2*n^3 / GEMM-kernel time; conversion kernels excluded by design)')
    lines.append(TRT_ROW.format('precision', 'attain', '@N', 'dsheet', 'a/ds', 'flags'))
    bests = {}
    for name, unit, datasheet_key, best_key in TRT_ROWS:
        entry = precisions.get(name)
        peak = datasheet.get(datasheet_key)
        if isinstance(entry, dict) and num(entry.get('best')) is not None:
            bests[best_key] = (entry.get('at_n', '-'), entry['best'])
            lines.append(TRT_ROW.format(f'{name} {unit}', fmt(entry['best']), entry.get('at_n', '-'),
                                        fmt(peak, 0), pct(entry['best'], peak, 0),
                                        entry.get('flags', '-')))
        else:
            why = (entry.get('error') if isinstance(entry, dict) else entry) or 'not measured'
            lines.append(f'  {name + " " + unit:<20}SKIP: {why}')
    return bests


def section_compute(lines, results, device_config, sust_prec, sust_median, trt):
    """Section [2]. With a TRT probe it is the ceiling and drives sanity, the torch
    sweep prints as a cross-check; without it the torch sweep is the ceiling.
    Returns (bests for sanity, torch bests)."""
    if not trt_ok(trt):
        torch_bests = section_compute_torch(
            lines, results, device_config, sust_prec, sust_median,
            '[2] COMPUTE CEILINGS  (burst = best over size sweep; plan against sustained)')
        return torch_bests, torch_bests
    sanity_bests = section_compute_trt(lines, trt, device_config)
    lines.append('')
    torch_bests = section_compute_torch(
        lines, results, device_config, sust_prec, sust_median,
        '[2b] COMPUTE CROSS-CHECK  (torch/cuBLAS default dispatch — NOT the attainable '
        'ceiling; typed int8 underperforms on some arches)')
    for keys in SANITY_GROUPS.values():
        if any(key in sanity_bests for key in keys):
            continue
        for key in keys:
            if key in torch_bests:
                sanity_bests[key] = torch_bests[key]
    return sanity_bests, torch_bests


def torch_sweep_rows(lines, gemm_sweep, datasheet, sust_prec, sust_median):
    bests = {}
    sust_row_metric = sustained_metric(sust_prec)
    for label, metric, datasheet_key in TORCH_ROWS:
        points = sweep_points(gemm_sweep, metric)
        if not points:
            lines.append(f'  {label:<22}SKIP: not measured')
            continue
        peak = datasheet.get(datasheet_key)
        n_best, point = max(points, key=lambda item: item[1]['best'])
        bests[metric] = (n_best, point['best'])
        clamped = point.get('clamped_reps')
        sustained = sust_median if metric == sust_row_metric else None
        lines.append(TORCH_ROW.format(
            label, fmt(point['best']), n_best, fmt(sustained),
            fmt(peak, 0), pct(point['best'], peak, 0), pct(sustained, peak, 0),
            fmt(point.get('spread_pct')), fmt(point.get('spread_clean_pct')),
            clamped if num(clamped) is not None else '-'))
    return bests


def torch_fp8_row(lines, gemm_sweep, datasheet, bests):
    fp8 = gemm_sweep.get('fp8_n4096_tflops')
    if num(fp8) is not None:
        peak = datasheet.get('peak_fp8_dense_tflops')
        bests['fp8'] = (4096, fp8)
        lines.append(TORCH_ROW.format('fp8 TFLOPS', fmt(fp8), 4096, '-',
                                      fmt(peak, 0), pct(fp8, peak, 0), '-', '-', '-', '-'))
    elif isinstance(fp8, str):
        lines.append(f'  {"fp8 TFLOPS":<22}SKIP: {fp8}')
    else:
        lines.append(f'  {"fp8 TFLOPS":<22}SKIP: not measured')


def torch_cuda_fp32_row(lines, results, datasheet, bests):
    cuda = num(results.get('cuda_fp32'))
    if cuda is None:
        lines.append(f'  {"cuda fp32 TFLOPS":<22}SKIP: not measured')
        return
    peak = datasheet.get('peak_fp32_cuda_tflops')
    bests['cuda_fp32'] = (8192, cuda)
    lines.append(TORCH_ROW.format('cuda fp32 TFLOPS', fmt(cuda, 2), 8192, '-',
                                  fmt(peak, 1), pct(cuda, peak, 0), '-', '-', '-', '-'))


def torch_clamp_summary(lines, gemm_sweep):
    total_clamped = 0
    counted = 0
    for _, metric, _ in TORCH_ROWS:
        for _, point in sweep_points(gemm_sweep, metric):
            clamped = num(point.get('clamped_reps'))
            if clamped is not None:
                total_clamped += int(clamped)
                counted += 1
    if counted:
        lines.append(f'  clamped reps across sweep: {total_clamped} '
                     f'(over {counted} points x 5 reps; overcurrent clamps are data, not lock failure)')
    else:
        lines.append('  clamped reps across sweep: - (per-rep clamp accounting absent in this schema)')


def section_compute_torch(lines, results, device_config, sust_prec, sust_median, header):
    lines.append(header)
    gemm_sweep = results.get('gemm_sweep')
    datasheet = device_config.get('datasheet') or {}
    if not isinstance(gemm_sweep, dict) or not gemm_sweep:
        lines.append('  SKIP: not measured')
        return {}
    lines.append(TORCH_ROW.format('precision', 'burst', '@N', 'sustain', 'dsheet',
                                  'b/ds', 's/ds', 'spread%', 'clean%', 'clamped'))
    bests = torch_sweep_rows(lines, gemm_sweep, datasheet, sust_prec, sust_median)
    torch_fp8_row(lines, gemm_sweep, datasheet, bests)
    torch_cuda_fp32_row(lines, results, datasheet, bests)
    torch_clamp_summary(lines, gemm_sweep)
    return bests


def best_over_buffers(bandwidth, kernel):
    best_value, best_size = None, None
    for size, row in bandwidth.items():
        value = num(row.get(kernel)) if isinstance(row, dict) else None
        if value is not None and (best_value is None or value > best_value):
            best_value, best_size = value, size
    return best_value, best_size


def haircut_worker_key(key):
    try:
        return int(str(key).rstrip('w'))
    except ValueError:
        return 1 << 30


def bandwidth_haircut(lines, results):
    haircut = results.get('haircut')
    if isinstance(haircut, dict) and haircut:
        parts = []
        for key in sorted(haircut, key=haircut_worker_key):
            value = haircut[key]
            parts.append(f'{key}: {fmt(value) if num(value) is not None else value}')
        lines.append('  CPU-load haircut (informational — shared-memory contention curve, '
                     'NOT the budget basis):')
        lines.append('    ' + '   '.join(parts))
    elif isinstance(haircut, str):
        lines.append(f'  CPU-load haircut: SKIP: {haircut}')
    else:
        lines.append('  CPU-load haircut: SKIP: not measured')


def section_bandwidth(lines, results, device_config):
    """Section [3]; returns bw_eff = best idle copy_RW, the budget basis of the
    solo-exclusive regime (the haircut curve is context only)."""
    lines.append('[3] BANDWIDTH  (GB/s; best per kernel over buffer sizes)')
    bandwidth = results.get('bandwidth')
    datasheet_bw = (device_config.get('datasheet') or {}).get('peak_bw_gbps')
    bw_eff = None
    if not isinstance(bandwidth, dict) or not bandwidth:
        lines.append('  SKIP: not measured')
    else:
        lines.append(BANDWIDTH_ROW.format('kernel', 'best', '@buffer', 'frac-ds'))
        for kernel in BANDWIDTH_KERNELS:
            best_value, best_size = best_over_buffers(bandwidth, kernel)
            if best_value is None:
                lines.append(f'  {kernel:<14}SKIP: not measured')
                continue
            lines.append(BANDWIDTH_ROW.format(
                kernel, fmt(best_value), str(best_size), pct(best_value, datasheet_bw, 0)))
            if kernel == 'copy_RW':
                bw_eff = best_value
        if bw_eff is not None:
            lines.append(f'  bw_eff (budget basis): {fmt(bw_eff)} GB/s = best idle copy_RW — '
                         'the solo-exclusive regime budgets against idle bandwidth')
        lines.append(f'  datasheet bandwidth: {fmt(datasheet_bw, 0)} GB/s')
    bandwidth_haircut(lines, results)
    return bw_eff


def fp16_square_peak(bests):
    fp16_peaks = [bests[metric][1] for metric in FP16_METRICS if metric in bests]
    return max(fp16_peaks) if fp16_peaks else None


def section_shaped(lines, results, bests):
    lines.append('[4] SHAPED GEMMs  (fp16; non-square transformer-like shapes)')
    shaped = results.get('shaped')
    if not isinstance(shaped, dict) or not shaped:
        lines.append('  SKIP: not measured')
        return
    square = fp16_square_peak(bests)
    reference = f'{fmt(square)} TFLOPS' if square is not None else 'unavailable'
    lines.append(f'  square fp16 burst peak (reference): {reference}')
    for shape in shaped:
        value = num(shaped[shape])
        if value is None:
            lines.append(f'  {shape:<18}SKIP: {shaped[shape]}')
        else:
            lines.append(f'  {shape:<18}{fmt(value):>8} TFLOPS   ({pct(value, square, 0)} of square peak)')


def same_size_burst(gemm_sweep, metric):
    if not isinstance(gemm_sweep, dict):
        return None
    point_4096 = gemm_sweep.get('4096') or gemm_sweep.get(4096)
    if not isinstance(point_4096, dict):
        return None
    entry = point_4096.get(metric)
    return num(entry.get('best')) if isinstance(entry, dict) else None


def sample_values(samples, key):
    values = [num(sample.get(key)) for sample in samples if isinstance(sample, dict)]
    return [value for value in values if value is not None and value >= 0]


def section_sustained(lines, results, bests, sust_prec, sust_median):
    """Section [5]. Sustained/burst is reported both vs the same-size burst (the
    sustained kernel's own configuration) and vs the global sweep best: kernel
    thermal fade and planning margin are different questions."""
    lines.append(f'[5] SUSTAINED vs BURST  (3-min continuous {sust_prec} GEMM at n=4096)')
    samples = results.get('sustained')
    if not isinstance(samples, list) or not samples:
        lines.append('  SKIP: not measured')
        return
    throughputs = sustained_throughputs(results)
    if not throughputs:
        lines.append('  SKIP: samples carry no throughput values')
        return
    best_sample = max(throughputs)
    clamped = sum(1 for value in throughputs if value < CLEAN_FACTOR * best_sample)
    lines.append(f'  samples: {len(throughputs)}   median: {fmt(sust_median)}   '
                 f'min: {fmt(min(throughputs))}   max: {fmt(best_sample)}')
    metric = sustained_metric(sust_prec)
    same = same_size_burst(results.get('gemm_sweep') or {}, metric)
    if same is not None:
        lines.append(f'  vs same-size burst (n=4096 {metric} best {fmt(same)}): '
                     f'{fmt(sust_median / same, 2) if num(sust_median) else "-"}')
    else:
        lines.append('  vs same-size burst: - (no n=4096 sweep point for the sustained precision)')
    if metric in bests:
        n_global, global_best = bests[metric]
        lines.append(f'  vs global burst    (sweep best {fmt(global_best)} @ n={n_global}): '
                     f'{fmt(sust_median / global_best, 2) if num(sust_median) else "-"}')
    else:
        lines.append('  vs global burst: - (no sweep points for the sustained precision)')
    temps = sample_values(samples, 'max_temp_c')
    clocks = sample_values(samples, 'gpu_mhz')
    lines.append(f'  max temp: {fmt(max(temps)) if temps else "-"} C   '
                 f'gpu clock min/max: '
                 f'{fmt(min(clocks), 0) if clocks else "-"}/{fmt(max(clocks), 0) if clocks else "-"} MHz')
    lines.append(f'  clamp incidence: {clamped}/{len(throughputs)} samples below '
                 f'{CLEAN_FACTOR:.1f}x the best sustained sample')


def section_drift(lines, drift):
    lines.append('[6] CLOCK DRIFT VERDICT  (under-load samples during the ceiling suites)')
    if drift is None:
        lines.append('  not sampled')
        return
    lines.append(f'  verdict: {drift.get("verdict", "unknown")}')
    for key in sorted(drift):
        if key == 'verdict':
            continue
        value = drift[key]
        if isinstance(value, (int, float, str, bool)) or value is None:
            lines.append(f'  {key}: {value}')
        elif isinstance(value, dict):
            flat = ' '.join(f'{sub_key}={sub_value}' for sub_key, sub_value in value.items()
                            if isinstance(sub_value, (int, float, str, bool)))
            if flat:
                lines.append(f'  {key}: {flat}')
        elif isinstance(value, list) and len(value) <= 8:
            lines.append(f'  {key}: {value}')


def judge_band(lines, verdicts, label, measured, peak, band, low_meaning, high_meaning, in_meaning):
    measured, peak = num(measured), num(peak)
    if measured is None or peak is None or not peak:
        lines.append(f'  {label:<26}SKIP: not measured (or no datasheet value)')
        return
    fraction = measured / peak
    low, high = float(band[0]), float(band[1])
    if fraction < low:
        verdict, meaning = 'FAIL', low_meaning
    elif fraction > high:
        verdict, meaning = 'FAIL', high_meaning
    else:
        verdict, meaning = 'PASS', in_meaning
    verdicts.append((label, verdict))
    lines.append(f'  {label:<26}{100 * fraction:5.1f}%  band {100 * low:.0f}-{100 * high:.0f}%  '
                 f'{verdict} — {meaning}')


def section_sanity(lines, device_config, bests, bw_eff, compute_source='torch'):
    lines.append('[7] SANITY VERDICTS  (bands from device config; out-of-band = FAIL)')
    source = 'TensorRT attainable (opt level 5)' if compute_source == 'TRT' else 'torch/cuBLAS sweep'
    lines.append(f'  compute ceiling source: {source}')
    bands = device_config.get('sanity_bands') or {}
    gemm_band = bands.get('gemm_frac_of_datasheet', list(GEMM_BAND_DEFAULT))
    bw_band = bands.get('bw_idle_frac_of_datasheet', list(BW_BAND_DEFAULT))
    datasheet = device_config.get('datasheet') or {}
    verdicts = []
    above_band = 'above band — suspect a sparsity/precision mixup in the datasheet comparison'
    in_band = 'measured fraction sits in the expected achievable window'
    judge_band(lines, verdicts, 'fp16 burst / datasheet', fp16_square_peak(bests),
               datasheet.get('peak_fp16_dense_tflops'), gemm_band,
               'below band — probe likely not on the tensor-core fp16 path, or device throttled hard',
               above_band, in_band)
    judge_band(lines, verdicts, 'int8 burst / datasheet', bests.get('int8_tops', (None, None))[1],
               datasheet.get('peak_int8_dense_tops'), gemm_band,
               'below band — int8 tensor-core path not engaged (no QDQ/calibration, or a slow tactic)',
               above_band, in_band)
    judge_band(lines, verdicts, 'fp8 burst / datasheet', bests.get('fp8', (None, None))[1],
               datasheet.get('peak_fp8_dense_tflops'), gemm_band,
               'below band — fp8 scaled-mm path probably not hitting the fp8 tensor pipe',
               above_band, in_band)
    judge_band(lines, verdicts, 'idle copy / datasheet', bw_eff, datasheet.get('peak_bw_gbps'), bw_band,
               'below band — contention or wrong kernel mix; idle copy should approach datasheet',
               'above band — check the byte accounting (copy counts read+write bytes)',
               'streaming probe is DRAM-limited as intended')
    return verdicts


def parse_args():
    parser = argparse.ArgumentParser(description='Render the 7-section ceilings TXT report.')
    parser.add_argument('results', help='ceilings results.json from measure_ceilings_thorough.py')
    parser.add_argument('--device', required=True, help='device config json')
    parser.add_argument('--provenance', required=True, help='provenance directory')
    parser.add_argument('--out', required=True, help='output ceilings_report.txt path')
    parser.add_argument('--drift', default=None, help='clock drift json (driftmon/v1)')
    parser.add_argument('--trt', default=None,
                        help='trt_compute.json from measure_compute_trt.py — when present it is '
                             'the authoritative section [2] compute ceiling and drives sanity')
    return parser.parse_args()


def load_inputs(args):
    results = load_json(args.results)
    if results is None:
        print(f'FATAL: cannot read results json: {args.results}', file=sys.stderr)
        sys.exit(1)
    device_config = load_json(args.device)
    if device_config is None:
        print(f'FATAL: cannot read device config: {args.device}', file=sys.stderr)
        sys.exit(1)
    provenance = Path(args.provenance)
    env = parse_env_txt(provenance / 'environment.txt')
    if not env:
        print(f'WARN: no readable environment.txt under {provenance} — identity lines fall back to config',
              file=sys.stderr)
    preflight = load_json(provenance / 'preflight.json')
    lock_verified = load_json(provenance / 'lock_verified.json')
    drift = load_json(args.drift)
    if args.drift and drift is None:
        print(f'WARN: drift json unreadable: {args.drift} — section [6] says not sampled',
              file=sys.stderr)
    trt = load_json(args.trt)
    if args.trt and trt is None:
        print(f'WARN: trt json unreadable: {args.trt} — section [2] falls back to torch cross-check',
              file=sys.stderr)
    return results, device_config, env, preflight, lock_verified, drift, trt


def main():
    args = parse_args()
    results, device_config, env, preflight, lock_verified, drift, trt = load_inputs(args)
    meta = results.get('meta') or {}
    sust_prec = str(meta.get('sustain_prec', 'fp16'))
    throughputs = sustained_throughputs(results)
    sust_median = round(statistics.median(throughputs), 1) if throughputs else None

    lines = [RULE, f'MEASURED DEVICE CEILINGS — {device_config.get("device_id", "unknown device")}', RULE]
    section_device_constants(lines, device_config, env, meta, preflight, lock_verified)
    lines.append('')
    # torch_bests stays the reference for the torch-measured shaped and sustained
    # comparisons even when the TRT probe supplies the sanity peaks.
    sanity_bests, torch_bests = section_compute(lines, results, device_config, sust_prec, sust_median, trt)
    lines.append('')
    bw_eff = section_bandwidth(lines, results, device_config)
    lines.append('')
    section_shaped(lines, results, torch_bests)
    lines.append('')
    section_sustained(lines, results, torch_bests, sust_prec, sust_median)
    lines.append('')
    section_drift(lines, drift)
    lines.append('')
    compute_source = 'TRT' if trt_ok(trt) else 'torch'
    verdicts = section_sanity(lines, device_config, sanity_bests, bw_eff, compute_source=compute_source) or []
    lines.append(RULE)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    summary = '  '.join(f'{label.split(" /")[0]}={verdict}' for label, verdict in verdicts)
    print(f'ceilings report -> {out}')
    print(f'sanity: {summary or "no sanity verdicts"}')


if __name__ == '__main__':
    main()
