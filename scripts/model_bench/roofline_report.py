#!/usr/bin/env python3
"""ncu_kernels.csv -> per-kernel roofline points + time-weighted bound census.

Writes {'summary', 'kernels'} JSON to --out (or stdout) and one '// N kernels; ...' line to stderr.
Ceilings resolve per key as explicit --ceil-* flag > --ceilings results.json ceilings block > the device
config's fallback_ceilings; there are no hardcoded defaults, and ops_per_tinst comes strictly from the
device config. NCU durations serve as time weights and rate denominators only, never as latency of record.
Exit codes: 0 ok; 1 unreadable/degenerate input; 2 refused (missing ceiling or ops_per_tinst constant).
"""
import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path

# Pre-declared classification thresholds (declared here, never tuned post-hoc):
# a kernel achieving < 30% of its own roof is latency-bound (launch overhead, tiny grids, sync);
# unmarked tensor kernels are a concern once they hold > 5% of the total kernel time.
LATENCY_FRAC = 0.30
TENSOR_UNKNOWN_WARN_PCT = 5.0

# fp8 is checked before int8: 'f8f8' and 'qmma' kernels must never fall into the i8 net
FP8_MARKERS = re.compile(r'e4m3|e5m2|qmma|f8f8|_fp8')
INT8_MARKERS = re.compile(r'imma|i8i8|_i8|_s8_|tensorop_i\d|int8')
FP16_MARKERS = re.compile(r'hmma|h884|h1688|h16816|f16f16|_fp16|_f16|tensorop_h|hgmma')

KEY_FOR_PIPE = {
    'fp16': 'tensor_ceiling_fp16_tflops',
    'int8': 'tensor_ceiling_int8_tops',
    'fp8': 'tensor_ceiling_fp8_tflops',
    'cuda': 'cudacore_fp32_tflops',
}
BW_KEY = 'dram_gbps_idle'
BW_KEY_ALIAS = 'bw_eff_gbps'          # older ceilings blocks; same value under the solo regime
FLAG_FOR_PIPE = {'fp16': '--ceil-fp16', 'int8': '--ceil-int8', 'fp8': '--ceil-fp8', 'cuda': '--ceil-cuda'}
TENSOR_PIPES = ('int8', 'fp16', 'fp8')
PIPE_ORDER = ('int8', 'fp16', 'fp8', 'cuda', 'tensor_unknown')
BOUNDS = ('memory', 'compute', 'latency', 'unclassified')

PRODUCING_STAGE = ("produced by model_bench/run_model_bench.sh stage 5 (ceilings "
                   "block) from the device ceilings run "
                   "(device_ceilings/run_device_ceilings.sh)")


def warn(msg):
    print(f"WARN: {msg}", file=sys.stderr)


def die(msg, code=1):
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(code)


def friendly_fused_label(name):
    """Decode the op bundle a generated fused-elementwise kernel name carries."""
    ops = re.findall(r'[A-Z][a-z]+', name.split('_0x')[0])
    op_set = set(ops)
    if {'Min', 'Max', 'Roun'} <= op_set:
        return 'quantize (scale + clamp + round, fused)'
    if 'Resi' in op_set:
        return 'residual add (fused elementwise)'
    if 'Conc' in op_set:
        return 'concat / reshape (fused)'
    if 'Slic' in op_set:
        return 'slice / copy (fused)'
    if op_set <= {'Move', 'Cast', 'Resh'}:
        return 'tensor copy / cast'
    return 'fused elementwise (' + '.'.join(op.lower() for op in ops[:4]) + ')'


def friendly(name):
    """Plain-language label for a mangled kernel name (what it does, not what it is called);
    first match wins."""
    lowered = name.lower()
    if 'copypackedkernel' in lowered or 'nhwctonchw' in lowered or 'nchwtonhwc' in lowered:
        return 'layout reformat (memory copy)'
    if 'permutationkernel' in lowered or 'cutensor' in lowered:
        return 'tensor transpose / permute'
    if 'stereo' in lowered and 'cv_cuda' in lowered:
        return 'cost volume (stereo correlation, custom plugin)'
    if 'conv3d' in lowered:
        return '3D conv (cost-volume filtering)'
    if 'depthwise' in lowered:
        return 'depthwise conv'
    if 'implicit_gemm' in lowered or 'tensorop_i' in lowered or re.search(r'xmma.*conv', lowered):
        if re.search(r'i8i8|_s8_|tensorop_i', lowered):
            precision = 'int8'
        else:
            precision = 'fp16' if 'f16f16' in lowered else 'fp32/tf32'
        return f'conv ({precision}, implicit GEMM)'
    if 'convex_upsample' in lowered or 'softargmax' in lowered:
        return 'disparity head (convex upsample + soft-argmax)'
    if 'upsample' in lowered or 'resize' in lowered or 'interp' in lowered:
        return 'upsample / resize'
    if 'instancenormalize' in lowered or 'normalizekernel' in lowered:
        return 'normalization'
    if 'softmax' in lowered:
        return 'softmax'
    if re.search(r'relu|sigmoid|tanh|elu\b|silu|gelu', lowered):
        return 'non-linear activation'
    if lowered.startswith('__myl_'):
        return friendly_fused_label(name)
    if 'reduce' in lowered or 'argmin' in lowered or 'argmax' in lowered:
        return 'reduction / arg-select'
    if 'gemm' in lowered or 'gemv' in lowered:
        return 'matrix multiply'
    return 'other (' + name[:28] + ')'


def detect_pipe(name):
    """Tensor-pipe precision from kernel-name markers; fp8 checked first."""
    lowered = name.lower()
    if FP8_MARKERS.search(lowered):
        return 'fp8'
    if INT8_MARKERS.search(lowered):
        return 'int8'
    if FP16_MARKERS.search(lowered):
        return 'fp16'
    return 'tensor_unknown'


def load_rows(path):
    """The NCU csv rows after the ==PROF== preamble (the real header row starts with "ID")."""
    try:
        with open(path, errors='ignore') as handle:
            lines = list(handle)
    except OSError as exc:
        die(f"cannot read {path}: {exc}")
    start = next((i for i, line in enumerate(lines)
                  if line.startswith('"ID"') or line.startswith('ID,')), None)
    if start is None:
        die(f"no NCU header row in {path} - not an ncu --csv log?")
    return list(csv.DictReader(lines[start:]))


def aggregate(rows):
    """Sum metric values per kernel launch, keyed on the NCU ID column; 'n/a' rows
    (unsupported counters) are dropped, never zero-filled."""
    metrics_by_kernel = collections.defaultdict(lambda: collections.defaultdict(float))
    names = {}
    for row in rows:
        kernel_id = row.get('ID', '')
        names[kernel_id] = row.get('Kernel Name', '?')
        try:
            value = float(str(row.get('Metric Value', '0')).replace(',', ''))
        except ValueError:
            continue
        metrics_by_kernel[kernel_id][row.get('Metric Name', '')] += value
    return metrics_by_kernel, names


def numeric(value):
    """Ceiling values may arrive as 'unavailable: <reason>' strings - only a positive number counts."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return value
    return None


def bytes_per_kernel(metrics, byte_sources):
    """Bytes moved by one kernel, first populated source wins: dram__bytes.sum (discrete GPUs),
    32 B x L2 lookup misses (the Jetson iGPU proxy for DRAM crossings), then the L2 traffic
    itself as an upper bound."""
    moved = metrics.get('dram__bytes.sum', 0)
    if moved:
        byte_sources.add('dram')
        return moved
    if metrics.get('lts__t_sectors_lookup_miss.sum', 0):
        byte_sources.add('l2-miss-approx')
        return 32 * metrics['lts__t_sectors_lookup_miss.sum']
    moved = metrics.get('lts__t_bytes.sum', 0) or 32 * (metrics.get('lts__t_sectors_op_read.sum', 0)
                                                        + metrics.get('lts__t_sectors_op_write.sum', 0))
    if moved:
        byte_sources.add('lts-proxy')
    return moved


def cuda_core_flops(metrics):
    """CUDA-core flops from the SASS thread-instruction counters (FMA = 2, half ops x2 for the pair)."""
    return (2 * metrics.get('sm__sass_thread_inst_executed_op_ffma_pred_on.sum', 0)
            + metrics.get('sm__sass_thread_inst_executed_op_fadd_pred_on.sum', 0)
            + metrics.get('sm__sass_thread_inst_executed_op_fmul_pred_on.sum', 0)
            + 2 * (2 * metrics.get('sm__sass_thread_inst_executed_op_hfma_pred_on.sum', 0)
                   + metrics.get('sm__sass_thread_inst_executed_op_hadd_pred_on.sum', 0)
                   + metrics.get('sm__sass_thread_inst_executed_op_hmul_pred_on.sum', 0)))


def collect_kernels(metrics_by_kernel, names, byte_sources):
    """Per-kernel raw quantities + pipe detection; kernels without a positive duration drop out."""
    kernels = []
    for kernel_id, metrics in metrics_by_kernel.items():
        name = names.get(kernel_id, '?')
        duration_s = metrics.get('gpu__time_duration.sum', 0) / 1e9
        if duration_s <= 0:
            continue
        tensor_insts = metrics.get('sm__inst_executed_pipe_tensor.sum', 0)
        kernels.append({'name': name, 'duration_s': duration_s,
                        'bytes': bytes_per_kernel(metrics, byte_sources),
                        'tensor_insts': tensor_insts, 'cuda_flops': cuda_core_flops(metrics),
                        'pipe': detect_pipe(name) if tensor_insts > 0 else 'cuda'})
    return kernels


def resolve_ceilings(args, needed_pipes, config):
    """Per key: explicit flag > results.json ceilings block > device-config fallback_ceilings.
    Returns ops/s per pipe (bytes/s for 'bw'), the raw values, and a per-key provenance map;
    a missing needed key is exit 2."""
    flag_values = {'fp16': args.ceil_fp16, 'int8': args.ceil_int8,
                   'fp8': args.ceil_fp8, 'cuda': args.ceil_cuda}
    ceilings_block, ceilings_path = {}, None
    if args.ceilings:
        ceilings_path = args.ceilings
        try:
            ceilings_doc = json.load(open(args.ceilings))
        except (OSError, json.JSONDecodeError) as exc:
            die(f"cannot read ceilings json {args.ceilings}: {exc}")
        # a results.json carries a 'ceilings' block; tolerate a bare ceilings dict passed directly
        ceilings_block = ceilings_doc.get('ceilings', ceilings_doc) or {}
    fallback = (config or {}).get('fallback_ceilings') or {}

    values, sources = {}, {}

    def resolve_key(key, flag_value, flag_name, required):
        if flag_value is not None:
            values[key], sources[key] = flag_value, 'flag'
            return
        value = numeric(ceilings_block.get(key))
        if value is None and key == BW_KEY:
            value = numeric(ceilings_block.get(BW_KEY_ALIAS))
        if value is not None:
            values[key], sources[key] = value, f'results.json:{ceilings_path}'
            return
        fallback_value = numeric(fallback.get(key))
        if fallback_value is not None:
            values[key], sources[key] = fallback_value, 'device-config-fallback'
            return
        if not required:
            return
        if ceilings_path:
            die(f"needed ceiling '{key}' missing or null in {ceilings_path} - "
                f"{PRODUCING_STAGE}. Re-run that stage, pass {flag_name}, or "
                f"add fallback_ceilings.{key} to the device config.", 2)
        die(f"needed ceiling '{key}' not given - pass {flag_name} "
            f"(or use --ceilings <results.json>, {PRODUCING_STAGE}).", 2)

    # bandwidth underpins every ridge/roof - always required
    resolve_key(BW_KEY, args.ceil_bw, '--ceil-bw', required=True)
    unknown_present = 'tensor_unknown' in needed_pipes
    for pipe in ('fp16', 'int8', 'fp8', 'cuda'):
        required = pipe in needed_pipes
        # tensor ceilings resolve softly even when their pipe is absent: the tensor_unknown
        # roof is the LOWEST tensor ceiling that exists
        soft = pipe in TENSOR_PIPES and unknown_present
        if required or soft:
            resolve_key(KEY_FOR_PIPE[pipe], flag_values[pipe], FLAG_FOR_PIPE[pipe], required=required)

    if unknown_present and not any(KEY_FOR_PIPE[pipe] in values for pipe in TENSOR_PIPES):
        die("tensor_unknown kernels detected but no tensor ceiling is "
            "available anywhere (tensor_ceiling_{fp16,int8,fp8}) - "
            f"{PRODUCING_STAGE}, or pass one --ceil-fp16/--ceil-int8/--ceil-fp8.", 2)

    ceiling_ops = {'bw': values[BW_KEY] * 1e9}
    for pipe in ('fp16', 'int8', 'fp8', 'cuda'):
        if KEY_FOR_PIPE[pipe] in values:
            ceiling_ops[pipe] = values[KEY_FOR_PIPE[pipe]] * 1e12
    if unknown_present:
        ceiling_ops['tensor_unknown'] = min(ceiling_ops[pipe] for pipe in TENSOR_PIPES if pipe in ceiling_ops)
    return ceiling_ops, values, sources


def resolve_ops_per_tinst(config, args, needed_pipes):
    """ops_per_tinst comes strictly from the device config - no defaults, no guesses
    (sm_110 constants do not transfer to sm_120)."""
    tensor_needed = [pipe for pipe in needed_pipes if pipe in TENSOR_PIPES or pipe == 'tensor_unknown']
    if not tensor_needed:
        return {}
    if config is None:
        die("tensor-pipe kernels detected but no --device config given - "
            "ops_per_tinst comes strictly from the device config "
            "(calibrated by model_bench/calibrate_ops_per_tinst.py).", 2)
    configured = config.get('ops_per_tinst')
    if not isinstance(configured, dict) or not configured:
        die(f"device config {args.device} has ops_per_tinst = "
            f"{json.dumps(configured)} - calibrate it on this box with "
            "model_bench/calibrate_ops_per_tinst.py (known-truth 4096^3 GEMM "
            "engine) and paste the printed constant into the config.", 2)
    constants = {key: value for key, value in configured.items() if numeric(value)}
    for pipe in tensor_needed:
        if pipe == 'tensor_unknown':
            if not constants:
                die(f"device config {args.device} ops_per_tinst has no usable "
                    "entries for tensor_unknown kernels - run "
                    "model_bench/calibrate_ops_per_tinst.py first.", 2)
            continue
        if pipe not in constants:
            die(f"detected tensor pipe '{pipe}' but device config {args.device} "
                f"has no ops_per_tinst['{pipe}'] - run "
                f"model_bench/calibrate_ops_per_tinst.py --precision {pipe} on the "
                "known-truth GEMM engine and paste the printed constant.", 2)
    if 'tensor_unknown' in tensor_needed:
        # lowest constant = most conservative flops estimate for unmarked kernels; their
        # memory-bound verdicts do not depend on it anyway (the flops cancel under the slope)
        constants['tensor_unknown'] = min(constants.values())
    return constants


def classify(kernel, ceiling_ops, ops_per_tinst):
    """Roofline point + bound for one kernel."""
    pipe = kernel['pipe']
    if kernel['tensor_insts'] > 0:
        flops = kernel['tensor_insts'] * ops_per_tinst[pipe]
    else:
        flops = kernel['cuda_flops']
    achieved = flops / kernel['duration_s']
    intensity = (flops / kernel['bytes']) if kernel['bytes'] > 0 else None
    ridge = ceiling_ops[pipe] / ceiling_ops['bw']
    if intensity is None:
        roof, bound = None, 'unclassified'
    else:
        roof = min(ceiling_ops[pipe], intensity * ceiling_ops['bw'])
        fraction_of_roof = achieved / roof if roof else 0
        if fraction_of_roof < LATENCY_FRAC:
            bound = 'latency'
        elif intensity < ridge:
            bound = 'memory'
        else:
            bound = 'compute'
    return {'name': kernel['name'][:60], 'label': friendly(kernel['name']),
            'us': round(kernel['duration_s'] * 1e6, 1), 'gb': round(kernel['bytes'] / 1e9, 5),
            'pipe': pipe,
            'intensity': round(intensity, 1) if intensity else None,
            'achieved_tops': round(achieved / 1e12, 3),
            'roof_tops': round(roof / 1e12, 3) if roof else None,
            'pct_of_roof': round(100 * achieved / roof, 1) if roof else None,
            'bound': bound}


def time_pct(points, total_us, predicate):
    return round(100 * sum(point['us'] for point in points if predicate(point)) / total_us, 1)


def build_summary(points, total_us, args, needed_pipes, byte_sources, ceiling_ops, ceiling_values,
                  ceiling_sources, ops_per_tinst, warnings):
    pipes_present = [pipe for pipe in PIPE_ORDER if pipe in needed_pipes]
    ceilings_used = dict(ceiling_values)
    ceilings_used['ops_per_tinst'] = ops_per_tinst
    ceilings_used['ridge_ops_per_byte'] = {pipe: round(ceiling_ops[pipe] / ceiling_ops['bw'], 0)
                                           for pipe in pipes_present}
    ceilings_used['per_key_source'] = ceiling_sources
    ceilings_used['source'] = f'results.json:{args.ceilings}' if args.ceilings else 'flags'
    fallback_keys = sorted(key for key, source in ceiling_sources.items()
                           if source == 'device-config-fallback')
    if fallback_keys:
        ceilings_used['source'] += '+device-config-fallback(' + ','.join(fallback_keys) + ')'

    summary = {
        'time_pct_by_bound': {bound: time_pct(points, total_us, lambda point, b=bound: point['bound'] == b)
                              for bound in BOUNDS},
        'time_pct_by_pipe': {pipe: time_pct(points, total_us, lambda point, p=pipe: point['pipe'] == p)
                             for pipe in pipes_present},
        'bytes_per_frame_gb': round(sum(point['gb'] for point in points), 4),
        'byte_source': '+'.join(sorted(byte_sources)) or 'none',
        'kernels_total': len(points),
        'ceilings_used': ceilings_used,
        'top_kernels': [{'label': point['label'], 'us': point['us'],
                         'time_pct': round(100 * point['us'] / total_us, 1),
                         'pipe': point['pipe'], 'bound': point['bound'],
                         'pct_of_roof': point['pct_of_roof'],
                         'intensity': point['intensity']}
                        for point in points[:args.top]],
    }
    if warnings:
        summary['warnings'] = warnings
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('ncu_csv')
    parser.add_argument('--ceilings', help='ceilings results.json with the ceilings block')
    parser.add_argument('--device', help='device config json (ops_per_tinst source)')
    parser.add_argument('--ceil-fp16', type=float, help='fp16 tensor ceiling, TFLOPS')
    parser.add_argument('--ceil-int8', type=float, help='int8 tensor ceiling, TOPS')
    parser.add_argument('--ceil-fp8', type=float, help='fp8 tensor ceiling, TFLOPS')
    parser.add_argument('--ceil-cuda', type=float, help='CUDA-core fp32 ceiling, TFLOPS')
    parser.add_argument('--ceil-bw', type=float, help='bandwidth ceiling, GB/s')
    parser.add_argument('--out', help='write JSON here (default stdout)')
    parser.add_argument('--top', type=int, default=10)
    args = parser.parse_args()

    config = None
    if args.device:
        try:
            config = json.load(open(args.device))
        except (OSError, json.JSONDecodeError) as exc:
            die(f"cannot read device config {args.device}: {exc}")
    if args.ceilings and config is None:
        die("--ceilings mode requires --device (ops_per_tinst source).", 2)

    metrics_by_kernel, names = aggregate(load_rows(args.ncu_csv))
    byte_sources = set()
    kernels = collect_kernels(metrics_by_kernel, names, byte_sources)
    if not kernels:
        die(f"no kernels with positive duration in {args.ncu_csv}")

    needed_pipes = {kernel['pipe'] for kernel in kernels}
    ceiling_ops, ceiling_values, ceiling_sources = resolve_ceilings(args, needed_pipes, config)
    ops_per_tinst = resolve_ops_per_tinst(config, args, needed_pipes)

    points = [classify(kernel, ceiling_ops, ops_per_tinst) for kernel in kernels]
    points.sort(key=lambda point: -point['us'])
    total_us = sum(point['us'] for point in points) or 1

    warnings = []
    unknown_pct = 100 * sum(point['us'] for point in points if point['pipe'] == 'tensor_unknown') / total_us
    if unknown_pct > TENSOR_UNKNOWN_WARN_PCT:
        msg = (f"tensor_unknown pipe holds {unknown_pct:.1f}% of kernel time "
               f"(>{TENSOR_UNKNOWN_WARN_PCT:.0f}% threshold) - unmarked tensor "
               "kernels classified against the lowest tensor ceiling; inspect "
               "kernel names and extend the precision markers if a pattern emerges")
        warn(msg)
        warnings.append(msg)

    summary = build_summary(points, total_us, args, needed_pipes, byte_sources, ceiling_ops,
                            ceiling_values, ceiling_sources, ops_per_tinst, warnings)
    doc = json.dumps({'summary': summary, 'kernels': points}, indent=1)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(doc + '\n')
    else:
        print(doc)
    print(f"// {len(points)} kernels; bytes via {summary['byte_source']}; "
          f"bound %: {summary['time_pct_by_bound']}", file=sys.stderr)


if __name__ == '__main__':
    main()
