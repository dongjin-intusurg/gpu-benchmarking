#!/usr/bin/env python3
"""ncu_kernels.csv -> per-kernel roofline points + time-weighted bound census.

Usage:
  roofline_report.py <ncu_kernels.csv> --ceilings <results.json> --device <cfg>
  roofline_report.py <ncu_kernels.csv> --ceil-fp16 X --ceil-int8 X --ceil-fp8 X
                     --ceil-cuda X --ceil-bw X [--device <cfg>]
  [--out <roofline.json>] [--top 10]

Both ceiling paths are first-class. The results.json path reads the run's
own measured ceilings block (tensor_ceiling_fp16_tflops, tensor_ceiling_int8_tops,
tensor_ceiling_fp8_tflops, cudacore_fp32_tflops, dram_gbps_idle) as emitted by
model_bench/run_model_bench.sh stage 5 from the device ceilings run
(device_ceilings/run_device_ceilings.sh). The idle bandwidth ceiling is the
correct roofline basis here because NCU runs solo-exclusive on a headless box.
Explicit --ceil-* flags override individual keys (archived-CSV replays); the
device config's optional fallback_ceilings block is the last resort, and every
non-default source is stamped into ceilings_used.

There are NO hardcoded ceiling defaults. A ceiling needed by a detected pipe
that is missing or null everywhere = exit 2 naming the key and the producing
stage. ops_per_tinst comes STRICTLY from the device config; a detected tensor
pipe without its constant = exit 2 naming model_bench/calibrate_ops_per_tinst.py.

Pre-declared thresholds and rules (declared here, not tuned post-hoc):
  latency-bound : kernel achieves < 30% of its own roofline roof - neither
                  pipe nor memory saturated (launch overhead, tiny grids, sync)
  memory-bound  : intensity < ridge (roof is the bandwidth slope) and >= 30%
  compute-bound : intensity >= ridge (roof is the flat pipe ceiling) and >= 30%
  unclassified  : no byte counter populated for the kernel (intensity unknown)
  tensor-pipe precision markers, checked in this order (first hit wins):
    fp8  : e4m3|e5m2|qmma|f8f8|_fp8   (checked BEFORE int8 - 'f8f8' and 'qmma'
                                       kernels must never fall into the i8 net)
    int8 : imma|i8i8|_i8|_s8_|tensorop_i<digit>|int8
    fp16 : hmma|h884|h1688|h16816|f16f16|_fp16|_f16|tensorop_h|hgmma
    none : pipe 'tensor_unknown' - classified against the LOWEST configured
           tensor ceiling (conservative roof), flops from the LOWEST configured
           ops_per_tinst; WARN when such kernels exceed 5% of total kernel time
  tensor_unknown WARN threshold: 5% of total kernel time
  n/a guard     : NCU rows whose Metric Value is not numeric ('n/a' for
                  unsupported counters) are dropped, never zero-filled
  bytes/frame fallback chain (first populated source wins, recorded per run):
    dram__bytes.sum                    -> 'dram'          (discrete GPUs)
    32 B x lts__t_sectors_lookup_miss  -> 'l2-miss-approx' (Jetson iGPU proxy)
    lts__t_bytes.sum | 32 B x sectors  -> 'lts-proxy'      (upper bound)

Measurement-honesty notes:
  - Durations are gpu__time_duration.sum under NCU replay clocks: they serve as
    time WEIGHTING and rate denominators only, never as latency of record (the
    p99 of record is the trtexec 1000-iteration pass).
  - Memory-bound verdicts are independent of ops_per_tinst: for a kernel under
    the bandwidth slope, frac = achieved/roof = (flops/dur)/(intensity x bw)
    = bytes/(dur x bw) - the flops cancel. The constant moves only the
    compute-side rates and the memory/compute split near the ridge.

Exit codes: 0 ok; 1 unreadable/degenerate input; 2 refused (missing ceiling or
missing ops_per_tinst constant - message names the fix).
"""
import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path

LATENCY_FRAC = 0.30           # below this fraction of own roof = latency-bound
TENSOR_UNKNOWN_WARN_PCT = 5.0 # unrecognized-tensor time share worth shouting about

FP8_MARKERS = re.compile(r'e4m3|e5m2|qmma|f8f8|_fp8')
INT8_MARKERS = re.compile(r'imma|i8i8|_i8|_s8_|tensorop_i\d|int8')
FP16_MARKERS = re.compile(r'hmma|h884|h1688|h16816|f16f16|_fp16|_f16|tensorop_h|hgmma')

# results.json ceilings-block key per pipe (the real emitted names)
KEY_FOR_PIPE = {
    'fp16': 'tensor_ceiling_fp16_tflops',
    'int8': 'tensor_ceiling_int8_tops',
    'fp8': 'tensor_ceiling_fp8_tflops',
    'cuda': 'cudacore_fp32_tflops',
}
BW_KEYS = ('dram_gbps_idle', 'bw_eff_gbps')  # same value under the solo regime
FLAG_FOR_PIPE = {'fp16': '--ceil-fp16', 'int8': '--ceil-int8',
                 'fp8': '--ceil-fp8', 'cuda': '--ceil-cuda'}
TENSOR_PIPES = ('int8', 'fp16', 'fp8')

PRODUCING_STAGE = ("produced by model_bench/run_model_bench.sh stage 5 (ceilings "
                   "block) from the device ceilings run "
                   "(device_ceilings/run_device_ceilings.sh)")


def warn(msg):
    print(f"WARN: {msg}", file=sys.stderr)


def die(msg, code=1):
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(code)


def friendly(name):
    """Plain-language label for a mangled kernel name (what it does, not what
    it's called). Order matters - first match wins."""
    n = name.lower()
    if 'copypackedkernel' in n or 'nhwctonchw' in n or 'nchwtonhwc' in n:
        return 'layout reformat (memory copy)'
    if 'permutationkernel' in n or 'cutensor' in n:
        return 'tensor transpose / permute'
    if 'stereo' in n and 'cv_cuda' in n:
        return 'cost volume (stereo correlation, custom plugin)'
    if 'conv3d' in n:
        return '3D conv (cost-volume filtering)'
    if 'depthwise' in n:
        return 'depthwise conv'
    if 'implicit_gemm' in n or 'tensorop_i' in n or re.search(r'xmma.*conv', n):
        prec = 'int8' if re.search(r'i8i8|_s8_|tensorop_i', n) else ('fp16' if 'f16f16' in n else 'fp32/tf32')
        return f'conv ({prec}, implicit GEMM)'
    if 'convex_upsample' in n or 'softargmax' in n:
        return 'disparity head (convex upsample + soft-argmax)'
    if 'upsample' in n or 'resize' in n or 'interp' in n:
        return 'upsample / resize'
    if 'instancenormalize' in n or 'normalizekernel' in n:
        return 'normalization'
    if 'softmax' in n:
        return 'softmax'
    if re.search(r'relu|sigmoid|tanh|elu\b|silu|gelu', n):
        return 'non-linear activation'
    if n.startswith('__myl_'):
        # fused elementwise: decode the op bundle from the generated kernel name
        ops = re.findall(r'[A-Z][a-z]+', name.split('_0x')[0])
        seq = set(ops)
        if {'Min', 'Max', 'Roun'} <= seq:
            return 'quantize (scale + clamp + round, fused)'
        if 'Resi' in seq:
            return 'residual add (fused elementwise)'
        if 'Conc' in seq:
            return 'concat / reshape (fused)'
        if 'Slic' in seq:
            return 'slice / copy (fused)'
        if seq <= {'Move', 'Cast', 'Resh'}:
            return 'tensor copy / cast'
        return 'fused elementwise (' + '.'.join(o.lower() for o in ops[:4]) + ')'
    if 'reduce' in n or 'argmin' in n or 'argmax' in n:
        return 'reduction / arg-select'
    if 'gemm' in n or 'gemv' in n:
        return 'matrix multiply'
    return 'other (' + name[:28] + ')'


def detect_pipe(name):
    """Tensor-pipe precision from kernel-name markers; fp8 checked first."""
    n = name.lower()
    if FP8_MARKERS.search(n):
        return 'fp8'
    if INT8_MARKERS.search(n):
        return 'int8'
    if FP16_MARKERS.search(n):
        return 'fp16'
    return 'tensor_unknown'


def load_rows(path):
    # skip the ==PROF== preamble; the real header row starts with "ID"
    try:
        with open(path, errors='ignore') as f:
            lines = list(f)
    except OSError as e:
        die(f"cannot read {path}: {e}")
    start = next((i for i, l in enumerate(lines)
                  if l.startswith('"ID"') or l.startswith('ID,')), None)
    if start is None:
        die(f"no NCU header row in {path} - not an ncu --csv log?")
    return list(csv.DictReader(lines[start:]))


def aggregate(rows):
    """Sum metric values per kernel launch, keyed on the NCU ID column."""
    K = collections.defaultdict(lambda: collections.defaultdict(float))
    names = {}
    for r in rows:
        k = r.get('ID', '')
        names[k] = r.get('Kernel Name', '?')
        try:
            v = float(str(r.get('Metric Value', '0')).replace(',', ''))
        except ValueError:
            continue  # 'n/a' rows (unsupported metric on this GPU) drop out here
        K[k][r.get('Metric Name', '')] += v
    return K, names


def numeric(v):
    """Ceiling values may arrive as 'unavailable: <reason>' strings - only a
    positive number counts as present."""
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 else None


def resolve_ceilings(args, needed_pipes, cfg):
    """Resolution order per key: explicit flag > results.json ceilings block >
    device-config fallback_ceilings. Returns ops/s (and bytes/s for bw) plus a
    per-key provenance map; missing needed key = exit 2."""
    flag_vals = {'fp16': args.ceil_fp16, 'int8': args.ceil_int8,
                 'fp8': args.ceil_fp8, 'cuda': args.ceil_cuda}
    cj_ceil, cj_path = {}, None
    if args.ceilings:
        cj_path = args.ceilings
        try:
            cj = json.load(open(args.ceilings))
        except (OSError, json.JSONDecodeError) as e:
            die(f"cannot read ceilings json {args.ceilings}: {e}")
        # ceilings results.json carries a 'ceilings' block; tolerate a bare
        # ceilings dict passed directly
        cj_ceil = cj.get('ceilings', cj) or {}
    fallback = (cfg or {}).get('fallback_ceilings') or {}

    vals, srcs = {}, {}

    def resolve_key(pipe_or_bw, key, flag_val, flag_name, required):
        if flag_val is not None:
            vals[key], srcs[key] = flag_val, 'flag'
            return
        v = numeric(cj_ceil.get(key))
        if v is None and key == 'dram_gbps_idle':      # bw alias in older blocks
            v = numeric(cj_ceil.get('bw_eff_gbps'))
        if v is not None:
            vals[key], srcs[key] = v, f'results.json:{cj_path}'
            return
        fv = numeric(fallback.get(key))
        if fv is not None:
            vals[key], srcs[key] = fv, 'device-config-fallback'
            return
        if required:
            if cj_path:
                die(f"needed ceiling '{key}' missing or null in {cj_path} - "
                    f"{PRODUCING_STAGE}. Re-run that stage, pass {flag_name}, or "
                    f"add fallback_ceilings.{key} to the device config.", 2)
            die(f"needed ceiling '{key}' not given - pass {flag_name} "
                f"(or use --ceilings <results.json>, {PRODUCING_STAGE}).", 2)

    # bandwidth underpins every ridge/roof - always required
    resolve_key('bw', 'dram_gbps_idle', args.ceil_bw, '--ceil-bw', required=True)
    for pipe in ('fp16', 'int8', 'fp8', 'cuda'):
        required = pipe in needed_pipes
        # tensor ceilings resolve softly even when their pipe is absent: the
        # tensor_unknown roof is the LOWEST tensor ceiling that exists
        soft = pipe in TENSOR_PIPES and 'tensor_unknown' in needed_pipes
        if required or soft:
            resolve_key(pipe, KEY_FOR_PIPE[pipe], flag_vals[pipe],
                        FLAG_FOR_PIPE[pipe], required=required)

    if 'tensor_unknown' in needed_pipes:
        tset = [vals[KEY_FOR_PIPE[p]] for p in TENSOR_PIPES if KEY_FOR_PIPE[p] in vals]
        if not tset:
            die("tensor_unknown kernels detected but no tensor ceiling is "
                "available anywhere (tensor_ceiling_{fp16,int8,fp8}) - "
                f"{PRODUCING_STAGE}, or pass one --ceil-fp16/--ceil-int8/--ceil-fp8.", 2)

    ceil_ops = {'bw': vals['dram_gbps_idle'] * 1e9}
    for pipe in ('fp16', 'int8', 'fp8', 'cuda'):
        if KEY_FOR_PIPE[pipe] in vals:
            ceil_ops[pipe] = vals[KEY_FOR_PIPE[pipe]] * 1e12
    if 'tensor_unknown' in needed_pipes:
        ceil_ops['tensor_unknown'] = min(ceil_ops[p] for p in TENSOR_PIPES if p in ceil_ops)
    return ceil_ops, vals, srcs


def resolve_ops_per_tinst(cfg, args, needed_pipes):
    """ops_per_tinst comes strictly from the device config - no defaults, no
    guesses (sm_110 constants do not transfer to sm_120)."""
    tensor_needed = [p for p in needed_pipes if p in TENSOR_PIPES or p == 'tensor_unknown']
    if not tensor_needed:
        return {}
    if cfg is None:
        die("tensor-pipe kernels detected but no --device config given - "
            "ops_per_tinst comes strictly from the device config "
            "(calibrated by model_bench/calibrate_ops_per_tinst.py).", 2)
    opt = cfg.get('ops_per_tinst')
    if not isinstance(opt, dict) or not opt:
        die(f"device config {args.device} has ops_per_tinst = "
            f"{json.dumps(opt)} - calibrate it on this box with "
            "model_bench/calibrate_ops_per_tinst.py (known-truth 4096^3 GEMM "
            "engine) and paste the printed constant into the config.", 2)
    consts = {k: v for k, v in opt.items() if numeric(v)}
    for p in tensor_needed:
        if p == 'tensor_unknown':
            if not consts:
                die(f"device config {args.device} ops_per_tinst has no usable "
                    "entries for tensor_unknown kernels - run "
                    "model_bench/calibrate_ops_per_tinst.py first.", 2)
            continue
        if p not in consts:
            die(f"detected tensor pipe '{p}' but device config {args.device} "
                f"has no ops_per_tinst['{p}'] - run "
                f"model_bench/calibrate_ops_per_tinst.py --precision {p} on the "
                "known-truth GEMM engine and paste the printed constant.", 2)
    if 'tensor_unknown' in tensor_needed:
        # lowest constant = most conservative flops estimate for unmarked kernels;
        # their memory-bound verdicts don't depend on it anyway (flops cancel)
        consts['tensor_unknown'] = min(consts.values())
    return consts


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('ncu_csv')
    ap.add_argument('--ceilings', help='ceilings results.json with the ceilings block')
    ap.add_argument('--device', help='device config json (ops_per_tinst source)')
    ap.add_argument('--ceil-fp16', type=float, help='fp16 tensor ceiling, TFLOPS')
    ap.add_argument('--ceil-int8', type=float, help='int8 tensor ceiling, TOPS')
    ap.add_argument('--ceil-fp8', type=float, help='fp8 tensor ceiling, TFLOPS')
    ap.add_argument('--ceil-cuda', type=float, help='CUDA-core fp32 ceiling, TFLOPS')
    ap.add_argument('--ceil-bw', type=float, help='bandwidth ceiling, GB/s')
    ap.add_argument('--out', help='write JSON here (default stdout)')
    ap.add_argument('--top', type=int, default=10)
    args = ap.parse_args()

    cfg = None
    if args.device:
        try:
            cfg = json.load(open(args.device))
        except (OSError, json.JSONDecodeError) as e:
            die(f"cannot read device config {args.device}: {e}")
    if args.ceilings and cfg is None:
        die("--ceilings mode requires --device (ops_per_tinst source).", 2)

    K, names = aggregate(load_rows(args.ncu_csv))

    # -- pass 1: per-kernel raw quantities + pipe detection --------------------
    byte_src = set()
    prelim = []
    for kid, m in K.items():
        name = names.get(kid, '?')
        dur = m.get('gpu__time_duration.sum', 0) / 1e9
        if dur <= 0:
            continue
        b = m.get('dram__bytes.sum', 0)
        if b:
            byte_src.add('dram')
        elif m.get('lts__t_sectors_lookup_miss.sum', 0):
            # L2 misses x 32B/sector ~ DRAM crossings (the Jetson-iGPU approximation)
            b = 32 * m['lts__t_sectors_lookup_miss.sum']
            byte_src.add('l2-miss-approx')
        else:
            b = m.get('lts__t_bytes.sum', 0) or 32 * (m.get('lts__t_sectors_op_read.sum', 0)
                                                      + m.get('lts__t_sectors_op_write.sum', 0))
            if b:
                byte_src.add('lts-proxy')
        tens = m.get('sm__inst_executed_pipe_tensor.sum', 0)
        cuda_flops = 2 * m.get('sm__sass_thread_inst_executed_op_ffma_pred_on.sum', 0) \
            + m.get('sm__sass_thread_inst_executed_op_fadd_pred_on.sum', 0) \
            + m.get('sm__sass_thread_inst_executed_op_fmul_pred_on.sum', 0) \
            + 2 * (2 * m.get('sm__sass_thread_inst_executed_op_hfma_pred_on.sum', 0)
                   + m.get('sm__sass_thread_inst_executed_op_hadd_pred_on.sum', 0)
                   + m.get('sm__sass_thread_inst_executed_op_hmul_pred_on.sum', 0))
        pipe = detect_pipe(name) if tens > 0 else 'cuda'
        prelim.append({'name': name, 'dur': dur, 'b': b, 'tens': tens,
                       'cuda_flops': cuda_flops, 'pipe': pipe})

    if not prelim:
        die(f"no kernels with positive duration in {args.ncu_csv}")

    needed_pipes = {k['pipe'] for k in prelim}
    ceil_ops, ceil_vals, ceil_srcs = resolve_ceilings(args, needed_pipes, cfg)
    ops_per_tinst = resolve_ops_per_tinst(cfg, args, needed_pipes)

    # -- pass 2: roofline point + bound per kernel -----------------------------
    out = []
    for k in prelim:
        pipe = k['pipe']
        flops = k['tens'] * ops_per_tinst[pipe] if k['tens'] > 0 else k['cuda_flops']
        achieved = flops / k['dur']                        # ops/s
        intensity = (flops / k['b']) if k['b'] > 0 else None  # ops/byte
        ridge = ceil_ops[pipe] / ceil_ops['bw']
        if intensity is None:
            roof, bound = None, 'unclassified'
        else:
            roof = min(ceil_ops[pipe], intensity * ceil_ops['bw'])
            frac = achieved / roof if roof else 0
            if frac < LATENCY_FRAC:
                bound = 'latency'
            elif intensity < ridge:
                bound = 'memory'
            else:
                bound = 'compute'
        out.append({'name': k['name'][:60], 'label': friendly(k['name']),
                    'us': round(k['dur'] * 1e6, 1), 'gb': round(k['b'] / 1e9, 5),
                    'pipe': pipe,
                    'intensity': round(intensity, 1) if intensity else None,
                    'achieved_tops': round(achieved / 1e12, 3),
                    'roof_tops': round(roof / 1e12, 3) if roof else None,
                    'pct_of_roof': round(100 * achieved / roof, 1) if roof else None,
                    'bound': bound})

    out.sort(key=lambda x: -x['us'])
    tot = sum(k['us'] for k in out) or 1

    warnings = []
    unk_pct = 100 * sum(k['us'] for k in out if k['pipe'] == 'tensor_unknown') / tot
    if unk_pct > TENSOR_UNKNOWN_WARN_PCT:
        msg = (f"tensor_unknown pipe holds {unk_pct:.1f}% of kernel time "
               f"(>{TENSOR_UNKNOWN_WARN_PCT:.0f}% threshold) - unmarked tensor "
               "kernels classified against the lowest tensor ceiling; inspect "
               "kernel names and extend the precision markers if a pattern emerges")
        warn(msg)
        warnings.append(msg)

    pipes_present = [p for p in ('int8', 'fp16', 'fp8', 'cuda', 'tensor_unknown')
                     if p in needed_pipes]
    ceilings_used = dict(ceil_vals)
    ceilings_used['ops_per_tinst'] = ops_per_tinst
    ceilings_used['ridge_ops_per_byte'] = {p: round(ceil_ops[p] / ceil_ops['bw'], 0)
                                           for p in pipes_present}
    ceilings_used['per_key_source'] = ceil_srcs
    ceilings_used['source'] = (f'results.json:{args.ceilings}' if args.ceilings
                               else 'flags')
    if any(s == 'device-config-fallback' for s in ceil_srcs.values()):
        fb = sorted(k for k, s in ceil_srcs.items() if s == 'device-config-fallback')
        ceilings_used['source'] += '+device-config-fallback(' + ','.join(fb) + ')'

    summary = {
        'time_pct_by_bound': {b: round(100 * sum(k['us'] for k in out if k['bound'] == b) / tot, 1)
                              for b in ('memory', 'compute', 'latency', 'unclassified')},
        'time_pct_by_pipe': {p: round(100 * sum(k['us'] for k in out if k['pipe'] == p) / tot, 1)
                             for p in pipes_present},
        'bytes_per_frame_gb': round(sum(k['gb'] for k in out), 4),
        'byte_source': '+'.join(sorted(byte_src)) or 'none',
        'kernels_total': len(out),
        'ceilings_used': ceilings_used,
        'top_kernels': [{'label': k['label'], 'us': k['us'],
                         'time_pct': round(100 * k['us'] / tot, 1),
                         'pipe': k['pipe'], 'bound': k['bound'],
                         'pct_of_roof': k['pct_of_roof'],
                         'intensity': k['intensity']}
                        for k in out[:args.top]],
    }
    if warnings:
        summary['warnings'] = warnings

    doc = json.dumps({'summary': summary, 'kernels': out}, indent=1)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(doc + '\n')
    else:
        print(doc)
    print(f"// {len(out)} kernels; bytes via {summary['byte_source']}; "
          f"bound %: {summary['time_pct_by_bound']}", file=sys.stderr)


if __name__ == '__main__':
    main()
