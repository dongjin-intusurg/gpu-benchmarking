#!/usr/bin/env python3
"""Attainable per-precision compute ceilings via TensorRT — GEMM kernel ONLY.

Measures the attainable GEMM peak for each quantization level (fp16, int8,
fp8) with the same instrument the model half uses — a trtexec-built engine at
--builderOptimizationLevel=5 (maximum tactic search) — but times ONLY the GEMM
kernel. A compute-ceiling probe must never include casting, quantize, or
reformat kernels in its timed window: those are memory-bound passes, so their
cost scales with 1/bandwidth and poisons the ceiling exactly on the devices
where bandwidth is scarce. Measured consequence that forced this design: on a
~254 GB/s unified-memory part the fp32->typed input conversions added ~10 ms
around a ~28 ms int8 GEMM and understated the compute ceiling by ~2x, while on
a ~1100 GB/s discrete part the same overhead was ~2% and invisible.

Two defenses, both always on (pre-declared, not post-hoc):
  1. PURITY BY CONSTRUCTION — engines are built so no conversion kernel exists:
       fp16: the ONNX declares fp16 tensors and the engine uses fp16 I/O
             (--inputIOFormats/--outputIOFormats fp16:chw) -> single GEMM kernel.
       int8: explicit QDQ graph + int8 engine I/O (the boundary absorbs the
             quantize) + fp16 output -> single fused DQ-GEMM kernel.
       fp8:  no fp8 I/O type exists in trtexec, so the quantize kernels cannot
             be eliminated — fp8 relies on defense 2 alone. Inputs are declared
             fp16 (not fp32) to at least halve the conversion traffic.
  2. PURITY BY MEASUREMENT — every point runs with --separateProfileRun and
     exports the per-layer profile. The attainable value is computed from the
     GEMM layer's own time; if the engine turned out pure (GEMM >= 99% of the
     profile) the end-to-end median is used instead (it is the same number
     without profiling instrumentation). The kernel census, GEMM share, and
     which method produced the value are recorded per point — a contaminated
     probe can never masquerade as a clean one.

GEMM-layer identification: layers whose name contains the ONNX node name
('gemm_core'), 'MatMul', or 'gemm' are summed; if nothing matches (fused
kernels on some backends are renamed wholesale), the single largest layer is
taken — the n^3 kernel dominates any conversion pass by construction at the
probed sizes. The census in the output makes this choice auditable.

Attainable = 2*n^3 / gemm_kernel_time. Burst ceiling = best over the size
sweep; the winning N is reported because the peak size is itself a device
characteristic. Quantized scales are arbitrary (they select kernels, not
timing). Build at opt level 5; time with --noDataTransfers --useSpinWait.

Burst pass: after the size sweep, each precision's winning variant is re-run
with --idleTime gaps between inferences. The gaps let the power state recover
and the clock boost before every kernel, so the pass captures the true burst
kernel rate on parts whose governor otherwise caps any continuous tensor load
(measured motivation: the discrete 300 W card settles ~7% below its short-load
clock within milliseconds, so a gapless sweep can never observe its burst,
while the unified-memory part gets natural gaps from its slow memory phases
and shows bursts up to 93% of datasheet). Burst runs join the sweep as
ordinary points tagged idle_ms, so best-over-sweep is the true peak either
way and the numbers stay auditable.

Env overrides:
  TRT_GEMM_SIZES        square sizes, space/comma list
                        (default "2048 4096 8192 12288 16384")
  TRT_GEMM_PRECISIONS   subset of "fp16 int8 fp8 fp32"  (default all four;
                        re-runs merge into an existing trt_compute.json)
  TRT_ITERATIONS        trtexec timing iterations       (default 300)
  TRT_DURATION_S        trtexec timing duration seconds (default 5)
  TRT_WARMUP_MS         trtexec warmup milliseconds     (default 1000)
  TRT_BUILD_TIMEOUT_S   per-build wall timeout seconds  (default 1800)
  TRT_BURST_IDLE_MS     idle gap for the burst pass; 0 disables (default 20)
  TRT_BURST_SIZES       sizes for the burst pass (default: winning N + 4096)

Writes <out>/trt_compute.json. stdlib + numpy + onnx.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPT_LEVEL = 5
UNIT = {'fp16': 'TFLOPS', 'int8': 'TOPS', 'fp8': 'TFLOPS', 'fp32': 'TFLOPS'}
GPU_COMPUTE_RE = re.compile(r'GPU Compute Time:.*?median\s*=\s*([0-9.]+)\s*ms', re.IGNORECASE)

# A layer is GEMM if its name carries the ONNX node name or an obvious GEMM
# marker; everything else (QuantizeLinear, Reformat, Cast, copy nodes) is
# overhead by definition of this probe. Conversion layers are often NAMED
# AFTER their consumer ("Reformatting CopyNode for Input Tensor 0 to
# gemm_core"), so an exclusion pattern outranks the GEMM match — otherwise the
# purity guarantee dies by nomenclature.
GEMM_NAME_RE = re.compile(r'gemm_core|matmul|gemm', re.IGNORECASE)
OVERHEAD_NAME_RE = re.compile(r'reformat|copy|cast|quant', re.IGNORECASE)

# Per-precision engine recipe: trtexec precision flags (byte-identical policy
# to build_model_engines.sh) + the I/O-binding variants to search. Binding
# formats are part of the search space: EVERY variant that builds is measured
# and the best GEMM rate wins. An I/O-format restriction can steer TRT to a
# slower tactic (measured: typed-fp16 I/O on the unified-memory part picked a
# 30 TFLOPS kernel where the default-I/O engine's GEMM layer runs >130), and
# a probe cannot know a kernel is slow without measuring the alternative.
# Variant dimensions searched per precision:
#   graph — activation x activation ('..') vs weight x activation ('.._wb',
#           B as an initializer). Model GEMMs are overwhelmingly weight-form
#           (linears, convs); activation-form is the attention corner case.
#           Measured trap: the graph compiler's fp16 activation-form MatMul ran ~31
#           TFLOPS on one unified-memory part regardless of I/O binding,
#           while the quantized paths and weight-form hit the tuned kernels.
#   io    — typed bindings (single-input for weight-form graphs) vs default.
IO1_FP16 = ['--inputIOFormats=fp16:chw', '--outputIOFormats=fp16:chw']
IO2_FP16 = ['--inputIOFormats=fp16:chw,fp16:chw', '--outputIOFormats=fp16:chw']
IO2_INT8 = ['--inputIOFormats=int8:chw,int8:chw', '--outputIOFormats=fp16:chw']
RECIPES = {
    'fp16': {
        'flags': ['--fp16'],
        'variants': [('fp16', IO2_FP16),          # activation-form (attention shape)
                     ('fp16_wb', IO1_FP16),       # weight-form MatMul (linear shape)
                     ('fp16_wb', []),
                     ('fp16_conv', [])],          # 1x1-conv form (implicit-GEMM path)
    },
    'int8': {
        'flags': ['--int8', '--fp16'],
        'variants': [('qdq-int8', IO2_INT8), ('qdq-int8', IO2_FP16),
                     ('qdq-int8_wb', IO1_FP16)],
    },
    'fp8': {
        'flags': ['--fp8', '--fp16'],
        'variants': [('qdq-fp8', IO2_FP16),
                     ('qdq-fp8_wb', IO1_FP16)],
    },
    # CUDA-core fp32 through the same instrument. --noTF32 is load-bearing:
    # TensorRT runs fp32 GEMMs on the TF32 tensor pipe BY DEFAULT, which would
    # silently measure the wrong silicon; disabling it pins the CUDA cores the
    # datasheet fp32 figure prices. Smaller sizes + fewer iterations because
    # fp32 GEMMs run ~50x slower than the tensor precisions.
    'fp32': {
        'flags': ['--noTF32'],
        'variants': [('fp32_wb', []),
                     ('fp32_conv', [])],
        'sizes': [2048, 4096, 8192],
        'iterations': 50,
    },
}

QDQ = {
    'qdq-int8': (TensorProto.INT8, 0.02),
    'qdq-fp8':  (TensorProto.FLOAT8E4M3FN, 0.25),
}


def say(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def find_trtexec():
    exe = shutil.which('trtexec')
    if exe:
        return exe
    for d in ('/usr/src/tensorrt/bin', '/opt/tensorrt/10.13/bin'):
        cand = os.path.join(d, 'trtexec')
        if os.path.exists(cand):
            return cand
    return None


def trt_version(trtexec):
    try:
        p = subprocess.run([trtexec, '--help'], capture_output=True, text=True, timeout=30)
        out = (p.stdout or '') + (p.stderr or '')
        m = re.search(r'TensorRT[^0-9]*([0-9]+\.[0-9]+\.[0-9]+)', out)
        if m:
            return m.group(1)
        m = re.search(r'TensorRT\s+v([0-9]{5,7})', out)
        if m:
            v = int(m.group(1))
            return f'{v // 10000}.{(v // 100) % 100}.{v % 100}'
    except Exception:
        pass
    return '?'


def make_gemm_onnx(n, path, graph):
    """A[n,n] x B[n,n] -> Y[n,n], node name 'gemm_core' so the profile row is
    identifiable. '_wb' graphs make B a random fp16 weight initializer
    (weight-form GEMM — the shape of every linear/conv; random values defeat
    sparsity shortcuts); otherwise B is a second activation input. QDQ graphs
    quantize both operands — the weight-side QuantizeLinear folds at build
    time, exactly like a real quantized model's weights. Scales pin the
    tensor-core kernel; values are irrelevant to a throughput probe."""
    # fp32 graphs carry fp32 tensors (CUDA-core probe); everything else fp16.
    dt = TensorProto.FLOAT if graph.startswith('fp32') else TensorProto.FLOAT16
    npdt = np.float32 if graph.startswith('fp32') else np.float16
    if graph.endswith('_conv'):
        # 1x1-conv form: Cin=Cout=n, HxW=n -> FLOPs = 2*Cout*Cin*H*W = 2*n^3,
        # identical work routed through the implicit-GEMM conv path — the
        # kernel family every vision model's layers actually run on. Spatial
        # size is the nearest-square factorization of n (e.g. 12288 -> 96x128).
        h = int(n ** 0.5)
        while n % h:
            h -= 1
        w_ = n // h
        A = helper.make_tensor_value_info('A', dt, [1, n, h, w_])
        Y = helper.make_tensor_value_info('Y', dt, [1, n, h, w_])
        w = np.random.default_rng(7).standard_normal((n, n, 1, 1)).astype(npdt)
        inits = [numpy_helper.from_array(w, name='B')]
        nodes = [helper.make_node('Conv', ['A', 'B'], ['Y'], name='gemm_core',
                                  kernel_shape=[1, 1])]
        g = helper.make_graph(nodes, f'gemm_{graph}_{n}', [A], [Y], initializer=inits)
        model = helper.make_model(g, opset_imports=[helper.make_opsetid('', 17)])
        model.ir_version = 9
        onnx.save(model, path)
        return
    wb = graph.endswith('_wb')
    base = graph[:-3] if wb else graph
    A = helper.make_tensor_value_info('A', dt, [n, n])
    Y = helper.make_tensor_value_info('Y', dt, [n, n])
    inputs, inits, nodes = [A], [], []
    if wb:
        w = np.random.default_rng(7).standard_normal((n, n)).astype(npdt)
        inits.append(numpy_helper.from_array(w, name='B'))
    else:
        inputs.append(helper.make_tensor_value_info('B', dt, [n, n]))
    if base in ('fp16', 'fp32'):
        a_in, b_in, opset = 'A', 'B', 17
    else:
        zp_dtype, scale = QDQ[base]
        inits += [helper.make_tensor('qscale', TensorProto.FLOAT16, [], [scale]),
                  helper.make_tensor('qzp', zp_dtype, [], [0.0])]
        for src in ('A', 'B'):
            nodes.append(helper.make_node('QuantizeLinear', [src, 'qscale', 'qzp'],
                                          [f'{src}_q'], name=f'{src}_quant'))
            nodes.append(helper.make_node('DequantizeLinear', [f'{src}_q', 'qscale', 'qzp'],
                                          [f'{src}_dq'], name=f'{src}_dequant'))
        a_in, b_in, opset = 'A_dq', 'B_dq', 19  # fp8 QuantizeLinear lands at opset 19
    nodes.append(helper.make_node('MatMul', [a_in, b_in], ['Y'], name='gemm_core'))
    g = helper.make_graph(nodes, f'gemm_{graph}_{n}', inputs, [Y], initializer=inits)
    model = helper.make_model(g, opset_imports=[helper.make_opsetid('', opset)])
    model.ir_version = 9
    onnx.save(model, path)


def parse_profile(path):
    """trtexec --exportProfile json -> [(name, ms_per_iter)] using medianMs
    when present, else averageMs. Header/no-name entries are skipped."""
    try:
        data = json.load(open(path))
    except Exception:
        return []
    rows = []
    for e in data if isinstance(data, list) else []:
        if not isinstance(e, dict) or 'name' not in e:
            continue
        ms = e.get('medianMs', e.get('averageMs'))
        if isinstance(ms, (int, float)) and ms >= 0:
            rows.append((e['name'], float(ms)))
    return rows


def split_gemm(rows):
    """(gemm_ms, total_ms, census). GEMM = name-matched layers; if none match,
    the single largest layer (the n^3 kernel dominates by construction)."""
    total = sum(ms for _, ms in rows)
    matched = [(n, ms) for n, ms in rows
               if GEMM_NAME_RE.search(n) and not OVERHEAD_NAME_RE.search(n)]
    if not matched and rows:
        matched = [max(rows, key=lambda r: r[1])]
    gemm = sum(ms for _, ms in matched)
    gemm_names = {n for n, _ in matched}
    census = [{'name': n[:70], 'ms': round(ms, 4),
               'pct': round(100 * ms / total, 1) if total else None,
               'gemm': n in gemm_names}
              for n, ms in sorted(rows, key=lambda r: -r[1])[:8]]
    return gemm, total, census


def run_point(trtexec, onnx_path, flags, io_flags, iterations, duration_s,
              warmup_ms, timeout_s, profile_path, idle_ms=0):
    """One build+time+profile invocation. Returns dict or an error string.
    idle_ms > 0 inserts a sleep between iterations (the burst pass): per-query
    GPU Compute Time and per-layer times are unaffected as metrics — only the
    power/clock state each kernel starts from changes."""
    cmd = [
        trtexec, f'--onnx={onnx_path}', *flags, *io_flags,
        f'--builderOptimizationLevel={OPT_LEVEL}',
        '--noDataTransfers', '--useSpinWait',
        f'--iterations={iterations}', f'--duration={duration_s}',
        f'--warmUp={warmup_ms}', '--avgRuns=1',
        '--profilingVerbosity=detailed', '--separateProfileRun',
        f'--exportProfile={profile_path}',
    ]
    if idle_ms > 0:
        cmd.append(f'--idleTime={idle_ms}')
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return f'TIMEOUT after {timeout_s}s'
    out = (p.stdout or '') + (p.stderr or '')
    m = GPU_COMPUTE_RE.search(out)
    if p.returncode != 0 or not m:
        errs = [ln for ln in out.splitlines() if '[E]' in ln or 'error' in ln.lower()]
        return 'rc=%d %s' % (p.returncode,
                             (errs[-1].strip() if errs
                              else (out.strip().splitlines() or ['no output'])[-1][:200]))
    e2e_ms = float(m.group(1))
    rows = parse_profile(profile_path)
    if not rows:
        # No profile means purity cannot be proven — degrade to the e2e median,
        # loudly labeled, rather than crashing away the whole precision.
        return {'e2e_ms': e2e_ms, 'gemm_ms': e2e_ms, 'census': [],
                'gemm_share_pct': None, 'method': 'e2e-no-profile'}
    gemm_ms, total_ms, census = split_gemm(rows)
    share = 100 * gemm_ms / total_ms if total_ms else 0.0
    # Pure engine: the e2e median (no profiler attached, thanks to
    # --separateProfileRun) is the same number measured better; contaminated
    # engine: the GEMM layer's own time.
    if share >= 99.0:
        return {'e2e_ms': e2e_ms, 'gemm_ms': e2e_ms, 'census': census,
                'gemm_share_pct': round(share, 1), 'method': 'pure-engine'}
    return {'e2e_ms': e2e_ms, 'gemm_ms': gemm_ms, 'census': census,
            'gemm_share_pct': round(share, 1), 'method': 'gemm-layer-profile'}


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else 'trt_ceilings'
    os.makedirs(out_dir, exist_ok=True)

    # Re-runs merge: precisions probed now replace their prior entries, the
    # rest carry forward — an fp32-only pass must not wipe the tensor results.
    prior = {}
    _pj = os.path.join(out_dir, 'trt_compute.json')
    if os.path.exists(_pj):
        try:
            prior = json.load(open(_pj)).get('precisions', {})
        except Exception:
            prior = {}

    raw = os.environ.get('TRT_GEMM_SIZES', '2048 4096 8192 12288 16384').replace(',', ' ')
    sizes = [int(x) for x in raw.split() if x.strip()]
    precisions = os.environ.get('TRT_GEMM_PRECISIONS', 'fp16 int8 fp8 fp32').replace(',', ' ').split()
    iterations = int(os.environ.get('TRT_ITERATIONS', '300'))
    duration_s = int(os.environ.get('TRT_DURATION_S', '5'))
    warmup_ms = int(os.environ.get('TRT_WARMUP_MS', '1000'))
    build_timeout = int(os.environ.get('TRT_BUILD_TIMEOUT_S', '1800'))
    burst_idle_ms = int(os.environ.get('TRT_BURST_IDLE_MS', '20'))
    braw = os.environ.get('TRT_BURST_SIZES', '').replace(',', ' ')
    burst_sizes = [int(x) for x in braw.split() if x.strip()]

    trtexec = find_trtexec()
    R = {
        'meta': {
            'instrument': 'trtexec',
            'trt_version': trt_version(trtexec) if trtexec else 'not-found',
            'trtexec_path': trtexec or 'not-found',
            'opt_level': OPT_LEVEL,
            'sizes': sizes,
            'iterations': iterations,
            'duration_s': duration_s,
            'warmup_ms': warmup_ms,
            'burst_idle_ms': burst_idle_ms,
            'flags_policy': {k: ' '.join(v['flags']) for k, v in RECIPES.items()},
            'note': ('attainable = 2*n^3 / GEMM-kernel time (typed-I/O pure engines, '
                     'or the GEMM layer from the per-layer profile); conversion/'
                     'reformat kernels are excluded by design — burst = best over sizes'),
            'start': time.strftime('%F %T'),
        },
        'precisions': {},
    }

    if trtexec is None:
        R['meta']['error'] = 'trtexec not found on PATH or standard install dirs'
        _dump(R, out_dir)
        say('FATAL: trtexec not found — cannot measure TRT compute ceilings')
        sys.exit(1)

    say(f'trtexec {R["meta"]["trt_version"]} at {trtexec}; sizes={sizes} precisions={precisions}')

    with tempfile.TemporaryDirectory(prefix='trt_gemm_') as tmp:
        for prec in precisions:
            recipe = RECIPES.get(prec)
            if recipe is None:
                R['precisions'][prec] = {'error': f'unknown precision "{prec}"'}
                continue
            unit = UNIT[prec]
            pts = []
            point_recipe = {}
            # recipe defaults (fp32 sweeps smaller sizes) — but an explicitly
            # set env always wins, so quick manual probes stay quick
            psizes = sizes if 'TRT_GEMM_SIZES' in os.environ else recipe.get('sizes', sizes)
            piters = iterations if 'TRT_ITERATIONS' in os.environ else recipe.get('iterations', iterations)
            say(f'== {prec} ({" ".join(recipe["flags"])}) ==')
            for n in psizes:
                t0 = time.time()
                prof_path = os.path.join(tmp, f'profile_{prec}_{n}.json')
                # Variant search (graph form x I/O binding): measure every
                # variant that builds; the best GEMM rate is the point's value,
                # every variant's outcome is recorded so the choice stays
                # auditable. ONNX files are cached per graph and deleted after
                # the size finishes (weight-form initializers reach ~0.5 GB).
                onnx_cache = {}
                variants = []
                for graph, io_flags in recipe['variants']:
                    tag = f'{graph} | ' + (' '.join(io_flags) or 'default-io')
                    onnx_path = onnx_cache.get(graph)
                    if onnx_path is None:
                        onnx_path = os.path.join(tmp, f'gemm_{graph}_{n}.onnx')
                        try:
                            make_gemm_onnx(n, onnx_path, graph)
                            onnx_cache[graph] = onnx_path
                        except Exception as ex:
                            onnx_cache[graph] = False
                            variants.append({'io': tag, 'ok': False,
                                             'error': f'onnx build failed: {str(ex)[:100]}'})
                            continue
                    elif onnx_path is False:
                        variants.append({'io': tag, 'ok': False,
                                         'error': 'onnx build failed (see sibling variant)'})
                        continue
                    res = run_point(trtexec, onnx_path, recipe['flags'], io_flags,
                                    piters, duration_s, warmup_ms, build_timeout,
                                    prof_path)
                    if isinstance(res, dict):
                        v = 2.0 * (n ** 3) / (res['gemm_ms'] / 1e3) / 1e12
                        variants.append({'io': tag, 'ok': True,
                                         'val': round(v, 1),
                                         'method': res['method'],
                                         'gemm_share_pct': res['gemm_share_pct'],
                                         'graph': graph, 'io_flags': io_flags,
                                         'res': res})
                    else:
                        variants.append({'io': tag, 'ok': False, 'error': str(res)})
                for p_ in set(v for v in onnx_cache.values() if v):
                    try:
                        os.remove(p_)
                    except OSError:
                        pass
                build_s = round(time.time() - t0, 1)
                good_v = [v for v in variants if v.get('ok')]
                if not good_v:
                    errs = '; '.join(f"{v['io']}: {v['error'][:60]}" for v in variants)
                    say(f'  n={n:6d}  FAILED all variants: {errs}  ({build_s}s)')
                    pts.append({'n': n, 'ok': False, 'error': errs, 'build_s': build_s})
                    continue
                win = max(good_v, key=lambda v: v['val'])
                point_recipe[n] = (win['graph'], win['io_flags'])
                res, val = win['res'], win['val']
                pts.append({'n': n, 'ok': True, unit.lower(): val,
                            'gemm_ms': round(res['gemm_ms'], 4),
                            'e2e_median_ms': round(res['e2e_ms'], 4),
                            'method': res['method'],
                            'gemm_share_pct': res['gemm_share_pct'],
                            'io_formats': win['io'],
                            'io_variants': [{k: v[k] for k in
                                             ('io', 'ok', 'val', 'method', 'gemm_share_pct', 'error')
                                             if k in v} for v in variants],
                            'kernels': res['census'],
                            'build_s': build_s})
                say(f'  n={n:6d}  {val:8.1f} {unit}   gemm={res["gemm_ms"]:.3f} ms '
                    f'({res["method"]}, share={res["gemm_share_pct"]}%, '
                    f'winner={win["io"][:40]}, {len(good_v)}/{len(variants)} variants)   ({build_s}s)')

            good = [p for p in pts if p.get('ok')]
            entry = {'unit': unit, 'flags': ' '.join(recipe['flags']), 'points': pts}
            if good:
                best_pt = max(good, key=lambda p: p[unit.lower()])
                entry['best'] = best_pt[unit.lower()]
                entry['at_n'] = best_pt['n']
                entry['best_method'] = best_pt['method']
            else:
                entry['best'] = None
                entry['error'] = 'all sizes failed — see points[].error and trt_compute_log.txt'

            # ---- burst pass: re-run the winning variant with idle gaps so the
            # power state recovers and the clock boosts before every kernel.
            # Burst runs join the sweep as ordinary points tagged idle_ms, so
            # best-over-all-points is the true peak either way.
            if burst_idle_ms > 0 and entry.get('best') is not None:
                bg, bio = point_recipe[entry['at_n']]
                btag = f'{bg} | ' + (' '.join(bio) or 'default-io') + f' | idle {burst_idle_ms}ms'
                bsizes = burst_sizes or sorted({entry['at_n'], 4096})
                say(f'-- {prec} burst pass (idle {burst_idle_ms} ms, {bg}): sizes {bsizes}')
                for nb in bsizes:
                    t0 = time.time()
                    onnx_path = os.path.join(tmp, f'gemm_burst_{bg}_{nb}.onnx')
                    prof_path = os.path.join(tmp, f'profile_burst_{prec}_{nb}.json')
                    try:
                        make_gemm_onnx(nb, onnx_path, bg)
                    except Exception as ex:
                        say(f'  n={nb:6d}  burst FAILED: onnx {str(ex)[:80]}')
                        continue
                    res = run_point(trtexec, onnx_path, recipe['flags'], bio,
                                    piters, duration_s, warmup_ms, build_timeout,
                                    prof_path, idle_ms=burst_idle_ms)
                    try:
                        os.remove(onnx_path)
                    except OSError:
                        pass
                    build_s = round(time.time() - t0, 1)
                    if not isinstance(res, dict):
                        say(f'  n={nb:6d}  burst FAILED: {res}  ({build_s}s)')
                        pts.append({'n': nb, 'ok': False, 'idle_ms': burst_idle_ms,
                                    'error': str(res), 'build_s': build_s})
                        continue
                    val = round(2.0 * (nb ** 3) / (res['gemm_ms'] / 1e3) / 1e12, 1)
                    pts.append({'n': nb, 'ok': True, unit.lower(): val,
                                'idle_ms': burst_idle_ms,
                                'gemm_ms': round(res['gemm_ms'], 4),
                                'e2e_median_ms': round(res['e2e_ms'], 4),
                                'method': res['method'],
                                'gemm_share_pct': res['gemm_share_pct'],
                                'io_formats': btag,
                                'kernels': res['census'],
                                'build_s': build_s})
                    say(f'  n={nb:6d}  {val:8.1f} {unit}   gemm={res["gemm_ms"]:.3f} ms '
                        f'(burst, {res["method"]})   ({build_s}s)')
                good = [p for p in pts if p.get('ok')]
                best_pt = max(good, key=lambda p: p[unit.lower()])
                entry['best'] = best_pt[unit.lower()]
                entry['at_n'] = best_pt['n']
                entry['best_method'] = best_pt['method']
                entry['best_idle_ms'] = best_pt.get('idle_ms', 0)
            R['precisions'][prec] = entry

    R['meta']['end'] = time.strftime('%F %T')
    kept = [k for k in prior if k not in R['precisions']]
    for k in kept:
        R['precisions'][k] = prior[k]
    if kept:
        R['meta']['merged_prior_precisions'] = kept
    _dump(R, out_dir)
    summ = []
    for prec, e in R['precisions'].items():
        b = e.get('best')
        summ.append(f'{prec}={b:.1f}{e.get("unit", "")}' if isinstance(b, (int, float)) else f'{prec}=n/a')
    say('DONE — ' + '  '.join(summ) + f'  -> {os.path.join(out_dir, "trt_compute.json")}')


def _dump(R, out_dir):
    with open(os.path.join(out_dir, 'trt_compute.json'), 'w') as f:
        json.dump(R, f, indent=1)


if __name__ == '__main__':
    main()
