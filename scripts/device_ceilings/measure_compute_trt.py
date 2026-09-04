#!/usr/bin/env python3
"""Attainable per-precision compute ceilings via TensorRT — GEMM kernel ONLY.

Builds square GEMM engines with trtexec at --builderOptimizationLevel=5 (the same
instrument the model half uses), times only the GEMM kernel, and reports
attainable = 2*n^3 / gemm_time with burst = best over the size sweep. Writes
<out>/trt_compute.json (meta, precisions.<prec>.{unit,flags,points,best,at_n,...});
prints '[HH:MM:SS] ...' progress lines ending in '[..] DONE — fp16=...'.

Env overrides:
  TRT_GEMM_SIZES        square sizes, space/comma list (default "2048 4096 8192 12288 16384")
  TRT_GEMM_PRECISIONS   subset of "fp16 int8 fp8 fp32" (default all four; re-runs merge
                        into an existing trt_compute.json)
  TRT_ITERATIONS        trtexec timing iterations       (default 300)
  TRT_DURATION_S        trtexec timing duration seconds (default 5)
  TRT_WARMUP_MS         trtexec warmup milliseconds     (default 1000)
  TRT_BUILD_TIMEOUT_S   per-build wall timeout seconds  (default 1800)
  TRT_BURST_IDLE_MS     idle gap for the burst pass; 0 disables (default 20)
  TRT_BURST_SIZES       sizes for the burst pass (default: winning N + 4096)
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
DEFAULT_SIZES = '2048 4096 8192 12288 16384'
DEFAULT_PRECISIONS = 'fp16 int8 fp8 fp32'
# An engine whose GEMM layer is >= 99% of the profile is "pure": its e2e median
# (no profiler attached) is the same number measured better.
PURE_SHARE_PCT = 99.0
GPU_COMPUTE_RE = re.compile(r'GPU Compute Time:.*?median\s*=\s*([0-9.]+)\s*ms', re.IGNORECASE)

# A compute-ceiling probe must never time casting, quantize or reformat kernels:
# they are memory-bound, so they understate the ceiling exactly on the devices
# where bandwidth is scarce (measured ~2x on a ~254 GB/s unified-memory part).
# Conversion layers are often NAMED AFTER their consumer ("Reformatting CopyNode
# for Input Tensor 0 to gemm_core"), so the exclusion pattern outranks the match.
GEMM_NAME_RE = re.compile(r'gemm_core|matmul|gemm', re.IGNORECASE)
OVERHEAD_NAME_RE = re.compile(r'reformat|copy|cast|quant', re.IGNORECASE)

# Per-precision recipe: trtexec precision flags plus the (graph form, I/O
# binding) variants to search. Every variant that builds is measured and the
# best GEMM rate wins, because an I/O-format restriction or the activation-form
# MatMul can steer TRT to a kernel several times slower than the tuned one and
# a probe cannot know that without measuring the alternative. Graph forms:
# activation x activation ('..'), weight x activation ('.._wb', B as an
# initializer — the shape of every linear/conv), and 1x1-conv ('.._conv', the
# implicit-GEMM path vision layers run on).
IO1_FP16 = ['--inputIOFormats=fp16:chw', '--outputIOFormats=fp16:chw']
IO2_FP16 = ['--inputIOFormats=fp16:chw,fp16:chw', '--outputIOFormats=fp16:chw']
IO2_INT8 = ['--inputIOFormats=int8:chw,int8:chw', '--outputIOFormats=fp16:chw']
RECIPES = {
    'fp16': {
        'flags': ['--fp16'],
        'variants': [('fp16', IO2_FP16),
                     ('fp16_wb', IO1_FP16),
                     ('fp16_wb', []),
                     ('fp16_conv', [])],
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
    # --noTF32 is load-bearing: TensorRT runs fp32 GEMMs on the TF32 tensor pipe
    # by default, which would measure the wrong silicon. Smaller sizes and fewer
    # iterations because fp32 GEMMs run ~50x slower than the tensor precisions.
    'fp32': {
        'flags': ['--noTF32'],
        'variants': [('fp32_wb', []),
                     ('fp32_conv', [])],
        'sizes': [2048, 4096, 8192],
        'iterations': 50,
    },
}

# Quantized scales are arbitrary: they select kernels, not timing.
QDQ = {
    'qdq-int8': (TensorProto.INT8, 0.02),
    'qdq-fp8': (TensorProto.FLOAT8E4M3FN, 0.25),
}


def say(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def find_trtexec():
    exe = shutil.which('trtexec')
    if exe:
        return exe
    for directory in ('/usr/src/tensorrt/bin', '/opt/tensorrt/10.13/bin'):
        candidate = os.path.join(directory, 'trtexec')
        if os.path.exists(candidate):
            return candidate
    return None


def trt_version(trtexec):
    try:
        proc = subprocess.run([trtexec, '--help'], capture_output=True, text=True, timeout=30)
        out = (proc.stdout or '') + (proc.stderr or '')
        match = re.search(r'TensorRT[^0-9]*([0-9]+\.[0-9]+\.[0-9]+)', out)
        if match:
            return match.group(1)
        match = re.search(r'TensorRT\s+v([0-9]{5,7})', out)
        if match:
            packed = int(match.group(1))
            return f'{packed // 10000}.{(packed // 100) % 100}.{packed % 100}'
    except Exception:
        pass
    return '?'


def save_model(nodes, name, inputs, outputs, initializers, opset, path):
    graph = helper.make_graph(nodes, name, inputs, outputs, initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', opset)])
    model.ir_version = 9
    onnx.save(model, path)


def random_weight(shape, dtype):
    """Random values defeat sparsity shortcuts; the seed keeps builds comparable."""
    return numpy_helper.from_array(np.random.default_rng(7).standard_normal(shape).astype(dtype), name='B')


def make_conv_onnx(n, path, graph, tensor_type, np_dtype):
    """1x1 conv with Cin=Cout=n and HxW=n: FLOPs = 2*n^3, identical work routed
    through the implicit-GEMM conv path. Spatial size is the nearest-square
    factorization of n (e.g. 12288 -> 96x128)."""
    height = int(n ** 0.5)
    while n % height:
        height -= 1
    width = n // height
    a_info = helper.make_tensor_value_info('A', tensor_type, [1, n, height, width])
    y_info = helper.make_tensor_value_info('Y', tensor_type, [1, n, height, width])
    nodes = [helper.make_node('Conv', ['A', 'B'], ['Y'], name='gemm_core', kernel_shape=[1, 1])]
    save_model(nodes, f'gemm_{graph}_{n}', [a_info], [y_info],
               [random_weight((n, n, 1, 1), np_dtype)], 17, path)


def make_gemm_onnx(n, path, graph):
    """A[n,n] x B[n,n] -> Y[n,n] with the node named 'gemm_core' so the profile
    row is identifiable. QDQ graphs quantize both operands; the weight-side
    QuantizeLinear folds at build time exactly like a real quantized model."""
    is_fp32 = graph.startswith('fp32')
    tensor_type = TensorProto.FLOAT if is_fp32 else TensorProto.FLOAT16
    np_dtype = np.float32 if is_fp32 else np.float16
    if graph.endswith('_conv'):
        make_conv_onnx(n, path, graph, tensor_type, np_dtype)
        return
    weight_form = graph.endswith('_wb')
    base = graph[:-3] if weight_form else graph
    a_info = helper.make_tensor_value_info('A', tensor_type, [n, n])
    y_info = helper.make_tensor_value_info('Y', tensor_type, [n, n])
    inputs, initializers, nodes = [a_info], [], []
    if weight_form:
        initializers.append(random_weight((n, n), np_dtype))
    else:
        inputs.append(helper.make_tensor_value_info('B', tensor_type, [n, n]))
    if base in ('fp16', 'fp32'):
        a_in, b_in, opset = 'A', 'B', 17
    else:
        zero_point_type, scale = QDQ[base]
        initializers += [helper.make_tensor('qscale', TensorProto.FLOAT16, [], [scale]),
                         helper.make_tensor('qzp', zero_point_type, [], [0.0])]
        for operand in ('A', 'B'):
            nodes.append(helper.make_node('QuantizeLinear', [operand, 'qscale', 'qzp'],
                                          [f'{operand}_q'], name=f'{operand}_quant'))
            nodes.append(helper.make_node('DequantizeLinear', [f'{operand}_q', 'qscale', 'qzp'],
                                          [f'{operand}_dq'], name=f'{operand}_dequant'))
        a_in, b_in, opset = 'A_dq', 'B_dq', 19  # fp8 QuantizeLinear lands at opset 19
    nodes.append(helper.make_node('MatMul', [a_in, b_in], ['Y'], name='gemm_core'))
    save_model(nodes, f'gemm_{graph}_{n}', inputs, [y_info], initializers, opset, path)


def parse_profile(path):
    """trtexec --exportProfile json -> [(name, ms_per_iter)], medianMs preferred
    over averageMs; header/no-name entries are skipped."""
    try:
        data = json.load(open(path))
    except Exception:
        return []
    rows = []
    for entry in data if isinstance(data, list) else []:
        if not isinstance(entry, dict) or 'name' not in entry:
            continue
        ms = entry.get('medianMs', entry.get('averageMs'))
        if isinstance(ms, (int, float)) and ms >= 0:
            rows.append((entry['name'], float(ms)))
    return rows


def split_gemm(rows):
    """(gemm_ms, total_ms, census). GEMM = name-matched layers; if none match
    (some backends rename fused kernels wholesale), the single largest layer —
    the n^3 kernel dominates any conversion pass at the probed sizes."""
    total = sum(ms for _, ms in rows)
    matched = [(name, ms) for name, ms in rows
               if GEMM_NAME_RE.search(name) and not OVERHEAD_NAME_RE.search(name)]
    if not matched and rows:
        matched = [max(rows, key=lambda row: row[1])]
    gemm = sum(ms for _, ms in matched)
    gemm_names = {name for name, _ in matched}
    census = [{'name': name[:70], 'ms': round(ms, 4),
               'pct': round(100 * ms / total, 1) if total else None,
               'gemm': name in gemm_names}
              for name, ms in sorted(rows, key=lambda row: -row[1])[:8]]
    return gemm, total, census


def trtexec_error(proc, out):
    errors = [line for line in out.splitlines() if '[E]' in line or 'error' in line.lower()]
    if errors:
        detail = errors[-1].strip()
    else:
        detail = (out.strip().splitlines() or ['no output'])[-1][:200]
    return 'rc=%d %s' % (proc.returncode, detail)


def run_point(trtexec, onnx_path, flags, io_flags, iterations, duration_s,
              warmup_ms, timeout_s, profile_path, idle_ms=0):
    """One build+time+profile invocation. Returns a result dict or an error
    string. idle_ms > 0 sleeps between iterations (the burst pass): per-query
    and per-layer times are unaffected as metrics, only the power/clock state
    each kernel starts from changes."""
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
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return f'TIMEOUT after {timeout_s}s'
    out = (proc.stdout or '') + (proc.stderr or '')
    match = GPU_COMPUTE_RE.search(out)
    if proc.returncode != 0 or not match:
        return trtexec_error(proc, out)
    e2e_ms = float(match.group(1))
    rows = parse_profile(profile_path)
    if not rows:
        # Without a profile purity cannot be proven: degrade to the e2e median,
        # loudly labeled, rather than losing the whole precision.
        return {'e2e_ms': e2e_ms, 'gemm_ms': e2e_ms, 'census': [],
                'gemm_share_pct': None, 'method': 'e2e-no-profile'}
    gemm_ms, total_ms, census = split_gemm(rows)
    share = 100 * gemm_ms / total_ms if total_ms else 0.0
    if share >= PURE_SHARE_PCT:
        return {'e2e_ms': e2e_ms, 'gemm_ms': e2e_ms, 'census': census,
                'gemm_share_pct': round(share, 1), 'method': 'pure-engine'}
    return {'e2e_ms': e2e_ms, 'gemm_ms': gemm_ms, 'census': census,
            'gemm_share_pct': round(share, 1), 'method': 'gemm-layer-profile'}


def tera_ops(n, gemm_ms):
    return 2.0 * (n ** 3) / (gemm_ms / 1e3) / 1e12


def env_int_list(name, default=''):
    raw = os.environ.get(name, default).replace(',', ' ')
    return [int(x) for x in raw.split() if x.strip()]


def read_settings():
    return {
        'sizes': env_int_list('TRT_GEMM_SIZES', DEFAULT_SIZES),
        'precisions': os.environ.get('TRT_GEMM_PRECISIONS', DEFAULT_PRECISIONS).replace(',', ' ').split(),
        'iterations': int(os.environ.get('TRT_ITERATIONS', '300')),
        'duration_s': int(os.environ.get('TRT_DURATION_S', '5')),
        'warmup_ms': int(os.environ.get('TRT_WARMUP_MS', '1000')),
        'build_timeout': int(os.environ.get('TRT_BUILD_TIMEOUT_S', '1800')),
        'burst_idle_ms': int(os.environ.get('TRT_BURST_IDLE_MS', '20')),
        'burst_sizes': env_int_list('TRT_BURST_SIZES'),
    }


def load_prior_precisions(out_dir):
    """Re-runs merge: precisions probed now replace their prior entries, the rest
    carry forward — an fp32-only pass must not wipe the tensor results."""
    path = os.path.join(out_dir, 'trt_compute.json')
    if not os.path.exists(path):
        return {}
    try:
        return json.load(open(path)).get('precisions', {})
    except Exception:
        return {}


def initial_results(trtexec, settings):
    return {
        'meta': {
            'instrument': 'trtexec',
            'trt_version': trt_version(trtexec) if trtexec else 'not-found',
            'trtexec_path': trtexec or 'not-found',
            'opt_level': OPT_LEVEL,
            'sizes': settings['sizes'],
            'iterations': settings['iterations'],
            'duration_s': settings['duration_s'],
            'warmup_ms': settings['warmup_ms'],
            'burst_idle_ms': settings['burst_idle_ms'],
            'flags_policy': {prec: ' '.join(recipe['flags']) for prec, recipe in RECIPES.items()},
            'note': ('attainable = 2*n^3 / GEMM-kernel time (typed-I/O pure engines, '
                     'or the GEMM layer from the per-layer profile); conversion/'
                     'reformat kernels are excluded by design — burst = best over sizes'),
            'start': time.strftime('%F %T'),
        },
        'precisions': {},
    }


class PrecisionProbe:
    """Sweeps one precision's recipe over sizes, then the burst pass."""

    def __init__(self, trtexec, tmp_dir, settings, precision):
        self.trtexec = trtexec
        self.tmp_dir = tmp_dir
        self.settings = settings
        self.precision = precision
        self.recipe = RECIPES[precision]
        self.unit = UNIT[precision]
        self.value_key = self.unit.lower()
        # Recipe defaults (fp32 sweeps smaller sizes), but an explicitly set env
        # always wins so quick manual probes stay quick.
        self.sizes = (settings['sizes'] if 'TRT_GEMM_SIZES' in os.environ
                      else self.recipe.get('sizes', settings['sizes']))
        self.iterations = (settings['iterations'] if 'TRT_ITERATIONS' in os.environ
                           else self.recipe.get('iterations', settings['iterations']))
        self.points = []
        self.winning_variant = {}

    def time_point(self, onnx_path, io_flags, profile_path, idle_ms=0):
        return run_point(self.trtexec, onnx_path, self.recipe['flags'], io_flags,
                         self.iterations, self.settings['duration_s'], self.settings['warmup_ms'],
                         self.settings['build_timeout'], profile_path, idle_ms=idle_ms)

    def measure_variants(self, n, profile_path):
        """Measure every (graph, I/O) variant that builds. ONNX files are cached
        per graph and deleted after the size finishes (weight-form
        initializers reach ~0.5 GB)."""
        onnx_cache = {}
        variants = []
        for graph, io_flags in self.recipe['variants']:
            tag = f'{graph} | ' + (' '.join(io_flags) or 'default-io')
            onnx_path = onnx_cache.get(graph)
            if onnx_path is None:
                onnx_path = os.path.join(self.tmp_dir, f'gemm_{graph}_{n}.onnx')
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
            result = self.time_point(onnx_path, io_flags, profile_path)
            if isinstance(result, dict):
                variants.append({'io': tag, 'ok': True,
                                 'val': round(tera_ops(n, result['gemm_ms']), 1),
                                 'method': result['method'],
                                 'gemm_share_pct': result['gemm_share_pct'],
                                 'graph': graph, 'io_flags': io_flags,
                                 'res': result})
            else:
                variants.append({'io': tag, 'ok': False, 'error': str(result)})
        for path in set(path for path in onnx_cache.values() if path):
            try:
                os.remove(path)
            except OSError:
                pass
        return variants

    def sweep_point(self, n):
        t0 = time.time()
        profile_path = os.path.join(self.tmp_dir, f'profile_{self.precision}_{n}.json')
        variants = self.measure_variants(n, profile_path)
        build_s = round(time.time() - t0, 1)
        good = [variant for variant in variants if variant.get('ok')]
        if not good:
            errors = '; '.join(f"{variant['io']}: {variant['error'][:60]}" for variant in variants)
            say(f'  n={n:6d}  FAILED all variants: {errors}  ({build_s}s)')
            self.points.append({'n': n, 'ok': False, 'error': errors, 'build_s': build_s})
            return
        winner = max(good, key=lambda variant: variant['val'])
        self.winning_variant[n] = (winner['graph'], winner['io_flags'])
        result, value = winner['res'], winner['val']
        self.points.append({'n': n, 'ok': True, self.value_key: value,
                            'gemm_ms': round(result['gemm_ms'], 4),
                            'e2e_median_ms': round(result['e2e_ms'], 4),
                            'method': result['method'],
                            'gemm_share_pct': result['gemm_share_pct'],
                            'io_formats': winner['io'],
                            'io_variants': [{key: variant[key] for key in
                                             ('io', 'ok', 'val', 'method', 'gemm_share_pct', 'error')
                                             if key in variant} for variant in variants],
                            'kernels': result['census'],
                            'build_s': build_s})
        say(f'  n={n:6d}  {value:8.1f} {self.unit}   gemm={result["gemm_ms"]:.3f} ms '
            f'({result["method"]}, share={result["gemm_share_pct"]}%, '
            f'winner={winner["io"][:40]}, {len(good)}/{len(variants)} variants)   ({build_s}s)')

    def burst_point(self, n, graph, io_flags, idle_ms, tag):
        t0 = time.time()
        onnx_path = os.path.join(self.tmp_dir, f'gemm_burst_{graph}_{n}.onnx')
        profile_path = os.path.join(self.tmp_dir, f'profile_burst_{self.precision}_{n}.json')
        try:
            make_gemm_onnx(n, onnx_path, graph)
        except Exception as ex:
            say(f'  n={n:6d}  burst FAILED: onnx {str(ex)[:80]}')
            return
        result = self.time_point(onnx_path, io_flags, profile_path, idle_ms=idle_ms)
        try:
            os.remove(onnx_path)
        except OSError:
            pass
        build_s = round(time.time() - t0, 1)
        if not isinstance(result, dict):
            say(f'  n={n:6d}  burst FAILED: {result}  ({build_s}s)')
            self.points.append({'n': n, 'ok': False, 'idle_ms': idle_ms,
                                'error': str(result), 'build_s': build_s})
            return
        value = round(tera_ops(n, result['gemm_ms']), 1)
        self.points.append({'n': n, 'ok': True, self.value_key: value,
                            'idle_ms': idle_ms,
                            'gemm_ms': round(result['gemm_ms'], 4),
                            'e2e_median_ms': round(result['e2e_ms'], 4),
                            'method': result['method'],
                            'gemm_share_pct': result['gemm_share_pct'],
                            'io_formats': tag,
                            'kernels': result['census'],
                            'build_s': build_s})
        say(f'  n={n:6d}  {value:8.1f} {self.unit}   gemm={result["gemm_ms"]:.3f} ms '
            f'(burst, {result["method"]})   ({build_s}s)')

    def best_point(self):
        good = [point for point in self.points if point.get('ok')]
        return max(good, key=lambda point: point[self.value_key]) if good else None

    def burst_pass(self, entry):
        """Re-run the winning variant with idle gaps so the power state recovers
        and the clock boosts before every kernel: a governor that caps any
        continuous tensor load (the discrete 300 W card settles ~7% below its
        short-load clock within milliseconds) can never show its burst in a
        gapless sweep. Burst runs join the sweep as ordinary points tagged
        idle_ms, so best-over-all-points is the true peak either way."""
        idle_ms = self.settings['burst_idle_ms']
        graph, io_flags = self.winning_variant[entry['at_n']]
        tag = f'{graph} | ' + (' '.join(io_flags) or 'default-io') + f' | idle {idle_ms}ms'
        sizes = self.settings['burst_sizes'] or sorted({entry['at_n'], 4096})
        say(f'-- {self.precision} burst pass (idle {idle_ms} ms, {graph}): sizes {sizes}')
        for n in sizes:
            self.burst_point(n, graph, io_flags, idle_ms, tag)
        best = self.best_point()
        entry['best'] = best[self.value_key]
        entry['at_n'] = best['n']
        entry['best_method'] = best['method']
        entry['best_idle_ms'] = best.get('idle_ms', 0)

    def run(self):
        say(f'== {self.precision} ({" ".join(self.recipe["flags"])}) ==')
        for n in self.sizes:
            self.sweep_point(n)
        entry = {'unit': self.unit, 'flags': ' '.join(self.recipe['flags']), 'points': self.points}
        best = self.best_point()
        if best is None:
            entry['best'] = None
            entry['error'] = 'all sizes failed — see points[].error and trt_compute_log.txt'
            return entry
        entry['best'] = best[self.value_key]
        entry['at_n'] = best['n']
        entry['best_method'] = best['method']
        if self.settings['burst_idle_ms'] > 0:
            self.burst_pass(entry)
        return entry


def summary_line(results, out_dir):
    parts = []
    for precision, entry in results['precisions'].items():
        best = entry.get('best')
        if isinstance(best, (int, float)):
            parts.append(f'{precision}={best:.1f}{entry.get("unit", "")}')
        else:
            parts.append(f'{precision}=n/a')
    return 'DONE — ' + '  '.join(parts) + f'  -> {os.path.join(out_dir, "trt_compute.json")}'


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else 'trt_ceilings'
    os.makedirs(out_dir, exist_ok=True)
    prior = load_prior_precisions(out_dir)
    settings = read_settings()
    trtexec = find_trtexec()
    results = initial_results(trtexec, settings)

    if trtexec is None:
        results['meta']['error'] = 'trtexec not found on PATH or standard install dirs'
        _dump(results, out_dir)
        say('FATAL: trtexec not found — cannot measure TRT compute ceilings')
        sys.exit(1)

    say(f'trtexec {results["meta"]["trt_version"]} at {trtexec}; '
        f'sizes={settings["sizes"]} precisions={settings["precisions"]}')

    with tempfile.TemporaryDirectory(prefix='trt_gemm_') as tmp_dir:
        for precision in settings['precisions']:
            if precision not in RECIPES:
                results['precisions'][precision] = {'error': f'unknown precision "{precision}"'}
                continue
            results['precisions'][precision] = PrecisionProbe(trtexec, tmp_dir, settings, precision).run()

    results['meta']['end'] = time.strftime('%F %T')
    kept = [precision for precision in prior if precision not in results['precisions']]
    for precision in kept:
        results['precisions'][precision] = prior[precision]
    if kept:
        results['meta']['merged_prior_precisions'] = kept
    _dump(results, out_dir)
    say(summary_line(results, out_dir))


def _dump(results, out_dir):
    with open(os.path.join(out_dir, 'trt_compute.json'), 'w') as f:
        json.dump(results, f, indent=1)


if __name__ == '__main__':
    main()
