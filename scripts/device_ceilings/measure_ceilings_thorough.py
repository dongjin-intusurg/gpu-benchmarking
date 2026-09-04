#!/usr/bin/env python3
"""Thorough ceiling characterization for any CUDA device (torch/cuBLASLt paths).

Six suites: tensor GEMM size sweep per precision, transformer-shaped GEMMs,
bandwidth kernels, CPU-load haircut, CUDA-core fp32, sustained-vs-burst.
Usage: measure_ceilings_thorough.py [out_dir]; writes <out_dir>/results.json
(keys: meta, gemm_sweep, shaped, bandwidth, haircut, cuda_fp32, sustained) and
prints one summary line per point. Env: CEIL_GEMM_SIZES, SUSTAIN_PREC (fp16|int8),
SUSTAIN_SECONDS. Assumes clocks are already locked by the caller; no sudo.
"""
import atexit
import glob
import json
import os
import subprocess
import sys
import time

import torch

DEVICE = 'cuda'
DEFAULT_GEMM_SIZES = [1024, 1536, 2048, 3072, 4096, 5120, 6144, 8192, 10240, 12288, 16384]
SHAPED_GEMMS = [(4096, 4096, 11008), (1280, 768, 3072), (8192, 8192, 1024), (512, 4096, 4096)]
BANDWIDTH_BUFFER_MB = (128, 512, 1024, 2048)
HAIRCUT_WORKERS = (0, 2, 4, 8, 12)
REPS = 5
# A rep is CLEAN iff within 10% of the best rep (pre-declared): overcurrent
# clamps on power-constrained modules depress single reps by 15-30% while clean
# reps agree within a few %. Spread is reported over clean reps and over all.
CLEAN_FACTOR = 0.9
# Each timed window is normalized to ~80 ms so short-kernel points are not
# disproportionately clamp-exposed.
WINDOW_SECONDS = 0.08


def gpu_freq_mhz():
    """GPU clock, MHz — Jetson devfreq if present, else nvidia-smi."""
    devfreq = glob.glob('/sys/class/devfreq/*gpu*/cur_freq')
    if devfreq:
        return int(open(devfreq[0]).read()) // 1_000_000
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=clocks.gr', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
        return int(out)
    except Exception:
        return -1


def max_temp_c():
    """Hottest sensor, C — thermal zones if present, else nvidia-smi GPU temp."""
    zones = glob.glob('/sys/devices/virtual/thermal/thermal_zone*/temp')
    if zones:
        return max(int(open(zone).read()) for zone in zones) / 1000.0
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
        return float(out)
    except Exception:
        return -1.0


def bench(fn, work_per_call, iters=20, warmup=5):
    """Throughput in work units per second (FLOP/s or bytes/s) over one timed window."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return work_per_call * iters / (start.elapsed_time(end) / 1e3)


def rep_stats(fn, work_per_call, iters=20):
    """Best/median over REPS window-normalized repeats plus clamp-aware spread."""
    pilot = bench(fn, work_per_call, max(3, iters // 4))
    seconds_per_iter = work_per_call / pilot
    window_iters = max(5, min(400, int(WINDOW_SECONDS / max(seconds_per_iter, 1e-6))))
    values = sorted(bench(fn, work_per_call, window_iters) for _ in range(REPS))
    best = values[-1]
    clean = [value for value in values if value >= CLEAN_FACTOR * best]
    return {'best': best, 'median': values[len(values) // 2],
            'spread_pct': round(100 * (best - values[0]) / best, 1),
            'spread_clean_pct': round(100 * (best - min(clean)) / best, 1),
            'clamped_reps': REPS - len(clean)}


def in_tera_units(stats):
    """Scale only the throughput channels to T-units; spread/clamp stay percent/count."""
    return {key: round(value / 1e12, 1) if key in ('best', 'median') else value
            for key, value in stats.items()}


def gemm_sizes():
    raw = os.environ.get('CEIL_GEMM_SIZES', '').replace(',', ' ').split()
    return [int(x) for x in raw] if raw else DEFAULT_GEMM_SIZES


def sweep_point(n):
    flop = 2.0 * n ** 3
    a16 = torch.rand(n, n, device=DEVICE, dtype=torch.float16)
    b16 = torch.rand(n, n, device=DEVICE, dtype=torch.float16)
    row = {}
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    row['fp16_fp32acc_tflops'] = in_tera_units(rep_stats(lambda: a16 @ b16, flop))
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    row['fp16_fp16acc_tflops'] = in_tera_units(rep_stats(lambda: a16 @ b16, flop))
    # Typed int8 (torch._int_mm) is unavailable or untrusted on some arches
    # (sm_120 pre-cu13x: cuBLAS INVALID_VALUE); degrade to a labelled string
    # rather than aborting the sweep. Consumers treat a non-dict cell as null.
    try:
        a8 = torch.randint(-128, 127, (n, n), device=DEVICE, dtype=torch.int8)
        b8 = torch.randint(-128, 127, (n, n), device=DEVICE, dtype=torch.int8)
        row['int8_tops'] = in_tera_units(rep_stats(lambda: torch._int_mm(a8, b8), flop))
        del a8, b8
    except Exception as ex:
        row['int8_tops'] = f'unavailable: {str(ex)[:80]}'
    del a16, b16
    torch.cuda.empty_cache()
    return row


def suite_gemm_sweep(results):
    print('== suite 1: tensor GEMM sweep ==', flush=True)
    for n in gemm_sizes():
        row = sweep_point(n)
        results['gemm_sweep'][n] = row
        int8 = row['int8_tops']
        int8_text = f'{int8["best"]:7.1f}' if isinstance(int8, dict) else '    n/a'
        print(f'  n={n:6d}  fp16(fp32acc) {row["fp16_fp32acc_tflops"]["best"]:7.1f}  '
              f'fp16(fp16acc) {row["fp16_fp16acc_tflops"]["best"]:7.1f}  '
              f'int8 {int8_text}', flush=True)
    suite_fp8_point(results)


def suite_fp8_point(results):
    try:
        n = 4096
        a8 = torch.rand(n, n, device=DEVICE, dtype=torch.float16).to(torch.float8_e4m3fn)
        b8 = torch.rand(n, n, device=DEVICE, dtype=torch.float16).to(torch.float8_e4m3fn).t().contiguous().t()
        scale_a = torch.tensor(1.0, device=DEVICE)
        scale_b = torch.tensor(1.0, device=DEVICE)

        def scaled_mm():
            return torch._scaled_mm(a8, b8, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float16)

        best = rep_stats(scaled_mm, 2.0 * n ** 3)['best']
        results['gemm_sweep']['fp8_n4096_tflops'] = round(best / 1e12, 1)
        print(f'  fp8 n=4096: {results["gemm_sweep"]["fp8_n4096_tflops"]} TFLOPS', flush=True)
    except Exception as ex:
        results['gemm_sweep']['fp8_n4096_tflops'] = f'unavailable: {str(ex)[:80]}'
        print('  fp8: unavailable', flush=True)


def suite_shaped(results):
    print('== suite 2: transformer-shaped GEMMs ==', flush=True)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    for (m, k, n) in SHAPED_GEMMS:
        a = torch.rand(m, k, device=DEVICE, dtype=torch.float16)
        b = torch.rand(k, n, device=DEVICE, dtype=torch.float16)
        stats = rep_stats(lambda: a @ b, 2.0 * m * k * n)
        key = f'{m}x{k}x{n}'
        results['shaped'][key] = round(stats['best'] / 1e12, 1)
        print(f'  {key}: {results["shaped"][key]} TFLOPS', flush=True)
        del a, b
        torch.cuda.empty_cache()


def bandwidth_kernels(nbytes):
    src = torch.empty(nbytes, dtype=torch.uint8, device=DEVICE)
    dst = torch.empty(nbytes, dtype=torch.uint8, device=DEVICE)
    fa = torch.empty(nbytes // 4, dtype=torch.float32, device=DEVICE).uniform_()
    fb = torch.empty(nbytes // 4, dtype=torch.float32, device=DEVICE).uniform_()
    fc = torch.empty(nbytes // 4, dtype=torch.float32, device=DEVICE)
    gbps = {}
    gbps['copy_RW'] = rep_stats(lambda: dst.copy_(src), 2.0 * nbytes, iters=30)['best'] / 1e9
    gbps['read_only'] = rep_stats(lambda: torch.sum(fa), 1.0 * nbytes, iters=30)['best'] / 1e9
    gbps['write_only'] = rep_stats(lambda: dst.fill_(7), 1.0 * nbytes, iters=30)['best'] / 1e9
    gbps['triad_2R1W'] = rep_stats(lambda: torch.add(fa, fb, alpha=2.0, out=fc), 3.0 * nbytes,
                                   iters=30)['best'] / 1e9
    del src, dst, fa, fb, fc
    torch.cuda.empty_cache()
    return {kernel: round(value, 1) for kernel, value in gbps.items()}


def suite_bandwidth(results):
    print('== suite 3: bandwidth suite ==', flush=True)
    for mb in BANDWIDTH_BUFFER_MB:
        results['bandwidth'][f'{mb}MB'] = bandwidth_kernels(mb * 1024 * 1024)
        print(f'  {mb:5d} MB: {results["bandwidth"][f"{mb}MB"]}', flush=True)


def suite_haircut(results):
    print('== suite 4: CPU-load haircut ==', flush=True)
    have_stress = subprocess.run(['which', 'stress-ng'], capture_output=True).returncode == 0
    nbytes = 1 << 30
    src = torch.empty(nbytes, dtype=torch.uint8, device=DEVICE)
    dst = torch.empty(nbytes, dtype=torch.uint8, device=DEVICE)
    for workers in HAIRCUT_WORKERS:
        key = f'{workers}w'
        if workers and not have_stress:
            results['haircut'][key] = 'stress-ng missing'
            continue
        stress = None
        if workers:
            stress_cmd = ['stress-ng', '--vm', str(workers), '--vm-bytes', '1G', '--timeout', '60']
            stress = subprocess.Popen(stress_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(3)
        gbps = rep_stats(lambda: dst.copy_(src), 2.0 * nbytes, iters=25)['best'] / 1e9
        results['haircut'][key] = round(gbps, 1)
        print(f'  {workers:2d} CPU vm-workers: {results["haircut"][key]} GB/s', flush=True)
        if stress:
            stress.terminate()
            stress.wait()
    del src, dst
    torch.cuda.empty_cache()


def suite_cuda_fp32(results):
    print('== suite 5: CUDA-core fp32 ==', flush=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    n = 8192
    a = torch.rand(n, n, device=DEVICE)
    b = torch.rand(n, n, device=DEVICE)
    results['cuda_fp32'] = round(rep_stats(lambda: a @ b, 2.0 * n ** 3)['best'] / 1e12, 2)
    print(f'  fp32 (no TF32): {results["cuda_fp32"]} TFLOPS', flush=True)
    del a, b
    torch.cuda.empty_cache()


def sustained_step(precision, n):
    if precision == 'int8':
        a = torch.randint(-128, 127, (n, n), device=DEVICE, dtype=torch.int8)
        b = torch.randint(-128, 127, (n, n), device=DEVICE, dtype=torch.int8)
        return lambda: torch._int_mm(a, b)
    a = torch.rand(n, n, device=DEVICE, dtype=torch.float16)
    b = torch.rand(n, n, device=DEVICE, dtype=torch.float16)
    return lambda: (a @ b)


def suite_sustained(results):
    precision = os.environ.get('SUSTAIN_PREC', 'fp16')
    seconds = int(os.environ.get('SUSTAIN_SECONDS', '180'))
    print(f'== suite 6: sustained {precision} GEMM, {seconds}s ==', flush=True)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    n = 4096
    flop = 2.0 * n ** 3
    step = sustained_step(precision, n)
    steps_per_sample = 60
    t_end = time.time() + seconds
    while time.time() < t_end:
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(steps_per_sample):
            step()
        end.record()
        torch.cuda.synchronize()
        tflops = flop * steps_per_sample / (start.elapsed_time(end) / 1e3) / 1e12
        sample = {'t': round(time.time() - (t_end - seconds), 0), 'tflops': round(tflops, 1),
                  'gpu_mhz': gpu_freq_mhz(), 'max_temp_c': max_temp_c()}
        results['sustained'].append(sample)
        print(f'  +{sample["t"]:4.0f}s  {tflops:6.1f} TFLOPS  {sample["gpu_mhz"]} MHz  '
              f'{sample["max_temp_c"]}°C', flush=True)


def dump_partial_on_exit(results, path):
    """A crash in a late suite must not discard the completed ones (a full run
    is ~12 GPU-minutes): whatever was measured is saved, flagged incomplete."""
    if os.path.exists(path):
        return
    results['meta']['incomplete'] = 'run aborted before finishing — partial results'
    with open(path, 'w') as f:
        json.dump(results, f, indent=1)
    print(f'\nABORTED — partial results saved to {path}', flush=True)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else 'thorough_ceilings'
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, 'results.json')
    results = {'meta': {}, 'gemm_sweep': {}, 'shaped': {}, 'bandwidth': {}, 'haircut': {},
               'cuda_fp32': None, 'sustained': []}
    atexit.register(dump_partial_on_exit, results, results_path)

    results['meta']['torch'] = torch.__version__
    results['meta']['gpu_freq_mhz_start'] = gpu_freq_mhz()
    results['meta']['start'] = time.strftime('%F %T')

    suite_gemm_sweep(results)
    suite_shaped(results)
    suite_bandwidth(results)
    suite_haircut(results)
    suite_cuda_fp32(results)
    suite_sustained(results)

    results['meta']['end'] = time.strftime('%F %T')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=1)
    print('\nDONE →', results_path, flush=True)


if __name__ == '__main__':
    main()
