#!/usr/bin/env python3
"""Thorough ceiling characterization for any CUDA device (torch/cuBLASLt paths).

Portable: Jetson (devfreq/thermal-zone sysfs) and discrete GPUs (nvidia-smi).
SUSTAIN_PREC=int8 switches suite 6 to a sustained int8 run.

Suites:
  1. tensor GEMM sweep  — fp16 (fp32-acc & fp16-acc), int8, fp8-if-available;
                          11 sizes x 3 repeats; median + best + spread
  2. shaped GEMMs       — non-square transformer-like (M,K,N) shapes at fp16
  3. bandwidth suite    — copy (R+W), read-only, write-only, triad; size sweep
  4. CPU-load haircut   — copy bandwidth under 0/2/4/8/12 stress-ng vm workers
  5. CUDA-core fp32     — TF32 disabled
  6. sustained vs burst — 3 min continuous fp16 GEMM; throughput/clock/temp
                          sampled every 15 s

Assumes clocks are already locked (verified by caller). Writes results.json +
a printed summary. No sudo required.
"""
import json, os, subprocess, sys, time
import torch

import glob
def gpu_freq_mhz():
    """GPU clock, MHz — Jetson devfreq if present, else nvidia-smi."""
    p = glob.glob('/sys/class/devfreq/*gpu*/cur_freq')
    if p:
        return int(open(p[0]).read()) // 1_000_000
    try:
        out = subprocess.run(['nvidia-smi','--query-gpu=clocks.gr','--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
        return int(out)
    except Exception:
        return -1
def max_temp_c():
    """Hottest sensor, C — thermal zones if present, else nvidia-smi GPU temp."""
    zs = glob.glob('/sys/devices/virtual/thermal/thermal_zone*/temp')
    if zs:
        return max(int(open(z).read()) for z in zs) / 1000.0
    try:
        out = subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
        return float(out)
    except Exception:
        return -1.0


OUT = sys.argv[1] if len(sys.argv) > 1 else 'thorough_ceilings'
os.makedirs(OUT, exist_ok=True)
R = {'meta': {}, 'gemm_sweep': {}, 'shaped': {}, 'bandwidth': {}, 'haircut': {},
     'cuda_fp32': None, 'sustained': []}

# A crash in a late suite must not discard the completed ones (a full run is
# ~12 GPU-minutes): always dump whatever was measured, flagged incomplete.
import atexit
def _dump_on_exit():
    path = os.path.join(OUT, 'results.json')
    if not os.path.exists(path):
        R['meta']['incomplete'] = 'run aborted before finishing — partial results'
        with open(path, 'w') as f:
            json.dump(R, f, indent=1)
        print(f'\nABORTED — partial results saved to {path}', flush=True)
atexit.register(_dump_on_exit)

dev = 'cuda'
R['meta']['torch'] = torch.__version__
R['meta']['gpu_freq_mhz_start'] = gpu_freq_mhz()
R['meta']['start'] = time.strftime('%F %T')

def bench(fn, flop_or_bytes, iters=20, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return flop_or_bytes * iters / (s.elapsed_time(e) / 1e3)

def rep3(fn, unit, iters=20):
    """5 repeats with window normalization and clamp-aware spread.

    Overcurrent clamps on power-constrained modules depress individual reps by
    15-30% while clean reps agree within a few %. Policy (pre-declared, not
    post-hoc): a rep is CLEAN iff within 10% of the best rep. We report
    spread over clean reps (reproducibility), spread over all reps (raw
    honesty), and the clamped count (power-envelope incidence data).
    Window normalization: iterations are scaled so each timed window is
    ~80 ms, so short-kernel points are not disproportionately clamp-exposed."""
    t0 = bench(fn, unit, max(3, iters//4))          # pilot to size the window
    per_iter = unit / t0                             # seconds per iteration
    iters_n = max(5, min(400, int(0.08 / max(per_iter, 1e-6))))
    vals = sorted(bench(fn, unit, iters_n) for _ in range(5))
    best = vals[-1]
    clean = [v for v in vals if v >= 0.9 * best]
    return {'best': best, 'median': vals[len(vals)//2],
            'spread_pct': round(100*(best-vals[0])/best, 1),
            'spread_clean_pct': round(100*(best-min(clean))/best, 1),
            'clamped_reps': 5 - len(clean)}

# ---------- 1. tensor GEMM sweep ----------
sizes = [1024, 1536, 2048, 3072, 4096, 5120, 6144, 8192, 10240, 12288, 16384]
print('== suite 1: tensor GEMM sweep ==', flush=True)
for n in sizes:
    flop = 2.0 * n**3
    a16 = torch.rand(n, n, device=dev, dtype=torch.float16)
    b16 = torch.rand(n, n, device=dev, dtype=torch.float16)
    row = {}
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    # only the throughput keys scale to T-units; the spread/clamp channels
    # stay as-is (percent and count)
    row['fp16_fp32acc_tflops'] = {k: round(v/1e12, 1) if k in ('best', 'median') else v
                                  for k, v in rep3(lambda: a16 @ b16, flop).items()}
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    row['fp16_fp16acc_tflops'] = {k: round(v/1e12, 1) if k in ('best', 'median') else v
                                  for k, v in rep3(lambda: a16 @ b16, flop).items()}
    # typed int8 (torch._int_mm) is unavailable/untrusted on some arches
    # (sm_120 pre-cu13x: cuBLAS INVALID_VALUE) — degrade to a labelled
    # 'unavailable' string rather than aborting the whole sweep, exactly as the
    # fp8 probe below does. Downstream consumers treat a non-dict cell as null.
    try:
        ai = torch.randint(-128, 127, (n, n), device=dev, dtype=torch.int8)
        bi = torch.randint(-128, 127, (n, n), device=dev, dtype=torch.int8)
        row['int8_tops'] = {k: round(v/1e12, 1) if k in ('best', 'median') else v
                            for k, v in rep3(lambda: torch._int_mm(ai, bi), flop).items()}
        del ai, bi
    except Exception as ex:
        row['int8_tops'] = f'unavailable: {str(ex)[:80]}'
    del a16, b16; torch.cuda.empty_cache()
    R['gemm_sweep'][n] = row
    i8 = row['int8_tops']
    i8s = f'{i8["best"]:7.1f}' if isinstance(i8, dict) else '    n/a'
    print(f'  n={n:6d}  fp16(fp32acc) {row["fp16_fp32acc_tflops"]["best"]:7.1f}  '
          f'fp16(fp16acc) {row["fp16_fp16acc_tflops"]["best"]:7.1f}  '
          f'int8 {i8s}', flush=True)

# fp8 attempt (Blackwell): torch._scaled_mm
try:
    n = 4096
    a8 = torch.rand(n, n, device=dev, dtype=torch.float16).to(torch.float8_e4m3fn)
    b8 = torch.rand(n, n, device=dev, dtype=torch.float16).to(torch.float8_e4m3fn).t().contiguous().t()
    sa = torch.tensor(1.0, device=dev); sb = torch.tensor(1.0, device=dev)
    f = lambda: torch._scaled_mm(a8, b8, scale_a=sa, scale_b=sb, out_dtype=torch.float16)
    R['gemm_sweep']['fp8_n4096_tflops'] = round(rep3(f, 2.0*n**3)['best']/1e12, 1)
    print(f'  fp8 n=4096: {R["gemm_sweep"]["fp8_n4096_tflops"]} TFLOPS', flush=True)
except Exception as ex:
    R['gemm_sweep']['fp8_n4096_tflops'] = f'unavailable: {str(ex)[:80]}'
    print('  fp8: unavailable', flush=True)

# ---------- 2. shaped GEMMs (fp16, fp16acc) ----------
print('== suite 2: transformer-shaped GEMMs ==', flush=True)
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
for (M, K, N) in [(4096, 4096, 11008), (1280, 768, 3072), (8192, 8192, 1024), (512, 4096, 4096)]:
    a = torch.rand(M, K, device=dev, dtype=torch.float16)
    b = torch.rand(K, N, device=dev, dtype=torch.float16)
    v = rep3(lambda: a @ b, 2.0*M*K*N)
    R['shaped'][f'{M}x{K}x{N}'] = round(v['best']/1e12, 1)
    print(f'  {M}x{K}x{N}: {R["shaped"][f"{M}x{K}x{N}"]} TFLOPS', flush=True)
    del a, b; torch.cuda.empty_cache()

# ---------- 3. bandwidth suite ----------
print('== suite 3: bandwidth suite ==', flush=True)
def bw_kernels(nbytes):
    n = nbytes
    src = torch.empty(n, dtype=torch.uint8, device=dev)
    dst = torch.empty(n, dtype=torch.uint8, device=dev)
    fa = torch.empty(n//4, dtype=torch.float32, device=dev).uniform_()
    fb = torch.empty(n//4, dtype=torch.float32, device=dev).uniform_()
    fc = torch.empty(n//4, dtype=torch.float32, device=dev)
    out = {}
    out['copy_RW'] = rep3(lambda: dst.copy_(src), 2.0*n, iters=30)['best']/1e9
    out['read_only'] = rep3(lambda: torch.sum(fa), 1.0*n, iters=30)['best']/1e9
    out['write_only'] = rep3(lambda: dst.fill_(7), 1.0*n, iters=30)['best']/1e9
    out['triad_2R1W'] = rep3(lambda: torch.add(fa, fb, alpha=2.0, out=fc), 3.0*n, iters=30)['best']/1e9
    del src, dst, fa, fb, fc; torch.cuda.empty_cache()
    return {k: round(v, 1) for k, v in out.items()}
for mb in (128, 512, 1024, 2048):
    R['bandwidth'][f'{mb}MB'] = bw_kernels(mb*1024*1024)
    print(f'  {mb:5d} MB: {R["bandwidth"][f"{mb}MB"]}', flush=True)

# ---------- 4. CPU-load haircut curve ----------
print('== suite 4: CPU-load haircut ==', flush=True)
have_stress = subprocess.run(['which', 'stress-ng'], capture_output=True).returncode == 0
n = 1 << 30
src = torch.empty(n, dtype=torch.uint8, device=dev); dst = torch.empty(n, dtype=torch.uint8, device=dev)
for workers in (0, 2, 4, 8, 12):
    proc = None
    if workers and have_stress:
        proc = subprocess.Popen(['stress-ng', '--vm', str(workers), '--vm-bytes', '1G',
                                 '--timeout', '60'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
    elif workers and not have_stress:
        R['haircut'][f'{workers}w'] = 'stress-ng missing'; continue
    v = rep3(lambda: dst.copy_(src), 2.0*n, iters=25)['best']/1e9
    R['haircut'][f'{workers}w'] = round(v, 1)
    print(f'  {workers:2d} CPU vm-workers: {R["haircut"][f"{workers}w"]} GB/s', flush=True)
    if proc: proc.terminate(); proc.wait()
del src, dst; torch.cuda.empty_cache()

# ---------- 5. CUDA-core fp32 ----------
print('== suite 5: CUDA-core fp32 ==', flush=True)
torch.backends.cuda.matmul.allow_tf32 = False
n = 8192
a = torch.rand(n, n, device=dev); b = torch.rand(n, n, device=dev)
R['cuda_fp32'] = round(rep3(lambda: a @ b, 2.0*n**3)['best']/1e12, 2)
print(f'  fp32 (no TF32): {R["cuda_fp32"]} TFLOPS', flush=True)
del a, b; torch.cuda.empty_cache()

# ---------- 6. sustained vs burst (3 min) ----------
SUST = os.environ.get('SUSTAIN_PREC', 'fp16')
print(f'== suite 6: sustained {SUST} GEMM, 3 minutes ==', flush=True)
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
n = 4096; flop = 2.0*n**3
if SUST == 'int8':
    a = torch.randint(-128,127,(n,n),device=dev,dtype=torch.int8)
    b = torch.randint(-128,127,(n,n),device=dev,dtype=torch.int8)
    step = lambda: torch._int_mm(a, b)
else:
    a = torch.rand(n, n, device=dev, dtype=torch.float16); b = torch.rand(n, n, device=dev, dtype=torch.float16)
    step = lambda: (a @ b)
t_end = time.time() + 180
while time.time() < t_end:
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(60): step()
    e.record(); torch.cuda.synchronize()
    tf = flop*60/(s.elapsed_time(e)/1e3)/1e12
    R['sustained'].append({'t': round(time.time()-(t_end-180), 0), 'tflops': round(tf, 1),
                           'gpu_mhz': gpu_freq_mhz(), 'max_temp_c': max_temp_c()})
    print(f'  +{R["sustained"][-1]["t"]:4.0f}s  {tf:6.1f} TFLOPS  {R["sustained"][-1]["gpu_mhz"]} MHz  {R["sustained"][-1]["max_temp_c"]}°C', flush=True)

R['meta']['end'] = time.strftime('%F %T')
with open(os.path.join(OUT, 'results.json'), 'w') as f:
    json.dump(R, f, indent=1)
print('\nDONE →', os.path.join(OUT, 'results.json'), flush=True)
