#!/usr/bin/env python3
"""Turn a completed ceilings run into a PASS / INVESTIGATE verdict.

Four gates, in order of authority:
  1. regime integrity  - preflight PASS + not smoke_only, lock verified, drift PASS
                         (a drift FAIL or smoke_only voids everything below it)
  2. self-sanity       - every ceiling as a fraction of datasheet is inside the
                         device config's band (works with no reference at all)
  3. reproducibility   - if a reference run is given, each ceiling is within
                         tolerance of it. This is the gate that catches a moved
                         peak (a lucky one-off tactic in either run): a GEMM
                         ceiling legitimately peaks at a mid size, so "high vs
                         neighbours" is the normal shape - only a shift between
                         runs is the signal.

Gates 1-2 need only this run. Gate 3 is skipped when no --reference is passed.
Exit 0 = PASS, 2 = INVESTIGATE, 1 = could not evaluate.

Usage:
  validate_ceilings.py <run_dir> [--reference <ref_dir>] [--tol-pct 5]
                       [--bw-tol-pct 1] ["""
import argparse, json, os, sys

def load(p):
    try: return json.load(open(p))
    except Exception: return None

def frac_band(cfg, key):
    b = (cfg.get('sanity_bands') or {}).get(key)
    if isinstance(b, (list, tuple)) and len(b) == 2: return float(b[0]), float(b[1])
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--reference')
    ap.add_argument('--device')
    ap.add_argument('--tol-pct', type=float, default=5.0)
    ap.add_argument('--bw-tol-pct', type=float, default=1.0)
    a = ap.parse_args()
    R = a.run_dir
    fails, warns, lines = [], [], []

    def raw(d, *names):
        for n in names:
            p = os.path.join(d, 'raw', n)
            if os.path.exists(p): return load(p)
        return None

    trt = raw(R, 'trt_compute.json')
    res = raw(R, 'results.json')
    prov = os.path.join(R, 'provenance')
    dev = load(a.device) if a.device else None

    # ---- gate 1: regime integrity ----
    pf = load(os.path.join(prov, 'preflight.json'))
    lv = load(os.path.join(prov, 'lock_verified.json'))
    dr = load(os.path.join(R, 'clock_drift.json'))
    if not pf or pf.get('verdict') != 'PASS': fails.append('preflight not PASS')
    if pf and pf.get('smoke_only'): fails.append('smoke_only:true - results are not certified')
    if not lv or lv.get('verdict') != 'PASS': fails.append('lock not verified PASS')
    if not dr or dr.get('verdict') == 'FAIL': fails.append('clock drift FAIL')
    elif dr.get('verdict') == 'WARN': warns.append(f"drift WARN (pct_at_target {dr.get('pct_at_target')})")
    lines.append(f"[1] regime    preflight={pf and pf.get('verdict')} "
                 f"lock={lv and lv.get('verdict')} drift={dr and dr.get('verdict')} "
                 f"pct_at_target={dr and dr.get('pct_at_target')}")

    # ---- gate 2: self-sanity bands ----
    if trt and dev:
        ds = dev.get('datasheet', {})
        peaks = {'fp16': ds.get('peak_fp16_dense_tflops'), 'int8': ds.get('peak_int8_dense_tops'),
                 'fp8': ds.get('peak_fp8_dense_tflops'), 'fp32': ds.get('peak_fp32_cuda_tflops')}
        band = frac_band(dev, 'gemm_frac_of_datasheet') or (0.45, 0.95)
        for p, pk in peaks.items():
            best = (trt['precisions'].get(p) or {}).get('best')
            if best and pk:
                fr = best / pk
                ok = band[0] <= fr <= band[1]
                (lines if ok else fails).append(
                    f"[2] sanity    {p:5} {best:>7}/{pk} = {fr*100:.0f}%  band {band[0]*100:.0f}-{band[1]*100:.0f}%  {'PASS' if ok else 'FAIL'}")
    else:
        warns.append('gate 2 skipped: need --device and trt_compute.json')

    # ---- gate 3: reproducibility ----
    if a.reference:
        ref = raw(a.reference, 'trt_compute.json')
        refres = raw(a.reference, 'results.json')
        if trt and ref:
            for p in ('fp16', 'int8', 'fp8', 'fp32'):
                n = (trt['precisions'].get(p) or {}).get('best')
                r = (ref['precisions'].get(p) or {}).get('best')
                if n and r:
                    d = (n - r) / r * 100
                    ok = abs(d) <= a.tol_pct
                    (lines if ok else warns).append(
                        f"[3] reprod    {p:5} {n:>7} vs {r:<7} {d:+.1f}%  tol {a.tol_pct}%  {'PASS' if ok else 'INVESTIGATE'}")
        if res and refres:
            def copy_best(x):
                bw = x.get('bandwidth', {})
                return max((v.get('copy_RW', 0) for v in bw.values()), default=0)
            n, r = copy_best(res), copy_best(refres)
            if n and r:
                d = (n - r) / r * 100
                ok = abs(d) <= a.bw_tol_pct
                (lines if ok else fails).append(
                    f"[3] reprod    copy  {n:>7} vs {r:<7} {d:+.1f}%  tol {a.bw_tol_pct}%  {'PASS' if ok else 'FAIL'}  (budget basis)")
    else:
        lines.append('[3] reprod    skipped (no --reference)')

    print('\n'.join(lines))
    for w in warns: print('  WARN  ' + w)
    for f in fails: print('  FAIL  ' + f)
    if fails: print('\nVERDICT: INVESTIGATE'); return 2
    if warns: print('\nVERDICT: PASS (with warnings)'); return 0
    print('\nVERDICT: PASS'); return 0

if __name__ == '__main__':
    sys.exit(main())
