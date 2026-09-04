#!/usr/bin/env python3
"""Turn a completed ceilings run into a PASS / INVESTIGATE verdict.

Three gates, in order of authority: regime integrity (preflight, lock, drift),
self-sanity (each ceiling inside the device config's datasheet band), and
reproducibility against an optional --reference run. Gate 3 is the one that
catches a moved peak: a GEMM ceiling legitimately peaks at a mid size, so only
a shift between runs is the signal. Prints one line per check, then WARN/FAIL
lines, then 'VERDICT: ...'. Exit 0 = PASS, 2 = INVESTIGATE, 1 = could not evaluate.

Usage:
  validate_ceilings.py <run_dir> [--device <cfg.json>] [--reference <ref_dir>]
                       [--tol-pct 5] [--bw-tol-pct 1]
"""
import argparse
import json
import os
import sys

GEMM_BAND_DEFAULT = (0.45, 0.95)
DATASHEET_KEYS = {
    'fp16': 'peak_fp16_dense_tflops',
    'int8': 'peak_int8_dense_tops',
    'fp8': 'peak_fp8_dense_tflops',
    'fp32': 'peak_fp32_cuda_tflops',
}


def load_json(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def load_raw(run_dir, name):
    path = os.path.join(run_dir, 'raw', name)
    return load_json(path) if os.path.exists(path) else None


def frac_band(device_config, key):
    band = (device_config.get('sanity_bands') or {}).get(key)
    if isinstance(band, (list, tuple)) and len(band) == 2:
        return float(band[0]), float(band[1])
    return None


def trt_best(trt, precision):
    return (trt['precisions'].get(precision) or {}).get('best')


def best_copy_bandwidth(results):
    bandwidth = results.get('bandwidth', {})
    return max((row.get('copy_RW', 0) for row in bandwidth.values()), default=0)


def gate_regime(run_dir, fails, warns, lines):
    provenance = os.path.join(run_dir, 'provenance')
    preflight = load_json(os.path.join(provenance, 'preflight.json'))
    lock = load_json(os.path.join(provenance, 'lock_verified.json'))
    drift = load_json(os.path.join(run_dir, 'clock_drift.json'))
    if not preflight or preflight.get('verdict') != 'PASS':
        fails.append('preflight not PASS')
    if preflight and preflight.get('smoke_only'):
        fails.append('smoke_only:true - results are not certified')
    if not lock or lock.get('verdict') != 'PASS':
        fails.append('lock not verified PASS')
    if not drift or drift.get('verdict') == 'FAIL':
        fails.append('clock drift FAIL')
    elif drift.get('verdict') == 'WARN':
        warns.append(f"drift WARN (pct_at_target {drift.get('pct_at_target')})")
    lines.append(f"[1] regime    preflight={preflight and preflight.get('verdict')} "
                 f"lock={lock and lock.get('verdict')} drift={drift and drift.get('verdict')} "
                 f"pct_at_target={drift and drift.get('pct_at_target')}")


def gate_sanity(trt, device_config, fails, warns, lines):
    if not (trt and device_config):
        warns.append('gate 2 skipped: need --device and trt_compute.json')
        return
    datasheet = device_config.get('datasheet', {})
    band_low, band_high = frac_band(device_config, 'gemm_frac_of_datasheet') or GEMM_BAND_DEFAULT
    for precision, datasheet_key in DATASHEET_KEYS.items():
        best = trt_best(trt, precision)
        peak = datasheet.get(datasheet_key)
        if not (best and peak):
            continue
        fraction = best / peak
        ok = band_low <= fraction <= band_high
        (lines if ok else fails).append(
            f"[2] sanity    {precision:5} {best:>7}/{peak} = {fraction * 100:.0f}%  "
            f"band {band_low * 100:.0f}-{band_high * 100:.0f}%  {'PASS' if ok else 'FAIL'}")


def gate_reproducibility(args, trt, results, fails, warns, lines):
    if not args.reference:
        lines.append('[3] reprod    skipped (no --reference)')
        return
    reference_trt = load_raw(args.reference, 'trt_compute.json')
    reference_results = load_raw(args.reference, 'results.json')
    if trt and reference_trt:
        for precision in DATASHEET_KEYS:
            current = trt_best(trt, precision)
            reference = trt_best(reference_trt, precision)
            if not (current and reference):
                continue
            delta_pct = (current - reference) / reference * 100
            ok = abs(delta_pct) <= args.tol_pct
            (lines if ok else warns).append(
                f"[3] reprod    {precision:5} {current:>7} vs {reference:<7} {delta_pct:+.1f}%  "
                f"tol {args.tol_pct}%  {'PASS' if ok else 'INVESTIGATE'}")
    if results and reference_results:
        current = best_copy_bandwidth(results)
        reference = best_copy_bandwidth(reference_results)
        if current and reference:
            delta_pct = (current - reference) / reference * 100
            ok = abs(delta_pct) <= args.bw_tol_pct
            (lines if ok else fails).append(
                f"[3] reprod    copy  {current:>7} vs {reference:<7} {delta_pct:+.1f}%  "
                f"tol {args.bw_tol_pct}%  {'PASS' if ok else 'FAIL'}  (budget basis)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir')
    parser.add_argument('--reference')
    parser.add_argument('--device')
    parser.add_argument('--tol-pct', type=float, default=5.0)
    parser.add_argument('--bw-tol-pct', type=float, default=1.0)
    args = parser.parse_args()

    fails, warns, lines = [], [], []
    trt = load_raw(args.run_dir, 'trt_compute.json')
    results = load_raw(args.run_dir, 'results.json')
    device_config = load_json(args.device) if args.device else None

    gate_regime(args.run_dir, fails, warns, lines)
    gate_sanity(trt, device_config, fails, warns, lines)
    gate_reproducibility(args, trt, results, fails, warns, lines)

    print('\n'.join(lines))
    for warn in warns:
        print('  WARN  ' + warn)
    for fail in fails:
        print('  FAIL  ' + fail)
    if fails:
        print('\nVERDICT: INVESTIGATE')
        return 2
    if warns:
        print('\nVERDICT: PASS (with warnings)')
        return 0
    print('\nVERDICT: PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
