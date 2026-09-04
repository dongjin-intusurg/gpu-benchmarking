#!/usr/bin/env python3
"""Demand budgets -> U_max, C, L, N, Score for measured rows, from any latency source.

    time_share = latency_ms/1e3 * hz          U_bw = (bytes_MB/1e3 * hz) / bw_eff_gbps
    U_vram     = vram_mb / vram_capacity_mb   U_max = max(budgets > 0);  C = 1/U_max
    L = deadline_ms / latency_ms;  N = min(L, C);  Score = N * arch_gflops * hz / 1e3 (TFLOP/s)

Input: --rows <json list or dict> (or - for stdin), or one row from --latency-ms/--hz/--deadline-ms.
Output: {'schema': 'budgets/v1', bw_eff_gbps, bw_ceiling_source, vram_capacity_mb, vram_capacity_source,
rows: [row + 'solo' | 'error']} to --out and/or stdout (--json-only); otherwise a summary table.
Exit 0 whenever the inputs parse; a row that cannot be scored carries 'error' instead of 'solo'.
"""
import argparse
import json
import os
import subprocess
import sys

# A missing bytes/frame is recorded as an incomplete budget, never as zero: zero would make the
# row look free on the axis that decides most generative models.
INCOMPLETE_CAVEAT = ' not measured, so the true binding budget may be higher and N lower'


def bw_eff_from_ceilings(path):
    """Bandwidth denominator: the best idle copy_RW over the ceilings buffer sweep (the
    solo-exclusive regime budgets against idle bandwidth, not a contended figure)."""
    if not path or not os.path.exists(path):
        return None, 'ceilings not provided'
    try:
        ceilings = json.load(open(path))
    except Exception as exc:
        return None, f'ceilings unreadable: {exc}'
    best = None
    for row in (ceilings.get('bandwidth') or {}).values():
        value = row.get('copy_RW') if isinstance(row, dict) else None
        if isinstance(value, (int, float)) and (best is None or value > best):
            best = value
    return best, ('thorough-ceilings idle copy_RW best' if best else 'no copy_RW in ceilings')


def _meminfo_total_mb():
    for line in open('/proc/meminfo'):
        if line.startswith('MemTotal:'):
            return float(line.split()[1]) / 1024.0
    return None


def vram_capacity_mb(device_cfg, override=None):
    """VRAM denominator: explicit override > device-config cap > unified MemTotal (Jetson) > nvidia-smi."""
    if override:
        return float(override), 'env-override'
    if device_cfg and os.path.exists(device_cfg):
        try:
            config = json.load(open(device_cfg))
        except Exception:
            config = {}
        cap = config.get('vram_budget_cap_mb')
        if cap:
            return float(cap), 'device-config-cap'
        if (config.get('platform') or '') == 'jetson':
            try:
                total = _meminfo_total_mb()
                if total is not None:
                    return total, 'meminfo-unified'
            except Exception:
                pass
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        if out:
            return float(out.splitlines()[0]), 'nvidia-smi-total'
    except Exception:
        pass
    return None, 'unavailable'


def _budgets_at(hz, latency_ms, bytes_mb, bw_eff, vram_mb, vram_cap):
    """The three budget gauges at one rate; a gauge is absent when its measurement is."""
    budgets = {'time_occupancy': latency_ms / 1e3 * hz}
    if bytes_mb is not None and bw_eff:
        budgets['dram_bandwidth'] = bytes_mb / 1e3 * hz / bw_eff
    if vram_mb and vram_cap:
        budgets['vram_footprint'] = float(vram_mb) / vram_cap
    return budgets


def _rate_sweep(solo, grid, latency_ms, bytes_mb, bw_eff, vram_mb, vram_cap):
    """N at each candidate rate with the period as the deadline, plus the highest rate at N >= 1.
    Time and bandwidth scale with the rate, VRAM does not; the declared hz governs only the headline N."""
    n_vs_hz = {}
    for rate in grid:
        u_max = max(_budgets_at(rate, latency_ms, bytes_mb, bw_eff, vram_mb, vram_cap).values())
        n_vs_hz[str(rate)] = round(min((1e3 / rate) / latency_ms, 1.0 / u_max), 3) if u_max > 0 else None
    solo['N_vs_hz'] = n_vs_hz
    solo['N_vs_hz_deadline'] = 'period (1000/hz ms)'
    if vram_mb and vram_cap and float(vram_mb) / vram_cap >= 1:
        solo['max_hz_at_N1'], solo['max_hz_bound_by'] = 0.0, 'vram'
        return
    candidates = {'time': 1e3 / latency_ms}
    if bytes_mb and bw_eff:      # 0 bytes (unknown) cannot bound the rate; None and 0 alike
        candidates['dram_bandwidth'] = bw_eff * 1e3 / bytes_mb
    binding = min(candidates, key=candidates.get)
    solo['max_hz_at_N1'], solo['max_hz_bound_by'] = round(candidates[binding], 2), binding


def score_row(row, bw_eff, vram_cap):
    """The solo verdict for one row: returns the row with 'solo' (or 'error') added."""
    latency_ms = row.get('latency_ms')
    hz = row.get('hz')
    deadline_ms = row.get('deadline_ms')
    if not latency_ms or latency_ms <= 0:
        row['error'] = 'no latency of record — nothing to score'
        return row
    if hz is None or deadline_ms is None:
        row['error'] = 'hz and deadline_ms are required (they are the mix definition, not measurements)'
        return row
    hz, deadline_ms, latency_ms = float(hz), float(deadline_ms), float(latency_ms)
    arch_gflops = float(row.get('arch_gflops') or 0.0)
    bytes_mb = row.get('bytes_per_frame_MB')
    vram_mb = row.get('vram_mb')

    budgets = _budgets_at(hz, latency_ms, bytes_mb, bw_eff, vram_mb, vram_cap)
    not_measured = []
    if 'dram_bandwidth' in budgets:
        row['bw_demand_gbps'] = bytes_mb / 1e3 * hz
    else:
        not_measured.append('dram_bandwidth' if bw_eff else 'dram_bandwidth (no bw_eff)')
        row.setdefault('bytes_source', 'none')
    if 'vram_footprint' not in budgets:
        not_measured.append('vram_footprint')

    if not (any(value > 0 for value in budgets.values()) or hz == 0):
        row['error'] = 'every budget is zero — nothing to divide'
        return row

    u_max = max((value for value in budgets.values() if value > 0), default=0.0)
    capacity = (1.0 / u_max) if u_max > 0 else float('inf')
    latency_margin = deadline_ms / latency_ms
    n = min(latency_margin, capacity)
    workload_gflops_per_s = arch_gflops * hz / 1e3
    solo = {
        'budgets': {key: round(value, 4) for key, value in budgets.items()},
        'U_max': round(u_max, 4),
        'C': round(capacity, 3) if capacity != float('inf') else None,
        'L': round(latency_margin, 3),
        'N': round(n, 3),
        'cause': 'latency-limited' if latency_margin < capacity else 'throughput-limited',
        'score_tflops': round(n * workload_gflops_per_s, 3),
        'budget_complete': not not_measured,
    }
    if not_measured:
        solo['budgets_not_measured'] = not_measured
        solo['caveat'] = 'U_max is a LOWER bound: ' + ', '.join(not_measured) + INCOMPLETE_CAVEAT
    if row.get('hz_grid'):
        _rate_sweep(solo, row['hz_grid'], latency_ms, bytes_mb, bw_eff, vram_mb, vram_cap)
    if hz == 0:
        solo['score_tflops'] = None
        solo['modal'] = ('hz=0: no sustained time/bandwidth bill; N is the verdict '
                         '(verify L against the CONTENDED latency)')
    row['solo'] = solo
    return row


def load_rows(args, parser):
    if args.rows:
        raw = sys.stdin.read() if args.rows == '-' else open(args.rows).read()
        rows = json.loads(raw)
        if isinstance(rows, dict):
            rows = [dict(value, name=key) for key, value in rows.items()]
        return rows
    if args.latency_ms is None:
        parser.error('give --rows, or --latency-ms with --hz and --deadline-ms')
    return [{'name': args.name or 'row', 'latency_ms': args.latency_ms, 'hz': args.hz,
             'deadline_ms': args.deadline_ms, 'arch_gflops': args.arch_gflops,
             'bytes_per_frame_MB': args.bytes_mb, 'vram_mb': args.vram_mb,
             'latency_source': args.latency_source}]


def print_table(scored):
    print(f"  {'row':26} {'latency':>9} {'U_max':>7} {'C':>7} {'L':>7} {'N':>7}  cause")
    for row in scored:
        solo = row.get('solo')
        if not solo:
            print(f"  {row.get('name', '?'):26} {row.get('error', 'unscored')}")
            continue
        capacity = solo['C'] if solo['C'] is not None else float('inf')
        incomplete = '' if solo['budget_complete'] else \
            '   [incomplete: ' + ','.join(solo['budgets_not_measured']) + ']'
        print(f"  {row['name']:26} {row['latency_ms']:9.3f} {solo['U_max']:7.4f} "
              f"{capacity:7.3f} {solo['L']:7.3f} {solo['N']:7.3f}  {solo['cause']}" + incomplete)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', help='JSON file (or - for stdin) holding a list of row dicts')
    parser.add_argument('--name')
    parser.add_argument('--latency-ms', type=float)
    parser.add_argument('--hz', type=float)
    parser.add_argument('--deadline-ms', type=float)
    parser.add_argument('--arch-gflops', type=float, default=0.0)
    parser.add_argument('--bytes-mb', type=float)
    parser.add_argument('--vram-mb', type=float)
    parser.add_argument('--latency-source', default='caller-supplied',
                        help='provenance of the latency: gpu-compute-time | e2e-gpu-total | ...')
    parser.add_argument('--ceilings', help='the ceilings run raw/results.json')
    parser.add_argument('--device', help='device config json (for the VRAM capacity)')
    parser.add_argument('--vram-capacity-mb', type=float)
    parser.add_argument('--out')
    parser.add_argument('--json-only', action='store_true')
    args = parser.parse_args()

    bw_eff, bw_source = bw_eff_from_ceilings(args.ceilings)
    vram_cap, vram_cap_source = vram_capacity_mb(args.device, args.vram_capacity_mb)
    rows = load_rows(args, parser)
    scored = [score_row(dict(row), bw_eff, vram_cap) for row in rows]
    doc = {'schema': 'budgets/v1',
           'bw_eff_gbps': bw_eff, 'bw_ceiling_source': bw_source,
           'vram_capacity_mb': vram_cap, 'vram_capacity_source': vram_cap_source,
           'rows': scored}

    if args.out:
        json.dump(doc, open(args.out, 'w'), indent=1)
    if args.json_only:
        print(json.dumps(doc, indent=1))
        return 0
    print(f"  bw_eff {bw_eff} GB/s ({bw_source})   vram cap {vram_cap} MB ({vram_cap_source})")
    print_table(scored)
    if args.out:
        print(f"  -> {args.out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
