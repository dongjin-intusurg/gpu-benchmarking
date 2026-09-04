#!/usr/bin/env python3
"""Demand budgets -> U_max, C, L, N, Score for ONE measured row, from ANY source.

This is the stage-5 solo block of run_model_bench.sh, lifted out so a latency
does not have to come from trtexec to earn an N. The formulas are unchanged:

    time_share = latency_ms/1e3 * hz
    U_bw       = (bytes_per_frame_MB/1e3 * hz) / bw_eff_gbps
    U_vram     = vram_mb / vram_capacity_mb
    U_max      = max(the budgets that are > 0)
    C          = 1 / U_max
    L          = deadline_ms / latency_ms
    N          = min(L, C)
    cause      = latency-limited if L < C else throughput-limited
    Score      = N * (arch_gflops * hz) / 1e3      TFLOP/s at the deadline

The latency of record differs by row type and the caller decides which it is:
a frame p99 for a single engine, the whole-request time for an end-to-end
driver. Everything downstream of these four inputs is identical either way.

One deliberate difference from the original: a MISSING bytes/frame is recorded
as an incomplete budget, not silently treated as zero. In the original device
rollup a model with no byte measurement contributes 0 to U_bw via
`sum(m.get('bw_demand_gbps', 0) ...)`, so it looks free on the bandwidth axis
instead of unmeasured. Here `budget_complete` is false and `bytes_source` says
'none', so a reader can tell "no bandwidth demand" from "bandwidth demand not
measured".

Usage:
  compute_budgets.py --latency-ms 16.507 --hz 30 --deadline-ms 33.3 \
      --arch-gflops 17.09 --bytes-mb 502.0 --vram-mb 260.4 \
      --ceilings <ceilings raw/results.json> [--device <device_cfg.json>]
      [--name depth_model] [--latency-source gpu-compute-time] [--json-only]

  # or score many rows at once from a JSON array on stdin / --rows file
  compute_budgets.py --rows rows.json --ceilings ... --device ...

Exit 0 always when the inputs parse; a row that cannot be scored carries
`"error"` instead of `solo`.
"""
import argparse, json, math, os, sys


def bw_eff_from_ceilings(path):
    """Budget-2 denominator: the best idle copy_RW over the buffer sweep.

    Same selection as run_model_bench.sh stage 5 - the solo-exclusive regime
    budgets against idle bandwidth, not a contended figure.
    """
    if not path or not os.path.exists(path):
        return None, 'ceilings not provided'
    try:
        cj = json.load(open(path))
    except Exception as e:
        return None, f'ceilings unreadable: {e}'
    best = None
    for row in (cj.get('bandwidth') or {}).values():
        v = row.get('copy_RW') if isinstance(row, dict) else None
        if isinstance(v, (int, float)) and (best is None or v > best):
            best = v
    return best, ('thorough-ceilings idle copy_RW best' if best else 'no copy_RW in ceilings')


def vram_capacity_mb(device_cfg, override=None):
    """Budget-3 denominator, same precedence as stage 5."""
    if override:
        return float(override), 'env-override'
    if device_cfg and os.path.exists(device_cfg):
        try:
            cfg = json.load(open(device_cfg))
        except Exception:
            cfg = {}
        cap = cfg.get('vram_budget_cap_mb')
        if cap:
            return float(cap), 'device-config-cap'
        if (cfg.get('platform') or '') == 'jetson':
            try:
                for line in open('/proc/meminfo'):
                    if line.startswith('MemTotal:'):
                        return float(line.split()[1]) / 1024.0, 'meminfo-unified'
            except Exception:
                pass
    try:
        import subprocess
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.total',
                              '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        if out:
            return float(out.splitlines()[0]), 'nvidia-smi-total'
    except Exception:
        pass
    return None, 'unavailable'


def score_row(row, bw_eff, vram_cap):
    """The stage-5 solo verdict for one row. Returns the row with 'solo' added."""
    name = row.get('name', '?')
    lat = row.get('latency_ms')
    hz = row.get('hz')
    dl = row.get('deadline_ms')
    if not lat or lat <= 0:
        row['error'] = 'no latency of record — nothing to score'
        return row
    if hz is None or dl is None:
        row['error'] = 'hz and deadline_ms are required (they are the mix definition, not measurements)'
        return row
    hz = float(hz); dl = float(dl); lat = float(lat)
    agf = float(row.get('arch_gflops') or 0.0)

    b, incomplete = {'time_occupancy': lat / 1e3 * hz}, []

    bytes_mb = row.get('bytes_per_frame_MB')
    if bytes_mb is not None and bw_eff:
        # GB/s == MB/ms, so bytes_MB/1e3 * hz is GB/s of demand
        row['bw_demand_gbps'] = bytes_mb / 1e3 * hz
        b['dram_bandwidth'] = row['bw_demand_gbps'] / bw_eff
    else:
        # NOT zero: unmeasured. Saying 0 here would make the row look free on
        # the axis that decides most generative models.
        incomplete.append('dram_bandwidth' if bw_eff else 'dram_bandwidth (no bw_eff)')
        row.setdefault('bytes_source', 'none')

    vram = row.get('vram_mb')
    if vram and vram_cap:
        b['vram_footprint'] = float(vram) / vram_cap
    else:
        incomplete.append('vram_footprint')

    if not (any(v > 0 for v in b.values()) or hz == 0):
        row['error'] = 'every budget is zero — nothing to divide'
        return row

    um = max((v for v in b.values() if v > 0), default=0.0)
    c = (1.0 / um) if um > 0 else float('inf')
    l = dl / lat
    n = min(l, c)
    w = agf * hz / 1e3

    row['solo'] = {
        'budgets': {k: round(v, 4) for k, v in b.items()},
        'U_max': round(um, 4),
        'C': round(c, 3) if c != float('inf') else None,
        'L': round(l, 3),
        'N': round(n, 3),
        'cause': 'latency-limited' if l < c else 'throughput-limited',
        'score_tflops': round(n * w, 3),
        'budget_complete': not incomplete,
    }
    if incomplete:
        row['solo']['budgets_not_measured'] = incomplete
        row['solo']['caveat'] = ('U_max is a LOWER bound: ' + ', '.join(incomplete) +
                                 ' not measured, so the true binding budget may be higher and N lower')
    grid = row.get('hz_grid')
    if grid:
        # the spec rate (and with it the deadline) is undecided for this row:
        # score the same budgets at each candidate rate with the period as the
        # deadline (time and bandwidth scale with hz, VRAM does not) and report
        # the highest rate that still yields N >= 1. The declared hz/deadline
        # placeholders govern the headline N above, not this sweep.
        nv = {}
        for h in grid:
            bh = {'time_occupancy': lat / 1e3 * h}
            if bytes_mb is not None and bw_eff:
                bh['dram_bandwidth'] = bytes_mb / 1e3 * h / bw_eff
            if vram and vram_cap:
                bh['vram_footprint'] = float(vram) / vram_cap
            umh = max(bh.values())
            nv[str(h)] = round(min((1e3 / h) / lat, 1.0 / umh), 3) if umh > 0 else None
        row['solo']['N_vs_hz'] = nv
        row['solo']['N_vs_hz_deadline'] = 'period (1000/hz ms)'
        if vram and vram_cap and float(vram) / vram_cap >= 1:
            row['solo']['max_hz_at_N1'], row['solo']['max_hz_bound_by'] = 0.0, 'vram'
        else:
            cands = {'time': 1e3 / lat}
            if bytes_mb is not None and bw_eff:
                cands['dram_bandwidth'] = bw_eff * 1e3 / bytes_mb
            k = min(cands, key=cands.get)
            row['solo']['max_hz_at_N1'], row['solo']['max_hz_bound_by'] = round(cands[k], 2), k
    if hz == 0:
        row['solo']['score_tflops'] = None
        row['solo']['modal'] = ('hz=0: no sustained time/bandwidth bill; N is the verdict '
                                '(verify L against the CONTENDED latency)')
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', help='JSON file (or - for stdin) holding a list of row dicts')
    ap.add_argument('--name'); ap.add_argument('--latency-ms', type=float)
    ap.add_argument('--hz', type=float); ap.add_argument('--deadline-ms', type=float)
    ap.add_argument('--arch-gflops', type=float, default=0.0)
    ap.add_argument('--bytes-mb', type=float); ap.add_argument('--vram-mb', type=float)
    ap.add_argument('--latency-source', default='caller-supplied',
                    help='provenance of the latency: gpu-compute-time | e2e-gpu-total | ...')
    ap.add_argument('--ceilings', help='the ceilings run raw/results.json')
    ap.add_argument('--device', help='device config json (for the VRAM capacity)')
    ap.add_argument('--vram-capacity-mb', type=float)
    ap.add_argument('--out'); ap.add_argument('--json-only', action='store_true')
    a = ap.parse_args()

    bw_eff, bw_src = bw_eff_from_ceilings(a.ceilings)
    vcap, vcap_src = vram_capacity_mb(a.device, a.vram_capacity_mb)

    if a.rows:
        raw = sys.stdin.read() if a.rows == '-' else open(a.rows).read()
        rows = json.loads(raw)
        if isinstance(rows, dict):
            rows = [dict(v, name=k) for k, v in rows.items()]
    else:
        if a.latency_ms is None:
            ap.error('give --rows, or --latency-ms with --hz and --deadline-ms')
        rows = [{'name': a.name or 'row', 'latency_ms': a.latency_ms, 'hz': a.hz,
                 'deadline_ms': a.deadline_ms, 'arch_gflops': a.arch_gflops,
                 'bytes_per_frame_MB': a.bytes_mb, 'vram_mb': a.vram_mb,
                 'latency_source': a.latency_source}]

    scored = [score_row(dict(r), bw_eff, vcap) for r in rows]
    doc = {'schema': 'budgets/v1',
           'bw_eff_gbps': bw_eff, 'bw_ceiling_source': bw_src,
           'vram_capacity_mb': vcap, 'vram_capacity_source': vcap_src,
           'rows': scored}

    if a.out:
        json.dump(doc, open(a.out, 'w'), indent=1)
    if a.json_only:
        print(json.dumps(doc, indent=1))
        return 0

    print(f"  bw_eff {bw_eff} GB/s ({bw_src})   vram cap {vcap} MB ({vcap_src})")
    print(f"  {'row':26} {'latency':>9} {'U_max':>7} {'C':>7} {'L':>7} {'N':>7}  cause")
    for r in scored:
        s = r.get('solo')
        if not s:
            print(f"  {r.get('name','?'):26} {r.get('error','unscored')}")
            continue
        print(f"  {r['name']:26} {r['latency_ms']:9.3f} {s['U_max']:7.4f} "
              f"{(s['C'] if s['C'] is not None else float('inf')):7.3f} {s['L']:7.3f} {s['N']:7.3f}  {s['cause']}"
              + ('' if s['budget_complete'] else '   [incomplete: ' + ','.join(s['budgets_not_measured']) + ']'))
    if a.out:
        print(f"  -> {a.out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
