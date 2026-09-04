#!/usr/bin/env python3
"""Period makespan of a co-located arm from the per-row frame traces
(trace_<row>.csv: frame,t_launch_ns,t_done_ns,gpu_ms; CLOCK_MONOTONIC ns, one launch per period).

  period_makespan.py <arm_dir> [period_ms] [--mix <resolved mix.json>] [--out <json>]

Default period = the fastest frame row's period when --mix is given, else the positional period,
else 33.333 ms. Keys: period_ms, periods, launch_phase_ms, aligned (every row's launch phase on
the shared grid within 2 ms - the shared-trigger precondition; the verdict marks the cell invalid
when False), makespan_p50/p99/max_ms (last completion in the window, relative to the window start,
over the windows where every row launched), all_rows_done_in_period_frac, overlap_ms_per_period_p50,
overlap_frac_of_period_p50, rows{<row>:{done_p99_ms, frame_overlapped_frac_p50, own_period_ms,
deadline_ms, n_frames, done_before_next_trigger_frac (t_done <= the row's OWN next nominal trigger),
deadline_miss_frac_from_trace, gpu_ms_p50/p99}}.
windows{"<P>ms": {...}}: one entry per distinct period among the frame rows; the P-window counts
the rows whose own period <= P (a 50 ms window = the >= 20 Hz rows; the 100 ms window = every row).
Rows with a missing/empty trace are reported as null, never a crash; rows_missing lists them.
overlap_basis: 'all-fire windows of the shared grid', or 'every period of the run' when the rows
never share a window (phase-shifted / disjoint grids) - the makespan keys then stay null.
"""
import sys, glob, os, json, argparse, warnings, numpy as np

def read_mix(path):
    """frame rows of a resolved mix (mixes.py resolve): hz / deadline / period per row"""
    m = json.load(open(path)); out = {}
    for r in m["rows"]:
        if r["role"] != "frame": continue
        hz = float(r["hz"]); out[r["name"]] = {"hz": hz, "deadline_ms": float(r["deadline_ms"]), "period_ms": (1000.0 / hz) if hz > 0 else None}
    return out

def load_trace(f):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); a = np.genfromtxt(f, delimiter=",", skip_header=1)
    except Exception:
        return None
    if a is None or a.size == 0: return None
    if a.ndim == 1: a = a.reshape(1, -1)
    if a.shape[1] < 4 or a.shape[0] < 1: return None
    a = a[np.isfinite(a[:, 1]) & np.isfinite(a[:, 2])]
    return a if a.shape[0] else None

def window_stats(valid, t0, P, names):
    """makespan per P-window over `names` (rows whose own period <= P). Window index = floor((t_launch - t0 + tol)/P): a row that
    launches several times inside the window (faster than P) contributes its latest completion."""
    tol = min(1.0, 0.1 * P); per_row = {}
    for n in names:
        a = valid[n]; k = np.floor(((a[:, 1] - t0) / 1e6 + tol) / P).astype(int); done = (a[:, 2] - t0) / 1e6 - k * P
        dd = {}
        for ki, di in zip(k.tolist(), done.tolist()): dd[ki] = max(dd.get(ki, -1e18), di)
        per_row[n] = dd
    common = sorted(set.intersection(*[set(v) for v in per_row.values()])) if per_row else []
    if len(common) > 2: common = common[1:-1]
    base = {"period_ms": P, "rows": list(names), "periods": len(common), "makespan_p50_ms": None, "makespan_p99_ms": None, "makespan_max_ms": None, "all_rows_done_frac": None}
    if not common: return base
    mk = np.array([max(per_row[n][k] for n in names) for k in common])
    base.update({"makespan_p50_ms": float(np.median(mk)), "makespan_p99_ms": float(np.percentile(mk, 99)), "makespan_max_ms": float(mk.max()), "all_rows_done_frac": float(np.mean(mk <= P))})
    return base

def own_period_stats(a, t0, period, deadline):
    """judge every frame against the row's OWN nominal grid: phase = median launch offset inside the period (phase-shifted regime aware);
    next trigger of frame k = t0 + phase + (k+1)*period."""
    n = int(a.shape[0]); out = {"n_frames": n, "own_period_ms": period, "deadline_ms": deadline, "done_before_next_trigger_frac": None, "deadline_miss_frac_from_trace": None,
                                "gpu_ms_p50": float(np.median(a[:, 3])) if np.isfinite(a[:, 3]).all() else None, "gpu_ms_p99": float(np.percentile(a[:, 3], 99)) if np.isfinite(a[:, 3]).all() else None}
    if deadline is not None and np.isfinite(a[:, 3]).all(): out["deadline_miss_frac_from_trace"] = float(np.mean(a[:, 3] > deadline))
    if period:
        # next nominal trigger of frame k = its OWN launch + period. The pacer slides after an overrun (row_loop: nxt = now), so a
        # fixed grid anchored at the first frame drifts away from the real launches once a row has overrun; the launch itself is
        # the trigger the frame answered to. slide = net drift of the launches against a fixed grid over the run (0 when no overrun)
        tl = (a[:, 1] - t0) / 1e6; td = (a[:, 2] - t0) / 1e6
        out["done_before_next_trigger_frac"] = float(np.mean(td <= tl + period)); out["launch_phase_own_ms"] = float(np.mod(tl[0], period))
        out["launch_slide_ms"] = float((tl[-1] - tl[0]) - (n - 1) * period) if n > 1 else 0.0
    return out

def analyse(d, per=None, mix=None):
    mix = mix or {}
    files = {os.path.basename(f)[6:-4]: f for f in sorted(glob.glob(os.path.join(d, "trace_*.csv")))}
    if mix: files = {k: v for k, v in files.items() if k in mix}          # only the rows of THIS arm's mix (stale traces from an earlier run in the same dir must not set the grid reference)
    names = list(mix) if mix else list(files)
    for n in files:
        if n not in names: names.append(n)                              # traces not in the mix (extra rows) are still analysed
    traces = {n: (load_trace(files[n]) if n in files else None) for n in names}
    valid = {n: a for n, a in traces.items() if a is not None}; missing = [n for n in names if traces[n] is None]
    if per is None:
        ps = [v["period_ms"] for v in mix.values() if v.get("period_ms")]
        per = min(ps) if ps else 1000 / 30
    out = {"period_ms": per, "periods": 0, "launch_phase_ms": {}, "aligned": None, "makespan_p50_ms": None, "makespan_p99_ms": None, "makespan_max_ms": None,
           "all_rows_done_in_period_frac": None, "overlap_ms_per_period_p50": None, "overlap_frac_of_period_p50": None, "rows": {n: None for n in names},
           "windows": {}, "rows_missing": missing, "mix": mix and {n: {"hz": v["hz"], "deadline_ms": v["deadline_ms"]} for n, v in mix.items()} or None}
    if not valid: return out, None
    t0 = min(a[:, 1].min() for a in valid.values()); pn = {}; phase = {}
    for n, a in valid.items():
        k = np.round((a[:, 1] - t0) / (per * 1e6)).astype(int)                  # window index of each launch (rows share one grid; round absorbs jitter)
        done = (a[:, 2] - t0) / 1e6 - k * per                                     # completion relative to the window start (ms)
        pn[n] = dict(zip(k.tolist(), done.tolist())); off = (a[:, 1] - t0) / 1e6 - k * per; phase[n] = float(off[0])   # LAUNCH phase = the FIRST frame's offset on the shared grid: the barrier releases every row within microseconds, so anything beyond the 2 ms tolerance is a launch on a different phase. Later frames are excluded on purpose - a row that overruns its period launches late from frame 1 on, which is load (reported as misses / done-before-next-trigger), not misalignment
    common = sorted(set.intersection(*[set(v) for v in pn.values()]))
    if len(common) > 2: common = common[1:-1]                                     # windows where every row launched
    # overlap: per period, the time during which >= 2 rows were executing at once (union of pairwise interval intersections)
    iv = {n: {int(k): ((a[i, 1] - t0) / 1e6, (a[i, 2] - t0) / 1e6) for i, k in enumerate(np.round((a[:, 1] - t0) / (per * 1e6)).astype(int))} for n, a in valid.items()}
    rn = list(iv); ov = []; ovrow = {n: [] for n in rn}
    for k in common:
        segs = [iv[n][k] for n in rn]; ev = sorted({t for s_ in segs for t in s_}); tot = 0.0
        for a_, b_ in zip(ev[:-1], ev[1:]):
            cnt = sum(1 for s_ in segs if s_[0] <= a_ and s_[1] >= b_)
            if cnt >= 2: tot += b_ - a_
        ov.append(tot)
        for n in rn:
            s0, s1 = iv[n][k]; o = 0.0
            for m in rn:
                if m == n: continue
                m0, m1 = iv[m][k]; o += max(0.0, min(s1, m1) - max(s0, m0))
            ovrow[n].append(min(o, s1 - s0) / max(s1 - s0, 1e-9))
    overlap_basis = "all-fire windows of the shared grid"
    if not common:
        # rows never share a window (phase-shifted regime with mixed rates, or disjoint grids): overlap over EVERY default period of the run
        # (time with >= 2 rows executing, bucketed by period) and per-frame overlap over all frames — the makespan keys stay null
        overlap_basis = "every period of the run (rows never share a window: phase-shifted / disjoint grids)"
        starts = np.concatenate([(a[:, 1] - t0) / 1e6 for a in valid.values()]); ends = np.concatenate([(a[:, 2] - t0) / 1e6 for a in valid.values()])
        ev = np.unique(np.concatenate([starts, ends])); nb = max(1, int(np.ceil(ends.max() / per))); bucket = np.zeros(nb)
        for a_, b_ in zip(ev[:-1].tolist(), ev[1:].tolist()):
            if b_ <= a_ or int(np.sum((starts <= a_) & (ends >= b_))) < 2: continue
            for k in range(int(a_ // per), min(int(np.ceil(b_ / per)), nb)):
                lo, hi = max(a_, k * per), min(b_, (k + 1) * per)
                if hi > lo: bucket[k] += hi - lo
        ov = bucket.tolist()
        for n, a in valid.items():
            s0 = (a[:, 1] - t0) / 1e6; s1 = (a[:, 2] - t0) / 1e6; o = np.zeros(a.shape[0])
            for m, b in valid.items():
                if m == n: continue
                m0 = (b[:, 1] - t0) / 1e6; m1 = (b[:, 2] - t0) / 1e6
                o += np.clip(np.minimum(s1[:, None], m1[None, :]) - np.maximum(s0[:, None], m0[None, :]), 0, None).sum(axis=1)
            ovrow[n] = (np.minimum(o, s1 - s0) / np.maximum(s1 - s0, 1e-9)).tolist()
    ov = np.array(ov) if ov else np.array([np.nan])
    mk = np.array([max(pn[n][k] for n in pn) for k in common]) if common else np.array([np.nan])
    out.update({"periods": len(common), "launch_phase_ms": phase, "aligned": bool(max(abs(v) for v in phase.values()) < 2.0), "overlap_basis": overlap_basis})
    slid = {n: float(((a[-1, 1] - a[0, 1]) / 1e6) - (a.shape[0] - 1) * (mix.get(n, {}).get("period_ms") or per)) for n, a in valid.items() if a.shape[0] > 1}
    out["rows_slid"] = {n: v for n, v in slid.items() if abs(v) > min(1.0, 0.1 * per)}
    out["makespan_basis"] = ("shared grid" if not out["rows_slid"] else
                             "shared grid; UPPER BOUND - the pacer of " + ", ".join(f"{n} slid {v:+.1f} ms" for n, v in out["rows_slid"].items()) +
                             " after overruns, so those rows' launches drifted off the grid and their completion is measured from the grid window, not from the slid launch")
    if common:
        out.update({"makespan_p50_ms": float(np.median(mk)), "makespan_p99_ms": float(np.percentile(mk, 99)), "makespan_max_ms": float(mk.max()), "all_rows_done_in_period_frac": float(np.mean(mk <= per))})
    if np.isfinite(ov).any():
        out.update({"overlap_ms_per_period_p50": float(np.nanmedian(ov)), "overlap_frac_of_period_p50": float(np.nanmedian(ov) / per)})
    for n, v in pn.items():
        r = {"done_p99_ms": float(np.percentile(list(v.values()), 99)), "frame_overlapped_frac_p50": float(np.median(ovrow[n])) if ovrow[n] else None}
        m = mix.get(n, {}); r.update(own_period_stats(valid[n], t0, m.get("period_ms") or per, m.get("deadline_ms")))
        out["rows"][n] = r
    # per-rate windows: one per distinct period among the rows (mix periods; without a mix, just the default period)
    periods = sorted({mix[n]["period_ms"] for n in valid if n in mix and mix[n].get("period_ms")}) or [per]
    for P in periods:
        rows_P = [n for n in valid if (mix.get(n, {}).get("period_ms") or per) <= P + 1e-6]
        if rows_P: out["windows"][f"{P:g}ms"] = window_stats(valid, t0, P, rows_P)
    return out, mk

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("period", nargs="?", type=float, default=None)
    ap.add_argument("--mix", default=None, help="resolved mix json (mixes.py resolve): per-row hz/deadline -> per-rate windows + own-period judgement")
    ap.add_argument("--out", default=None, help="output json (default <dir>/period_makespan.json)")
    a = ap.parse_args(); d = a.dir; mix = read_mix(a.mix) if a.mix else {}
    out, mk = analyse(d, a.period, mix)
    json.dump(out, open(a.out or os.path.join(d, "period_makespan.json"), "w"), indent=1)
    if out["periods"] == 0 and not any(v for v in out["rows"].values()):
        print("no traces in", d, "(rows:", out["rows_missing"], ") — wrote a null period_makespan.json"); sys.exit(1)
    if out["aligned"] is False: print(f"NOT ALIGNED {d}: rows launched on different phases ({ {k: round(v,1) for k, v in out['launch_phase_ms'].items()} } ms) - makespan is not a shared-trigger number (the verdict marks this cell invalid); overlap is still valid")
    if out["rows_missing"]: print(f"WARNING {d}: no/empty trace for {out['rows_missing']} — reported as null")
    f = lambda v: "n/a" if v is None else f"{v:.2f}"
    pc = lambda v, d=1: "n/a" if v is None else f"{100*v:.{d}f}%"
    ovl = ", ".join(f"{n} {pc(r['frame_overlapped_frac_p50'], 0)} of its frame overlapped" for n, r in out["rows"].items() if r and r.get("frame_overlapped_frac_p50") is not None)
    print(f"{d}: {out['periods']} periods · makespan p50 {f(out['makespan_p50_ms'])} / p99 {f(out['makespan_p99_ms'])} / max {f(out['makespan_max_ms'])} ms · all rows done within the period: "
          f"{pc(out['all_rows_done_in_period_frac'])} · overlap {f(out['overlap_ms_per_period_p50'])} ms/period ({pc(out['overlap_frac_of_period_p50'], 0)}): " + ovl)
    for w, s in out["windows"].items():
        print(f"  window {w}: rows {s['rows']} · {s['periods']} windows · makespan p50 {f(s['makespan_p50_ms'])} / p99 {f(s['makespan_p99_ms'])} / max {f(s['makespan_max_ms'])} ms · complete {pc(s['all_rows_done_frac'])}")
    for n, r in out["rows"].items():
        if r: print(f"  row {n}: own period {f(r['own_period_ms'])} ms · done before next trigger {pc(r['done_before_next_trigger_frac'])} · deadline miss (trace) {pc(r['deadline_miss_frac_from_trace'], 2)} · n {r['n_frames']}" + (f" · pacer slid {r['launch_slide_ms']:+.1f} ms" if abs(r.get('launch_slide_ms') or 0) > 1 else ""))
if __name__ == "__main__": main()
