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
import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np

ALIGNED_TOLERANCE_MS = 2.0
DEFAULT_PERIOD_MS = 1000 / 30
COL_LAUNCH, COL_DONE, COL_GPU_MS = 1, 2, 3


def read_mix(path):
    """frame rows of a resolved mix (mixes.py resolve): hz / deadline / period per row"""
    mix = json.load(open(path))
    out = {}
    for row in mix["rows"]:
        if row["role"] != "frame":
            continue
        hz = float(row["hz"])
        out[row["name"]] = {"hz": hz, "deadline_ms": float(row["deadline_ms"]),
                            "period_ms": (1000.0 / hz) if hz > 0 else None}
    return out


def load_trace(path):
    """-> rows x 4 array of finite launch/done frames, or None for a missing/empty/unreadable trace"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace = np.genfromtxt(path, delimiter=",", skip_header=1)
    except Exception:
        return None
    if trace is None or trace.size == 0:
        return None
    if trace.ndim == 1:
        trace = trace.reshape(1, -1)
    if trace.shape[1] < 4 or trace.shape[0] < 1:
        return None
    trace = trace[np.isfinite(trace[:, COL_LAUNCH]) & np.isfinite(trace[:, COL_DONE])]
    return trace if trace.shape[0] else None


def _grid_tolerance_ms(period_ms):
    return min(1.0, 0.1 * period_ms)


def _launch_ms(trace, t0_ns):
    return (trace[:, COL_LAUNCH] - t0_ns) / 1e6


def _done_ms(trace, t0_ns):
    return (trace[:, COL_DONE] - t0_ns) / 1e6


def _shared_windows(per_row_windows):
    """Window indices where every row launched; first and last (partial) windows dropped when enough."""
    if not per_row_windows:
        return []
    common = sorted(set.intersection(*[set(windows) for windows in per_row_windows.values()]))
    if len(common) > 2:
        common = common[1:-1]
    return common


def window_stats(valid, t0_ns, period_ms, names):
    """makespan per P-window over `names` (rows whose own period <= P).
    Window index = floor((t_launch - t0 + tol)/P): a row that launches several times inside the window
    (faster than P) contributes its latest completion."""
    tolerance = _grid_tolerance_ms(period_ms)
    latest_done = {}
    for name in names:
        trace = valid[name]
        window = np.floor((_launch_ms(trace, t0_ns) + tolerance) / period_ms).astype(int)
        done = _done_ms(trace, t0_ns) - window * period_ms
        per_window = {}
        for window_index, done_ms in zip(window.tolist(), done.tolist()):
            per_window[window_index] = max(per_window.get(window_index, -1e18), done_ms)
        latest_done[name] = per_window
    common = _shared_windows(latest_done)
    stats = {"period_ms": period_ms, "rows": list(names), "periods": len(common), "makespan_p50_ms": None,
             "makespan_p99_ms": None, "makespan_max_ms": None, "all_rows_done_frac": None}
    if not common:
        return stats
    makespan = np.array([max(latest_done[name][window_index] for name in names) for window_index in common])
    stats.update({"makespan_p50_ms": float(np.median(makespan)),
                  "makespan_p99_ms": float(np.percentile(makespan, 99)),
                  "makespan_max_ms": float(makespan.max()),
                  "all_rows_done_frac": float(np.mean(makespan <= period_ms))})
    return stats


def own_period_stats(trace, t0_ns, period_ms, deadline_ms):
    """Judge every frame against the row's OWN grid. The next nominal trigger of frame k is its own launch +
    period: the pacer slides after an overrun (row_loop: nxt = now), so a grid anchored at the first frame
    drifts away from the real launches once a row has overrun. launch_slide_ms = net drift of the launches
    against a fixed grid over the run (0 when no overrun)."""
    n_frames = int(trace.shape[0])
    gpu_ms = trace[:, COL_GPU_MS]
    gpu_finite = np.isfinite(gpu_ms).all()
    out = {"n_frames": n_frames, "own_period_ms": period_ms, "deadline_ms": deadline_ms,
           "done_before_next_trigger_frac": None, "deadline_miss_frac_from_trace": None,
           "gpu_ms_p50": float(np.median(gpu_ms)) if gpu_finite else None,
           "gpu_ms_p99": float(np.percentile(gpu_ms, 99)) if gpu_finite else None}
    if deadline_ms is not None and gpu_finite:
        out["deadline_miss_frac_from_trace"] = float(np.mean(gpu_ms > deadline_ms))
    if period_ms:
        launch = _launch_ms(trace, t0_ns)
        done = _done_ms(trace, t0_ns)
        out["done_before_next_trigger_frac"] = float(np.mean(done <= launch + period_ms))
        out["launch_phase_own_ms"] = float(np.mod(launch[0], period_ms))
        slide = float((launch[-1] - launch[0]) - (n_frames - 1) * period_ms) if n_frames > 1 else 0.0
        out["launch_slide_ms"] = slide
    return out


def _trace_files(arm_dir, mix):
    """{row: trace path}; with a mix only ITS rows count (stale traces from an earlier run in the same dir
    must not set the grid reference)."""
    paths = sorted(glob.glob(os.path.join(arm_dir, "trace_*.csv")))
    files = {os.path.basename(path)[len("trace_"):-len(".csv")]: path for path in paths}
    if mix:
        files = {name: path for name, path in files.items() if name in mix}
    return files


def _null_result(period_ms, names, missing, mix):
    mix_summary = {name: {"hz": row["hz"], "deadline_ms": row["deadline_ms"]} for name, row in mix.items()}
    return {"period_ms": period_ms, "periods": 0, "launch_phase_ms": {}, "aligned": None,
            "makespan_p50_ms": None, "makespan_p99_ms": None, "makespan_max_ms": None,
            "all_rows_done_in_period_frac": None,
            "overlap_ms_per_period_p50": None, "overlap_frac_of_period_p50": None,
            "rows": {name: None for name in names}, "windows": {}, "rows_missing": missing,
            "mix": mix and mix_summary or None}


def _grid_windows(trace, t0_ns, period_ms):
    """Window index of every launch on the shared grid (rows share one grid; round absorbs jitter)."""
    return np.round((trace[:, COL_LAUNCH] - t0_ns) / (period_ms * 1e6)).astype(int)


def _overlap_shared_grid(intervals, common):
    """Per all-fire window: time with >= 2 rows executing (union of pairwise intersections), and per row the
    overlapped fraction of its own frame."""
    names = list(intervals)
    overlap_per_window = []
    overlapped_frac = {name: [] for name in names}
    for window_index in common:
        segments = [intervals[name][window_index] for name in names]
        edges = sorted({edge for segment in segments for edge in segment})
        total = 0.0
        for left, right in zip(edges[:-1], edges[1:]):
            covering = sum(1 for segment in segments if segment[0] <= left and segment[1] >= right)
            if covering >= 2:
                total += right - left
        overlap_per_window.append(total)
        for name in names:
            start, end = intervals[name][window_index]
            overlap = 0.0
            for other in names:
                if other == name:
                    continue
                other_start, other_end = intervals[other][window_index]
                overlap += max(0.0, min(end, other_end) - max(start, other_start))
            overlapped_frac[name].append(min(overlap, end - start) / max(end - start, 1e-9))
    return overlap_per_window, overlapped_frac


def _overlap_every_period(valid, t0_ns, period_ms):
    """Rows never share a window (phase-shifted regime with mixed rates, or disjoint grids): overlap over
    EVERY default period of the run (time with >= 2 rows executing, bucketed by period) and per-frame
    overlap over all frames."""
    starts = np.concatenate([_launch_ms(trace, t0_ns) for trace in valid.values()])
    ends = np.concatenate([_done_ms(trace, t0_ns) for trace in valid.values()])
    edges = np.unique(np.concatenate([starts, ends]))
    n_buckets = max(1, int(np.ceil(ends.max() / period_ms)))
    bucket = np.zeros(n_buckets)
    for left, right in zip(edges[:-1].tolist(), edges[1:].tolist()):
        if right <= left or int(np.sum((starts <= left) & (ends >= right))) < 2:
            continue
        for window_index in range(int(left // period_ms), min(int(np.ceil(right / period_ms)), n_buckets)):
            low = max(left, window_index * period_ms)
            high = min(right, (window_index + 1) * period_ms)
            if high > low:
                bucket[window_index] += high - low
    overlapped_frac = {}
    for name, trace in valid.items():
        start = _launch_ms(trace, t0_ns)
        end = _done_ms(trace, t0_ns)
        overlap = np.zeros(trace.shape[0])
        for other, other_trace in valid.items():
            if other == name:
                continue
            other_start = _launch_ms(other_trace, t0_ns)
            other_end = _done_ms(other_trace, t0_ns)
            pairwise_end = np.minimum(end[:, None], other_end[None, :])
            pairwise_start = np.maximum(start[:, None], other_start[None, :])
            overlap += np.clip(pairwise_end - pairwise_start, 0, None).sum(axis=1)
        overlapped_frac[name] = (np.minimum(overlap, end - start) / np.maximum(end - start, 1e-9)).tolist()
    return bucket.tolist(), overlapped_frac


def _makespan_basis(rows_slid):
    if not rows_slid:
        return "shared grid"
    slid = ", ".join(f"{name} slid {value:+.1f} ms" for name, value in rows_slid.items())
    return ("shared grid; UPPER BOUND - the pacer of " + slid + " after overruns, so those rows' launches "
            "drifted off the grid and their completion is measured from the grid window, "
            "not from the slid launch")


def analyse(arm_dir, period_ms=None, mix=None):
    mix = mix or {}
    files = _trace_files(arm_dir, mix)
    names = list(mix) if mix else list(files)
    for name in files:
        if name not in names:
            names.append(name)
    traces = {name: (load_trace(files[name]) if name in files else None) for name in names}
    valid = {name: trace for name, trace in traces.items() if trace is not None}
    missing = [name for name in names if traces[name] is None]
    if period_ms is None:
        periods = [row["period_ms"] for row in mix.values() if row.get("period_ms")]
        period_ms = min(periods) if periods else DEFAULT_PERIOD_MS
    out = _null_result(period_ms, names, missing, mix)
    if not valid:
        return out, None

    t0_ns = min(trace[:, COL_LAUNCH].min() for trace in valid.values())
    done_by_window = {}
    phase = {}
    intervals = {}
    for name, trace in valid.items():
        window = _grid_windows(trace, t0_ns, period_ms)
        done = _done_ms(trace, t0_ns) - window * period_ms
        done_by_window[name] = dict(zip(window.tolist(), done.tolist()))
        # LAUNCH phase = the FIRST frame's offset on the shared grid: the barrier releases every row within
        # microseconds, so anything beyond the tolerance is a launch on a different phase. Later frames are
        # excluded on purpose - a row that overruns its period launches late from frame 1 on, which is load
        # (reported as misses / done-before-next-trigger), not misalignment.
        offset = _launch_ms(trace, t0_ns) - window * period_ms
        phase[name] = float(offset[0])
        launch = _launch_ms(trace, t0_ns)
        finish = _done_ms(trace, t0_ns)
        intervals[name] = {int(window_index): (launch[i], finish[i]) for i, window_index in enumerate(window)}
    common = _shared_windows(done_by_window)

    overlap_basis = "all-fire windows of the shared grid"
    overlap_per_window, overlapped_frac = _overlap_shared_grid(intervals, common)
    if not common:
        overlap_basis = "every period of the run (rows never share a window: phase-shifted / disjoint grids)"
        overlap_per_window, overlapped_frac = _overlap_every_period(valid, t0_ns, period_ms)
    overlap = np.array(overlap_per_window) if overlap_per_window else np.array([np.nan])
    makespan = np.array([np.nan])
    if common:
        makespan = np.array([max(done_by_window[name][window_index] for name in done_by_window)
                             for window_index in common])

    out.update({"periods": len(common), "launch_phase_ms": phase,
                "aligned": bool(max(abs(value) for value in phase.values()) < ALIGNED_TOLERANCE_MS),
                "overlap_basis": overlap_basis})
    slid = {name: float(((trace[-1, COL_LAUNCH] - trace[0, COL_LAUNCH]) / 1e6)
                        - (trace.shape[0] - 1) * (mix.get(name, {}).get("period_ms") or period_ms))
            for name, trace in valid.items() if trace.shape[0] > 1}
    out["rows_slid"] = {name: value for name, value in slid.items()
                        if abs(value) > _grid_tolerance_ms(period_ms)}
    out["makespan_basis"] = _makespan_basis(out["rows_slid"])
    if common:
        out.update({"makespan_p50_ms": float(np.median(makespan)),
                    "makespan_p99_ms": float(np.percentile(makespan, 99)),
                    "makespan_max_ms": float(makespan.max()),
                    "all_rows_done_in_period_frac": float(np.mean(makespan <= period_ms))})
    if np.isfinite(overlap).any():
        out.update({"overlap_ms_per_period_p50": float(np.nanmedian(overlap)),
                    "overlap_frac_of_period_p50": float(np.nanmedian(overlap) / period_ms)})
    for name, done in done_by_window.items():
        frame_overlapped = float(np.median(overlapped_frac[name])) if overlapped_frac[name] else None
        row = {"done_p99_ms": float(np.percentile(list(done.values()), 99)),
               "frame_overlapped_frac_p50": frame_overlapped}
        mix_row = mix.get(name, {})
        row.update(own_period_stats(valid[name], t0_ns, mix_row.get("period_ms") or period_ms,
                                    mix_row.get("deadline_ms")))
        out["rows"][name] = row
    # per-rate windows: one per distinct period among the rows (without a mix, just the default period)
    distinct_periods = sorted({mix[name]["period_ms"] for name in valid
                               if name in mix and mix[name].get("period_ms")})
    for window_period in distinct_periods or [period_ms]:
        rows_in_window = [name for name in valid
                          if (mix.get(name, {}).get("period_ms") or period_ms) <= window_period + 1e-6]
        if rows_in_window:
            out["windows"][f"{window_period:g}ms"] = window_stats(valid, t0_ns, window_period, rows_in_window)
    return out, makespan


def _fmt(value):
    return "n/a" if value is None else f"{value:.2f}"


def _fmt_pct(value, digits=1):
    return "n/a" if value is None else f"{100 * value:.{digits}f}%"


def print_summary(arm_dir, out):
    if out["aligned"] is False:
        phases = {name: round(value, 1) for name, value in out['launch_phase_ms'].items()}
        print(f"NOT ALIGNED {arm_dir}: rows launched on different phases ({phases} ms) - makespan is not a "
              f"shared-trigger number (the verdict marks this cell invalid); overlap is still valid")
    if out["rows_missing"]:
        print(f"WARNING {arm_dir}: no/empty trace for {out['rows_missing']} — reported as null")
    overlapped = ", ".join(f"{name} {_fmt_pct(row['frame_overlapped_frac_p50'], 0)} of its frame overlapped"
                           for name, row in out["rows"].items()
                           if row and row.get("frame_overlapped_frac_p50") is not None)
    print(f"{arm_dir}: {out['periods']} periods · makespan p50 {_fmt(out['makespan_p50_ms'])} / "
          f"p99 {_fmt(out['makespan_p99_ms'])} / max {_fmt(out['makespan_max_ms'])} ms · "
          f"all rows done within the period: {_fmt_pct(out['all_rows_done_in_period_frac'])} · "
          f"overlap {_fmt(out['overlap_ms_per_period_p50'])} ms/period "
          f"({_fmt_pct(out['overlap_frac_of_period_p50'], 0)}): " + overlapped)
    for label, stats in out["windows"].items():
        print(f"  window {label}: rows {stats['rows']} · {stats['periods']} windows · makespan "
              f"p50 {_fmt(stats['makespan_p50_ms'])} / p99 {_fmt(stats['makespan_p99_ms'])} / "
              f"max {_fmt(stats['makespan_max_ms'])} ms · complete {_fmt_pct(stats['all_rows_done_frac'])}")
    for name, row in out["rows"].items():
        if not row:
            continue
        slide = row.get('launch_slide_ms') or 0
        slid = f" · pacer slid {row['launch_slide_ms']:+.1f} ms" if abs(slide) > 1 else ""
        print(f"  row {name}: own period {_fmt(row['own_period_ms'])} ms · done before next trigger "
              f"{_fmt_pct(row['done_before_next_trigger_frac'])} · deadline miss (trace) "
              f"{_fmt_pct(row['deadline_miss_frac_from_trace'], 2)} · n {row['n_frames']}" + slid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir")
    parser.add_argument("period", nargs="?", type=float, default=None)
    parser.add_argument("--mix", default=None,
                        help="resolved mix json (mixes.py resolve): per-row hz/deadline -> per-rate windows "
                             "+ own-period judgement")
    parser.add_argument("--out", default=None, help="output json (default <dir>/period_makespan.json)")
    args = parser.parse_args()
    arm_dir = args.dir
    mix = read_mix(args.mix) if args.mix else {}
    out, _ = analyse(arm_dir, args.period, mix)
    json.dump(out, open(args.out or os.path.join(arm_dir, "period_makespan.json"), "w"), indent=1)
    if out["periods"] == 0 and not any(row for row in out["rows"].values()):
        print("no traces in", arm_dir, "(rows:", out["rows_missing"], ")",
              "— wrote a null period_makespan.json")
        sys.exit(1)
    print_summary(arm_dir, out)


if __name__ == "__main__":
    main()
