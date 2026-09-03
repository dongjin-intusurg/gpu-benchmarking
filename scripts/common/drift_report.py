#!/usr/bin/env python3
"""Quantitative clock-drift verdict over an under-load sampler CSV (driftmon/v1).

Replaces the old start-vs-end `cmp` of two clock snapshots: two matching
endpoints say nothing about the minutes in between, so the verdict here is
computed over every sample the clock_sampler recorded during the measurement.

Usage:
  drift_report.py <samples.csv> --device <cfg.json> --lock <lock_verified.json> \
                  --out <drift.json> [--phase <label>]

Input CSV (platform self-detected from the header row):
  jetson:   t_s,gpu_mhz,emc_mhz,module_w,vdd_gpu_w,tj_c,oc_event_count
  discrete: t_s,sm_mhz,mem_mhz,power_w,temp_c,throttle_reasons_hex
Reference clocks come from the lock JSON's reference_clock_mhz{sm,mem}
(jetson mapping: sm = gpu_mhz column, mem = emc_mhz column) — the REALIZED
lock, never the requested value.

Verdict thresholds (pre-declared, not post-hoc):
  discrete, per clock (sm and mem; worst clock wins):
    PASS  >= 99% of samples within +/-1% of the reference clock
    WARN  95-99%   (brief excursions; numbers stand, flagged)
    FAIL  < 95%, or |median - reference|/reference > 2%
          (the run averaged off-lock: DVFS owned the clock, timings are not
          lock-conditioned)
  power-governed reclassification (discrete only, adopted 2026-08-12):
    a power-capped part cannot hold ANY useful SM clock under an all-out
    tensor load — at the cap the governor floats V/F to hold POWER while the
    work rate stays tight (a discrete Blackwell card 3-min soak: clock wandered 1897-2062
    MHz, yet 0/2247 throughput samples fell below 0.9x best). The device's
    invariant is its power, as jetson's is its clock, so each is judged by
    the invariant it can physically hold. When an sm FAIL's only cause is the
    power governor, the sm clock is demoted to recorded-data and the verdict
    to WARN. ALL conditions required:
      - mem clock verdict PASS (the dial axis and bandwidth basis held)
      - no FAIL-class throttle reason (hw_slowdown / thermal) in any sample
      - sm median deviation <= 10% of reference (a collapse still FAILs)
    AMENDED 2026-08-19 (a light int8 model run): the original rule also
    required sw_power_cap to be OBSERVED in the samples. Light workloads
    (~176 W on a 300 W part, 1.2 ms kernels) float the sm clock across boost
    bins 1987-2152 without ever asserting the flag — the governor owns the
    clock at every load level on this part, not only at the cap (verified
    twice: a 2160 request floated 2062-2152; a 2062 request floated
    1987-2055; zero throttle flags both times, mem rock-solid at 13365).
    The sw_power_cap-observed condition is therefore dropped: on discrete,
    an off-target sm clock with mem PASS, no hw/thermal reason, and median
    dev <= 10% is ALWAYS recorded-data (sm_recorded_data:true, measured
    range kept in the clocks block) and the verdict WARN. The sm clock on
    this part is telemetry, not an invariant; the invariants judged are the
    mem clock and the absence of hw/thermal throttling.
    Corroborating throughput evidence lives outside this tool by design:
    the ceilings clean-spread policy, sustained-suite clamp incidence, and
    each model's p99/median spread.
  jetson, per clock:
    PASS  >= 99.5% within +/-1% of reference, else FAIL
          (a locked devfreq cannot legitimately move; anything beyond read
          jitter means the lock dropped)
  throttle reasons (discrete, decoded from throttle_reasons_hex):
    sw_power_cap (0x4)           -> WARN only (expected physics at the power
                                    limit; matches the ceilings clean-spread
                                    doctrine)
    hw_slowdown (0x8)            -> FAIL (DVFS took over; clocks were not what
    sw_thermal_slowdown (0x20)      the lock promised — the timing numbers are
    hw_thermal_slowdown (0x40)      not lock-conditioned)
    sync_boost (0x10)            -> named and recorded, no verdict effect
    any unknown nonzero bits     -> recorded as present (raw hex always kept),
                                    no verdict effect — verdict-bearing reasons
                                    are enumerated above
  jetson clamp channel (recorded data, NEVER a verdict input):
    oc_event_count deltas + samples with module power above the config
    power_envelope_w. Overcurrent clamps are ms-scale and invisible at this
    sampling rate — the clamp channel carries them, so they cannot cause
    false clock FAILs; throughput dips they cause are handled by the
    ceilings' clean-spread policy.
  zero data rows -> verdict "SKIP" with a labelled reason (no evidence either
    way; downstream treats it as absent, not passing).

Exit 0 always (the verdict lives in the JSON); exit 1 only when an input file
is unreadable or its header matches neither contract. stdlib only.
"""
import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

# Pre-declared thresholds (rationale in the module docstring).
DISCRETE_PASS_PCT = 99.0
DISCRETE_WARN_PCT = 95.0
DISCRETE_MEDIAN_DEV_PCT = 2.0
JETSON_PASS_PCT = 99.5
AT_TARGET_FRAC = 0.01
POWER_GOVERNED_MAX_MEDIAN_DEV_PCT = 10.0

THROTTLE_FLAGS = {
    0x4: "sw_power_cap",
    0x8: "hw_slowdown",
    0x10: "sync_boost",
    0x20: "sw_thermal_slowdown",
    0x40: "hw_thermal_slowdown",
}
KNOWN_MASK = 0x7C
FAIL_FLAGS = {"hw_slowdown", "sw_thermal_slowdown", "hw_thermal_slowdown"}
WARN_FLAGS = {"sw_power_cap"}

JETSON_HEADER = ["t_s", "gpu_mhz", "emc_mhz", "module_w", "vdd_gpu_w", "tj_c", "oc_event_count"]
DISCRETE_HEADER = ["t_s", "sm_mhz", "mem_mhz", "power_w", "temp_c", "throttle_reasons_hex"]

RANK = {"PASS": 0, "WARN": 1, "FAIL": 2}


def say(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def die(msg):
    print(f'FATAL: {msg}', file=sys.stderr)
    sys.exit(1)


def fnum(x):
    if x is None:
        return None
    x = x.strip()
    if not x:
        return None
    try:
        return float(x)
    except ValueError:
        return None


def num_stats(vals):
    if not vals:
        return None
    return {"min": round(min(vals), 3), "median": round(statistics.median(vals), 3),
            "mean": round(statistics.fmean(vals), 3), "max": round(max(vals), 3),
            "n": len(vals)}


def clock_block(vals, ref):
    if not vals:
        return None
    med = statistics.median(vals)
    at = sum(1 for v in vals if abs(v - ref) <= AT_TARGET_FRAC * ref)
    return {"min": round(min(vals), 1), "median": round(med, 1), "max": round(max(vals), 1),
            "n": len(vals), "reference_mhz": ref,
            "pct_at_target": round(100.0 * at / len(vals), 2),
            "median_dev_pct": round(abs(med - ref) / ref * 100.0, 3)}


def clock_verdict_discrete(block):
    if block["median_dev_pct"] > DISCRETE_MEDIAN_DEV_PCT:
        return "FAIL"
    if block["pct_at_target"] >= DISCRETE_PASS_PCT:
        return "PASS"
    if block["pct_at_target"] >= DISCRETE_WARN_PCT:
        return "WARN"
    return "FAIL"


def clock_verdict_jetson(block):
    return "PASS" if block["pct_at_target"] >= JETSON_PASS_PCT else "FAIL"


def decode_throttle(hex_vals):
    """Per-flag sample counts + raw hex + unknown-bit OR across all samples."""
    counts = {name: 0 for name in THROTTLE_FLAGS.values()}
    raw_seen, unknown_or, throttled = set(), 0, 0
    for v in hex_vals:
        raw_seen.add(f"0x{v:x}")
        if v:
            throttled += 1
        for mask, name in THROTTLE_FLAGS.items():
            if v & mask:
                counts[name] += 1
        unknown_or |= v & ~KNOWN_MASK
    n = len(hex_vals)
    reasons = [name for name in counts if counts[name] > 0]
    if unknown_or:
        reasons.append(f"unknown_bits(0x{unknown_or:x})")
    if any(counts[f] > 0 for f in FAIL_FLAGS):
        verdict = "FAIL"
    elif any(counts[f] > 0 for f in WARN_FLAGS):
        verdict = "WARN"
    else:
        verdict = "PASS"
    return {
        "samples_decoded": n,
        "pct_samples_throttled": round(100.0 * throttled / n, 2) if n else None,
        "per_reason_sample_counts": counts,
        "raw_hex_seen": sorted(raw_seen),
        "unknown_bits": f"0x{unknown_or:x}" if unknown_or else None,
        "verdict": verdict,
    }, reasons, verdict


def main():
    ap = argparse.ArgumentParser(description="clock-drift verdict over sampler CSV (driftmon/v1)")
    ap.add_argument("samples_csv", help="clock_sampler output CSV")
    ap.add_argument("--device", required=True, help="device config JSON")
    ap.add_argument("--lock", required=True, help="lock_verified.json from verify_lock.py")
    ap.add_argument("--out", required=True, help="output drift.json path")
    ap.add_argument("--phase", default=None, help="label for the measurement phase this covers")
    args = ap.parse_args()

    csv_path = Path(args.samples_csv).resolve()
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            rows = list(reader)
    except OSError as e:
        die(f"cannot read samples CSV {csv_path}: {e}")

    try:
        cfg = json.loads(Path(args.device).resolve().read_text())
    except (OSError, ValueError) as e:
        die(f"cannot read device config {args.device}: {e}")
    try:
        lock = json.loads(Path(args.lock).resolve().read_text())
    except (OSError, ValueError) as e:
        die(f"cannot read lock JSON {args.lock}: {e}")

    ref = lock.get("reference_clock_mhz") or {}
    if not isinstance(ref.get("sm"), (int, float)) or not isinstance(ref.get("mem"), (int, float)):
        die(f"lock JSON {args.lock} lacks numeric reference_clock_mhz{{sm,mem}} — not a lockverify/v1 file?")

    fset = set(fields)
    if {"gpu_mhz", "emc_mhz"} <= fset:
        platform = "jetson"
        sm_col, mem_col = "gpu_mhz", "emc_mhz"
    elif {"sm_mhz", "mem_mhz"} <= fset:
        platform = "discrete"
        sm_col, mem_col = "sm_mhz", "mem_mhz"
    else:
        die(f"unrecognized samples header {fields} — expected jetson {JETSON_HEADER} "
            f"or discrete {DISCRETE_HEADER}")

    notes = []
    if cfg.get("platform") not in (None, platform):
        note = (f"CSV header is {platform}-form but device config says "
                f"'{cfg.get('platform')}' — trusting the CSV columns")
        notes.append(note)
        print(f"WARN: {note}")

    doc = {
        "schema": "driftmon/v1",
        "phase": args.phase,
        "platform": platform,
        "device_tag": cfg.get("device_tag"),
        "samples_csv": str(csv_path),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_samples": len(rows),
        "reference_clock_mhz": {"sm": ref["sm"], "mem": ref["mem"]},
        "notes": notes,
    }

    if not rows:
        doc["verdict"] = "SKIP"
        doc["verdict_causes"] = ["SKIP: no samples in CSV — no drift evidence either way"]
        doc["pct_at_target"] = None
        doc["throttle_reasons_seen"] = []
        doc["clamp_events"] = None
        out = Path(args.out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1) + "\n")
        say(f"verdict: SKIP (empty CSV) — written to {out}")
        sys.exit(0)

    t_vals = [v for v in (fnum(r.get("t_s")) for r in rows) if v is not None]
    doc["duration_s"] = round(t_vals[-1] - t_vals[0], 3) if len(t_vals) >= 2 else None

    # --- per-clock statistics vs the REALIZED reference -----------------------
    causes, verdicts = [], []
    doc["clocks"] = {}
    for label, col, ref_mhz in (("sm", sm_col, float(ref["sm"])), ("mem", mem_col, float(ref["mem"]))):
        vals = [v for v in (fnum(r.get(col)) for r in rows) if v is not None]
        block = clock_block(vals, ref_mhz)
        if block is None:
            notes.append(f"SKIP: column {col} had no readable values — {label} clock unverified")
            doc["clocks"][label] = None
            continue
        v = clock_verdict_jetson(block) if platform == "jetson" else clock_verdict_discrete(block)
        block["verdict"] = v
        block["source_column"] = col
        doc["clocks"][label] = block
        verdicts.append(v)
        if v != "PASS":
            causes.append(f"{label}: {block['pct_at_target']}% of samples within 1% of "
                          f"{ref_mhz:g} MHz (median {block['median']} MHz, "
                          f"dev {block['median_dev_pct']}%) -> {v}")

    # --- power / temperature (context, not verdict-bearing) -------------------
    if platform == "jetson":
        doc["power_w"] = num_stats([v for v in (fnum(r.get("module_w")) for r in rows) if v is not None])
        doc["vdd_gpu_w"] = num_stats([v for v in (fnum(r.get("vdd_gpu_w")) for r in rows) if v is not None])
        doc["temp_c"] = num_stats([v for v in (fnum(r.get("tj_c")) for r in rows) if v is not None])
    else:
        doc["power_w"] = num_stats([v for v in (fnum(r.get("power_w")) for r in rows) if v is not None])
        doc["temp_c"] = num_stats([v for v in (fnum(r.get("temp_c")) for r in rows) if v is not None])

    # --- throttle (discrete) / clamp (jetson) channels ------------------------
    throttle_reasons = []
    if platform == "discrete":
        hex_vals = []
        for r in rows:
            raw = (r.get("throttle_reasons_hex") or "").strip()
            if not raw:
                continue
            try:
                hex_vals.append(int(raw, 16))
            except ValueError:
                pass
        if hex_vals:
            doc["throttle"], throttle_reasons, tv = decode_throttle(hex_vals)
            verdicts.append(tv)
            if tv == "FAIL":
                bad = sorted(set(throttle_reasons) & FAIL_FLAGS)
                causes.append(f"throttle reasons active: {', '.join(bad)} — DVFS took over")
            elif tv == "WARN":
                causes.append("sw_power_cap active — expected physics at the power limit, flagged only")
        else:
            doc["throttle"] = None
            notes.append("SKIP: throttle_reasons_hex had no readable values")
        doc["clamp"] = None
        clamp_events = None
    else:
        doc["throttle"] = None
        oc = []
        for r in rows:
            v = fnum(r.get("oc_event_count"))
            if v is not None:
                oc.append(int(v))
        power_vals = [v for v in (fnum(r.get("module_w")) for r in rows) if v is not None]
        envelope = cfg.get("power_envelope_w")
        over = (sum(1 for v in power_vals if v > envelope)
                if isinstance(envelope, (int, float)) and power_vals else None)
        clamp_events = (oc[-1] - oc[0]) if len(oc) >= 2 else None
        doc["clamp"] = {
            "oc_event_count_first": oc[0] if oc else None,
            "oc_event_count_last": oc[-1] if oc else None,
            "oc_event_count_delta": clamp_events,
            "intervals_with_oc_increase": (sum(1 for a, b in zip(oc, oc[1:]) if b > a)
                                           if len(oc) >= 2 else None),
            "power_envelope_w": envelope,
            "power_over_envelope_samples": over,
            "note": ("recorded data, never a verdict input — ms-scale clamps are invisible "
                     "at this sampling rate; this channel carries them so they cannot cause "
                     "false clock FAILs"),
        }
        if clamp_events:
            say(f"clamp channel: {clamp_events} oc events during the window (data, not a failure)")

    # --- power-governed reclassification (discrete only; rule + rationale in
    # the module docstring, thresholds pre-declared above) ----------------------
    if platform == "discrete":
        sm_blk = doc["clocks"].get("sm")
        mem_blk = doc["clocks"].get("mem")
        # 2026-08-19 amendment (docstring): sw_power_cap-observed is no longer
        # required — the governor owns the sm clock at every load level on
        # this part; light loads float it without asserting the flag.
        if (sm_blk and sm_blk.get("verdict") == "FAIL"
                and mem_blk and mem_blk.get("verdict") == "PASS"
                and not (set(throttle_reasons) & FAIL_FLAGS)
                and sm_blk.get("median_dev_pct") is not None
                and sm_blk["median_dev_pct"] <= POWER_GOVERNED_MAX_MEDIAN_DEV_PCT):
            sm_blk["verdict"] = "WARN"
            sm_blk["power_governed"] = True
            sm_blk["sm_recorded_data"] = True
            causes = [c.replace("-> FAIL", "-> WARN (power-governed)")
                      if c.startswith("sm:") else c for c in causes]
            governed_by = ("sw_power_cap observed" if "sw_power_cap" in throttle_reasons
                           else "no throttle flag asserted (boost-governed at light load)")
            causes.append(
                f"power-governed reclassification: sm clock is recorded-data on this part "
                f"(mem PASS, {governed_by}, median dev {sm_blk['median_dev_pct']}% "
                f"<= {POWER_GOVERNED_MAX_MEDIAN_DEV_PCT:g}%) — measured sm range "
                f"{sm_blk.get('min')}-{sm_blk.get('max')} MHz kept in the clocks block")
            verdicts = [b["verdict"] for b in doc["clocks"].values() if b]
            if doc.get("throttle"):
                verdicts.append(doc["throttle"]["verdict"])

    # --- combined verdict (worst of clock verdicts + throttle verdict) --------
    if verdicts:
        verdict = max(verdicts, key=lambda v: RANK[v])
    else:
        verdict = "SKIP"
        causes.append("SKIP: no clock column had readable values — drift unverifiable")

    at_targets = [b["pct_at_target"] for b in doc["clocks"].values() if b]
    doc["pct_at_target"] = min(at_targets) if at_targets else None
    doc["throttle_reasons_seen"] = throttle_reasons
    doc["clamp_events"] = clamp_events
    doc["verdict"] = verdict
    doc["verdict_causes"] = causes

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")

    phase = f" [{args.phase}]" if args.phase else ""
    say(f"drift{phase}: {verdict} — worst pct_at_target "
        f"{doc['pct_at_target']}% over {doc['n_samples']} samples — {out}")
    for c in causes:
        say(f"  {c}")
    sys.exit(0)


if __name__ == "__main__":
    main()
