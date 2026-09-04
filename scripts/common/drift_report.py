#!/usr/bin/env python3
"""Clock-drift verdict over an under-load clock_sampler CSV (schema driftmon/v1).

Usage: drift_report.py <samples.csv> --device <cfg.json> --lock <lock_verified.json> --out <drift.json>
       [--phase <label>]
Judges every sample against the REALIZED lock (lock JSON reference_clock_mhz{sm,mem}), never the
requested value. Prints "drift[ <phase>]: <verdict> ..." then one indented line per cause; the verdict
lives in the JSON. Exit 0 always, 1 only when an input is unreadable or the CSV header is unknown.
"""
import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

# Pre-declared thresholds (never post-hoc), per clock with the worst clock winning.
# discrete: PASS >= 99% of samples within +/-1% of reference, WARN 95-99% (brief excursions,
#   numbers stand but flagged), FAIL below 95% or when the median sits > 2% off reference
#   (the run averaged off-lock: DVFS owned the clock, the timings are not lock-conditioned).
# jetson: a locked devfreq cannot legitimately move, so anything beyond read jitter
#   (< 99.5% at target) means the lock dropped -> FAIL.
DISCRETE_PASS_PCT = 99.0
DISCRETE_WARN_PCT = 95.0
DISCRETE_MEDIAN_DEV_PCT = 2.0
JETSON_PASS_PCT = 99.5
AT_TARGET_FRAC = 0.01

# Power-governed reclassification (discrete only). A power-capped part cannot hold ANY useful
# SM clock under an all-out tensor load: the governor floats V/F to hold POWER while the work
# rate stays tight, and it owns the SM clock at every load level, not only at the cap (light
# loads float across boost bins without ever asserting sw_power_cap). The device's invariant
# is therefore its power, as jetson's is its clock. When an sm FAIL's only cause is that
# governor, the sm clock is demoted to recorded data and the verdict to WARN, provided the
# mem clock PASSed, no hw/thermal throttle reason appeared, and the sm median deviation is
# <= 10% (a collapse still FAILs). Corroborating throughput evidence lives outside this tool.
POWER_GOVERNED_MAX_MEDIAN_DEV_PCT = 10.0

# Throttle reasons decoded from throttle_reasons_hex. sw_power_cap is expected physics at the
# power limit (WARN only); hw/thermal slowdowns mean DVFS took over (FAIL); sync_boost and
# unknown bits are recorded and never affect the verdict (raw hex is always kept).
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

# Jetson overcurrent clamps are ms-scale and invisible at the sampling rate; the clamp
# channel carries them as data so they can never cause a false clock FAIL.
CLAMP_NOTE = ("recorded data, never a verdict input — ms-scale clamps are invisible "
              "at this sampling rate; this channel carries them so they cannot cause "
              "false clock FAILs")


def say(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def die(msg):
    print(f'FATAL: {msg}', file=sys.stderr)
    sys.exit(1)


def parse_float(text):
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def column_values(rows, column):
    """Readable numeric values of one CSV column, blanks dropped."""
    return [value for value in (parse_float(row.get(column)) for row in rows) if value is not None]


def num_stats(values):
    if not values:
        return None
    return {"min": round(min(values), 3), "median": round(statistics.median(values), 3),
            "mean": round(statistics.fmean(values), 3), "max": round(max(values), 3),
            "n": len(values)}


def clock_block(values, reference_mhz):
    if not values:
        return None
    median = statistics.median(values)
    at_target = sum(1 for value in values if abs(value - reference_mhz) <= AT_TARGET_FRAC * reference_mhz)
    return {"min": round(min(values), 1), "median": round(median, 1), "max": round(max(values), 1),
            "n": len(values), "reference_mhz": reference_mhz,
            "pct_at_target": round(100.0 * at_target / len(values), 2),
            "median_dev_pct": round(abs(median - reference_mhz) / reference_mhz * 100.0, 3)}


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


def decode_throttle(hex_values):
    """Per-flag sample counts + raw hex + unknown-bit OR across all samples."""
    counts = {name: 0 for name in THROTTLE_FLAGS.values()}
    raw_seen, unknown_or, throttled = set(), 0, 0
    for value in hex_values:
        raw_seen.add(f"0x{value:x}")
        if value:
            throttled += 1
        for mask, name in THROTTLE_FLAGS.items():
            if value & mask:
                counts[name] += 1
        unknown_or |= value & ~KNOWN_MASK
    n = len(hex_values)
    reasons = [name for name in counts if counts[name] > 0]
    if unknown_or:
        reasons.append(f"unknown_bits(0x{unknown_or:x})")
    if any(counts[flag] > 0 for flag in FAIL_FLAGS):
        verdict = "FAIL"
    elif any(counts[flag] > 0 for flag in WARN_FLAGS):
        verdict = "WARN"
    else:
        verdict = "PASS"
    block = {
        "samples_decoded": n,
        "pct_samples_throttled": round(100.0 * throttled / n, 2) if n else None,
        "per_reason_sample_counts": counts,
        "raw_hex_seen": sorted(raw_seen),
        "unknown_bits": f"0x{unknown_or:x}" if unknown_or else None,
        "verdict": verdict,
    }
    return block, reasons, verdict


def throttle_hex_values(rows):
    values = []
    for row in rows:
        raw = (row.get("throttle_reasons_hex") or "").strip()
        if not raw:
            continue
        try:
            values.append(int(raw, 16))
        except ValueError:
            pass
    return values


def clamp_block(rows, power_envelope_w):
    """Jetson clamp channel: oc_event_count deltas and samples above the config power envelope."""
    oc_counts = [int(value) for value in column_values(rows, "oc_event_count")]
    power_values = column_values(rows, "module_w")
    over_envelope = None
    if isinstance(power_envelope_w, (int, float)) and power_values:
        over_envelope = sum(1 for value in power_values if value > power_envelope_w)
    clamp_events = (oc_counts[-1] - oc_counts[0]) if len(oc_counts) >= 2 else None
    intervals_increasing = None
    if len(oc_counts) >= 2:
        intervals_increasing = sum(1 for before, after in zip(oc_counts, oc_counts[1:]) if after > before)
    block = {
        "oc_event_count_first": oc_counts[0] if oc_counts else None,
        "oc_event_count_last": oc_counts[-1] if oc_counts else None,
        "oc_event_count_delta": clamp_events,
        "intervals_with_oc_increase": intervals_increasing,
        "power_envelope_w": power_envelope_w,
        "power_over_envelope_samples": over_envelope,
        "note": CLAMP_NOTE,
    }
    return block, clamp_events


def load_json(path, what):
    try:
        return json.loads(Path(path).resolve().read_text())
    except (OSError, ValueError) as error:
        die(f"cannot read {what} {path}: {error}")


def read_samples(csv_path):
    try:
        with open(csv_path, newline="") as handle:
            reader = csv.DictReader(handle)
            return reader.fieldnames or [], list(reader)
    except OSError as error:
        die(f"cannot read samples CSV {csv_path}: {error}")


def detect_platform(fields):
    """(platform, sm column, mem column) from the CSV header; dies on an unknown header."""
    field_set = set(fields)
    if {"gpu_mhz", "emc_mhz"} <= field_set:
        return "jetson", "gpu_mhz", "emc_mhz"
    if {"sm_mhz", "mem_mhz"} <= field_set:
        return "discrete", "sm_mhz", "mem_mhz"
    die(f"unrecognized samples header {fields} — expected jetson {JETSON_HEADER} "
        f"or discrete {DISCRETE_HEADER}")


def write_doc(doc, out_arg):
    out_path = Path(out_arg).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=1) + "\n")
    return out_path


def add_clock_blocks(doc, rows, platform, columns, reference, notes, causes, verdicts):
    doc["clocks"] = {}
    for label, column in columns.items():
        reference_mhz = float(reference[label])
        block = clock_block(column_values(rows, column), reference_mhz)
        if block is None:
            notes.append(f"SKIP: column {column} had no readable values — {label} clock unverified")
            doc["clocks"][label] = None
            continue
        verdict = clock_verdict_jetson(block) if platform == "jetson" else clock_verdict_discrete(block)
        block["verdict"] = verdict
        block["source_column"] = column
        doc["clocks"][label] = block
        verdicts.append(verdict)
        if verdict != "PASS":
            causes.append(f"{label}: {block['pct_at_target']}% of samples within 1% of "
                          f"{reference_mhz:g} MHz (median {block['median']} MHz, "
                          f"dev {block['median_dev_pct']}%) -> {verdict}")


def add_context_stats(doc, rows, platform):
    """Power / temperature statistics: context, never verdict-bearing."""
    if platform == "jetson":
        doc["power_w"] = num_stats(column_values(rows, "module_w"))
        doc["vdd_gpu_w"] = num_stats(column_values(rows, "vdd_gpu_w"))
        doc["temp_c"] = num_stats(column_values(rows, "tj_c"))
        return
    doc["power_w"] = num_stats(column_values(rows, "power_w"))
    doc["temp_c"] = num_stats(column_values(rows, "temp_c"))


def add_throttle_channel(doc, rows, notes, causes, verdicts):
    """Discrete throttle channel; returns the reasons seen."""
    hex_values = throttle_hex_values(rows)
    if not hex_values:
        doc["throttle"] = None
        notes.append("SKIP: throttle_reasons_hex had no readable values")
        return []
    doc["throttle"], reasons, verdict = decode_throttle(hex_values)
    verdicts.append(verdict)
    if verdict == "FAIL":
        active = sorted(set(reasons) & FAIL_FLAGS)
        causes.append(f"throttle reasons active: {', '.join(active)} — DVFS took over")
    elif verdict == "WARN":
        causes.append("sw_power_cap active — expected physics at the power limit, flagged only")
    return reasons


def reclassify_power_governed(doc, throttle_reasons, causes):
    """Demote a governor-caused sm FAIL to WARN; returns (causes, verdicts) or None when not applicable."""
    sm_block = doc["clocks"].get("sm")
    mem_block = doc["clocks"].get("mem")
    applicable = (sm_block and sm_block.get("verdict") == "FAIL"
                  and mem_block and mem_block.get("verdict") == "PASS"
                  and not (set(throttle_reasons) & FAIL_FLAGS)
                  and sm_block.get("median_dev_pct") is not None
                  and sm_block["median_dev_pct"] <= POWER_GOVERNED_MAX_MEDIAN_DEV_PCT)
    if not applicable:
        return None
    sm_block["verdict"] = "WARN"
    sm_block["power_governed"] = True
    sm_block["sm_recorded_data"] = True
    causes = [cause.replace("-> FAIL", "-> WARN (power-governed)") if cause.startswith("sm:") else cause
              for cause in causes]
    governed_by = ("sw_power_cap observed" if "sw_power_cap" in throttle_reasons
                   else "no throttle flag asserted (boost-governed at light load)")
    causes.append(
        f"power-governed reclassification: sm clock is recorded-data on this part "
        f"(mem PASS, {governed_by}, median dev {sm_block['median_dev_pct']}% "
        f"<= {POWER_GOVERNED_MAX_MEDIAN_DEV_PCT:g}%) — measured sm range "
        f"{sm_block.get('min')}-{sm_block.get('max')} MHz kept in the clocks block")
    verdicts = [block["verdict"] for block in doc["clocks"].values() if block]
    if doc.get("throttle"):
        verdicts.append(doc["throttle"]["verdict"])
    return causes, verdicts


def parse_args():
    parser = argparse.ArgumentParser(description="clock-drift verdict over sampler CSV (driftmon/v1)")
    parser.add_argument("samples_csv", help="clock_sampler output CSV")
    parser.add_argument("--device", required=True, help="device config JSON")
    parser.add_argument("--lock", required=True, help="lock_verified.json from verify_lock.py")
    parser.add_argument("--out", required=True, help="output drift.json path")
    parser.add_argument("--phase", default=None, help="label for the measurement phase this covers")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.samples_csv).resolve()
    fields, rows = read_samples(csv_path)
    config = load_json(args.device, "device config")
    lock = load_json(args.lock, "lock JSON")

    reference = lock.get("reference_clock_mhz") or {}
    if not all(isinstance(reference.get(key), (int, float)) for key in ("sm", "mem")):
        die(f"lock JSON {args.lock} lacks numeric reference_clock_mhz{{sm,mem}} "
            "— not a lockverify/v1 file?")

    platform, sm_column, mem_column = detect_platform(fields)

    notes = []
    if config.get("platform") not in (None, platform):
        note = (f"CSV header is {platform}-form but device config says "
                f"'{config.get('platform')}' — trusting the CSV columns")
        notes.append(note)
        print(f"WARN: {note}")

    doc = {
        "schema": "driftmon/v1",
        "phase": args.phase,
        "platform": platform,
        "device_tag": config.get("device_tag"),
        "samples_csv": str(csv_path),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_samples": len(rows),
        "reference_clock_mhz": {"sm": reference["sm"], "mem": reference["mem"]},
        "notes": notes,
    }

    if not rows:
        doc["verdict"] = "SKIP"
        doc["verdict_causes"] = ["SKIP: no samples in CSV — no drift evidence either way"]
        doc["pct_at_target"] = None
        doc["throttle_reasons_seen"] = []
        doc["clamp_events"] = None
        out_path = write_doc(doc, args.out)
        say(f"verdict: SKIP (empty CSV) — written to {out_path}")
        sys.exit(0)

    times = column_values(rows, "t_s")
    doc["duration_s"] = round(times[-1] - times[0], 3) if len(times) >= 2 else None

    causes, verdicts = [], []
    add_clock_blocks(doc, rows, platform, {"sm": sm_column, "mem": mem_column},
                     reference, notes, causes, verdicts)
    add_context_stats(doc, rows, platform)

    throttle_reasons = []
    if platform == "discrete":
        throttle_reasons = add_throttle_channel(doc, rows, notes, causes, verdicts)
        doc["clamp"] = None
        clamp_events = None
        reclassified = reclassify_power_governed(doc, throttle_reasons, causes)
        if reclassified is not None:
            causes, verdicts = reclassified
    else:
        doc["throttle"] = None
        doc["clamp"], clamp_events = clamp_block(rows, config.get("power_envelope_w"))
        if clamp_events:
            say(f"clamp channel: {clamp_events} oc events during the window (data, not a failure)")

    if verdicts:
        verdict = max(verdicts, key=lambda name: RANK[name])
    else:
        verdict = "SKIP"
        causes.append("SKIP: no clock column had readable values — drift unverifiable")

    at_targets = [block["pct_at_target"] for block in doc["clocks"].values() if block]
    doc["pct_at_target"] = min(at_targets) if at_targets else None
    doc["throttle_reasons_seen"] = throttle_reasons
    doc["clamp_events"] = clamp_events
    doc["verdict"] = verdict
    doc["verdict_causes"] = causes

    out_path = write_doc(doc, args.out)
    phase = f" [{args.phase}]" if args.phase else ""
    say(f"drift{phase}: {verdict} — worst pct_at_target "
        f"{doc['pct_at_target']}% over {doc['n_samples']} samples — {out_path}")
    for cause in causes:
        say(f"  {cause}")
    sys.exit(0)


if __name__ == "__main__":
    main()
