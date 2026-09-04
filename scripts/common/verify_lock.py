#!/usr/bin/env python3
"""Post-lock clock verification: prove the lock actually took BEFORE measuring (schema lockverify/v1).

Usage: verify_lock.py --device <cfg.json> --out <lock_verified.json> [--requested sm=<mhz>,mem=<mhz>]
Always writes reference_clock_mhz{sm,mem} - the REALIZED lock drift_report.py judges against, never
the requested value. Prints "reference_clock_mhz: ..." and "verdict: <PASS|WARN|FAIL> ...".
Exit 0 on PASS/WARN, 1 on FAIL (re-lock remediation on stderr) or unreadable input. stdlib only;
torch runs only inside the discrete load subprocess.
"""
import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

# Pre-declared thresholds, never post-hoc.
# jetson: devfreq min == max == cur (exact Hz) equal to the config lock_targets_mhz within 1 MHz
#   (Hz->MHz rounding guard only) is authoritative, unprivileged and needs no load; anything else
#   FAILs. --requested is ignored (targets live in the device config).
# discrete: an idle read-back proves nothing (idle cards legitimately downclock even when locked,
#   and -lgc/-lmc have silently failed while the log claimed success), so the verdict comes from
#   a 5 s fp16 GEMM load: 2 s warm, 3 s sampled at 10 Hz.
#   FAIL  mem median off requested by > 1% (memory locks exactly or not at all - no boost table)
#   FAIL  sm median < 0.75 x requested (silent-lock-failure signature: clocks never left idle)
#   FAIL  sm p95 - p5 > 30 MHz (a locked SM is flat under load; wander means DVFS owns it)
#   WARN  sm median in [0.75, 0.97) x requested: legitimate boost-table realization; proceed with
#         reference_clock_mhz.sm := realized median so drift is judged against what the silicon holds
#   PASS  otherwise; reference := requested.
WARM_S = 2.0
SAMPLE_S = 3.0
RATE_HZ = 10
LOAD_S = WARM_S + SAMPLE_S
GEMM_N = 2048
MEM_TOL_FRAC = 0.01
SM_FLOOR_FRAC = 0.75
SM_BOOST_OK_FRAC = 0.97
SM_FLAT_MHZ = 30.0
JETSON_TOL_MHZ = 1.0

# A torch build without the card's SM architecture fails the GEMM with cuBLAS INVALID_VALUE.
CU13X_REMEDIATION = ("upgrade torch to a cu13x build: pip install torch "
                     "--index-url https://download.pytorch.org/whl/cu130 "
                     "(add --break-system-packages if PEP 668 blocks), then re-run")

# Child prints LOAD_RUNNING only after the first GEMM completed, so the parent's
# warm/sample window starts once the GPU is provably under tensor load.
LOAD_SRC = r'''
import sys, time
try:
    import torch
except Exception as e:
    print("TORCH_IMPORT_FAIL: %s" % e, file=sys.stderr)
    sys.exit(3)
try:
    assert torch.cuda.is_available(), "cuda not available in torch"
    n = int(sys.argv[2])
    a = torch.randn(n, n, device="cuda", dtype=torch.half)
    b = torch.randn(n, n, device="cuda", dtype=torch.half)
    (a @ b).sum().item()
except Exception as e:
    print("TORCH_GEMM_FAIL: %s" % e, file=sys.stderr)
    sys.exit(4)
print("LOAD_RUNNING", flush=True)
t_end = time.time() + float(sys.argv[1])
while time.time() < t_end:
    b = a @ b
torch.cuda.synchronize()
print("LOAD_DONE", flush=True)
'''

JETSON_REMEDIATION = ("re-lock: sudo nvpmodel -m 0 (MAXN) if needed, then sudo jetson_clocks; "
                      "devfreq min==max==cur is the authoritative check — re-run this script after")


def say(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def die(msg):
    print(f'FATAL: {msg}', file=sys.stderr)
    sys.exit(1)


def write_json(path, doc):
    out_path = Path(path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=1) + "\n")
    return out_path


def smi_query(fields):
    """One nvidia-smi CSV query row as a list of strings, None on failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return [value.strip() for value in result.stdout.strip().splitlines()[0].split(",")]


def percentile(sorted_values, q):
    index = int(round(q * (len(sorted_values) - 1)))
    return sorted_values[max(0, min(len(sorted_values) - 1, index))]


def read_devfreq_node(node):
    """(raw Hz, MHz) dicts of min/max/cur_freq; None entries where the attr is unreadable."""
    scale = float(node.get("scale_hz_to_mhz", 1e-6))
    directory = Path(node["dir"])
    raw_hz = {}
    for attr in ("min_freq", "max_freq", "cur_freq"):
        try:
            raw_hz[attr] = int((directory / attr).read_text().strip())
        except (OSError, ValueError):
            raw_hz[attr] = None
    mhz = {attr: (None if value is None else round(value * scale, 3)) for attr, value in raw_hz.items()}
    return directory, raw_hz, mhz


def verify_jetson(config, doc):
    clock_sources = config["clock_sources"]
    targets = config["lock_targets_mhz"]
    readings, causes = {}, []
    for name, key in (("gpu", "gpu_mhz"), ("emc", "emc_mhz")):
        directory, raw_hz, mhz = read_devfreq_node(clock_sources[key])
        target = float(targets[name])
        locked = (None not in raw_hz.values()
                  and raw_hz["min_freq"] == raw_hz["max_freq"] == raw_hz["cur_freq"]
                  and abs(mhz["cur_freq"] - target) < JETSON_TOL_MHZ)
        if not locked:
            causes.append(f"{name}: min/max/cur = {mhz['min_freq']}/{mhz['max_freq']}/"
                          f"{mhz['cur_freq']} MHz vs target {target:g} MHz")
        readings[name] = {"dir": str(directory), "raw_hz": raw_hz, "mhz": mhz,
                          "target_mhz": target, "locked": locked}

    doc["jetson"] = readings
    doc["reference_clock_mhz"] = {"sm": float(targets["gpu"]), "mem": float(targets["emc"])}
    doc["requested_clock_mhz"] = dict(doc["reference_clock_mhz"])
    doc["verdict"] = "PASS" if not causes else "FAIL"
    doc["verdict_causes"] = causes
    return causes


def parse_requested_arg(requested_arg):
    requested = {"sm": None, "mem": None}
    if not requested_arg:
        return requested
    for part in requested_arg.split(","):
        if "=" not in part:
            die(f"bad --requested fragment '{part}' (want sm=<mhz>,mem=<mhz>)")
        key, value = part.split("=", 1)
        key = key.strip()
        if key not in requested:
            die(f"bad --requested key '{key}' (want sm/mem)")
        requested[key] = float(value)
    return requested


def resolve_requested(config, requested_arg):
    """Requested sm/mem MHz: --requested, else config mem_lock_mhz (the memory bin that actually
    holds under load outranks the nameplate target - requesting an unholdable bin produces an
    honest-but-useless FAIL), else lock_targets_mhz, which may name an nvidia-smi field."""
    requested = parse_requested_arg(requested_arg)
    targets = config.get("lock_targets_mhz") or {}
    if requested["mem"] is None and isinstance(config.get("mem_lock_mhz"), (int, float)):
        requested["mem"] = float(config["mem_lock_mhz"])
    pending = []
    for key in ("sm", "mem"):
        if requested[key] is not None:
            continue
        target = targets.get(key)
        if isinstance(target, (int, float)):
            requested[key] = float(target)
        elif isinstance(target, str):
            pending.append((key, target))
        else:
            die(f"no requested {key} clock: pass --requested or set lock_targets_mhz.{key}")
    if pending:
        values = smi_query(",".join(field for _, field in pending))
        if values is None or len(values) < len(pending):
            die("nvidia-smi query for max clocks failed — pass --requested sm=<mhz>,mem=<mhz>")
        for (key, _), value in zip(pending, values):
            try:
                requested[key] = float(value)
            except ValueError:
                die(f"nvidia-smi returned non-numeric max clock for {key}: '{value}'")
    return requested


def make_clock_reader():
    """(read_fn -> (sm_mhz, mem_mhz) or None, backend name); pynvml preferred, nvidia-smi fallback."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        def read_pynvml():
            return (float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)),
                    float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)))
        return read_pynvml, "pynvml"
    except Exception:
        def read_smi():
            values = smi_query("clocks.sm,clocks.mem")
            if values is None or len(values) < 2:
                return None
            try:
                return (float(values[0]), float(values[1]))
            except ValueError:
                return None
        return read_smi, "nvidia-smi"


def start_load_subprocess():
    """Launch the GEMM load and return it once LOAD_RUNNING arrived; dies with a remediation otherwise."""
    say(f"launching {LOAD_S:.0f} s fp16 GEMM n={GEMM_N} load subprocess")
    proc = subprocess.Popen([sys.executable, "-c", LOAD_SRC, str(LOAD_S), str(GEMM_N)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    line = (proc.stdout.readline() or "").strip()
    if line == "LOAD_RUNNING":
        return proc
    try:
        _, err = proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        _, err = proc.communicate()
    err = (err or "").strip()
    if proc.returncode == 3 or "TORCH_IMPORT_FAIL" in err:
        die(f"torch unavailable for the GEMM load — {CU13X_REMEDIATION}")
    if proc.returncode == 4 or "TORCH_GEMM_FAIL" in err:
        die(f"torch GEMM failed (sm_120 cuBLAS INVALID_VALUE signature) — {CU13X_REMEDIATION}")
    die(f"load subprocess failed before running (rc={proc.returncode}): {err[:400]}")


def sample_under_load(read_clocks, proc, notes):
    """(sm values, mem values) sampled at RATE_HZ after the warm window, never past the load window."""
    t_load = time.monotonic()
    time.sleep(WARM_S)
    sm_values, mem_values = [], []
    t0 = time.monotonic()
    for i in range(int(SAMPLE_S * RATE_HZ)):
        target_time = t0 + i / RATE_HZ
        now = time.monotonic()
        # trailing idle reads would fake wander
        if now > t_load + LOAD_S - 0.05:
            notes.append(f"sampling stopped early at {len(sm_values)} samples (load window expired)")
            break
        if target_time > now:
            time.sleep(target_time - now)
        try:
            clocks = read_clocks()
        except Exception:
            clocks = None
        if clocks is not None:
            sm_values.append(clocks[0])
            mem_values.append(clocks[1])
    try:
        proc.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        notes.append("load subprocess had to be killed after the sampling window")
    if proc.returncode not in (0, None):
        notes.append(f"load subprocess exited rc={proc.returncode} — samples may include idle time")
    return sm_values, mem_values


def judge_discrete(doc, requested, sm_values, mem_values, notes):
    """Apply the discrete thresholds; fills doc realized/checks/reference/verdict, returns causes."""
    sm_sorted = sorted(sm_values)
    sm_median = statistics.median(sm_values)
    sm_p5 = percentile(sm_sorted, 0.05)
    sm_p95 = percentile(sm_sorted, 0.95)
    mem_median = statistics.median(mem_values)
    doc["discrete"]["realized"] = {
        "sm": {"median": round(sm_median, 1), "p5": round(sm_p5, 1), "p95": round(sm_p95, 1),
               "min": round(min(sm_values), 1), "max": round(max(sm_values), 1), "n": len(sm_values)},
        "mem": {"median": round(mem_median, 1), "min": round(min(mem_values), 1),
                "max": round(max(mem_values), 1), "n": len(mem_values)},
    }

    causes, checks = [], []
    mem_dev = abs(mem_median - requested["mem"]) / requested["mem"]
    mem_ok = mem_dev <= MEM_TOL_FRAC
    checks.append({"name": "mem_median_within_1pct_of_requested", "limit_pct": 1.0,
                   "value_pct": round(mem_dev * 100, 3), "pass": mem_ok})
    if not mem_ok:
        causes.append(f"mem median {mem_median:.0f} MHz is {mem_dev * 100:.1f}% off "
                      f"requested {requested['mem']:.0f} MHz (limit 1%)")

    ratio = sm_median / requested["sm"]
    floor_ok = ratio >= SM_FLOOR_FRAC
    checks.append({"name": "sm_median_floor", "limit_ratio": SM_FLOOR_FRAC,
                   "value_ratio": round(ratio, 4), "pass": floor_ok})
    if not floor_ok:
        causes.append(f"sm median {sm_median:.0f} MHz is {ratio:.2f}x requested "
                      f"{requested['sm']:.0f} MHz (floor {SM_FLOOR_FRAC}x) "
                      "— the silent-lock-failure signature")

    flatness = sm_p95 - sm_p5
    flat_ok = flatness <= SM_FLAT_MHZ
    checks.append({"name": "sm_flatness_p95_minus_p5", "limit_mhz": SM_FLAT_MHZ,
                   "value_mhz": round(flatness, 1), "pass": flat_ok})
    if not flat_ok:
        causes.append(f"sm p95-p5 = {flatness:.0f} MHz under load (limit {SM_FLAT_MHZ:.0f}) "
                      "— DVFS owns the clock")

    boost_warn = not causes and floor_ok and ratio < SM_BOOST_OK_FRAC
    checks.append({"name": "sm_boost_table_band", "band_ratio": [SM_FLOOR_FRAC, SM_BOOST_OK_FRAC],
                   "value_ratio": round(ratio, 4), "in_band": boost_warn})
    doc["checks"] = checks

    if causes:
        verdict, reference_sm = "FAIL", requested["sm"]
    elif boost_warn:
        verdict, reference_sm = "WARN", round(sm_median, 1)
        msg = (f"sm realized {sm_median:.0f} MHz for a {requested['sm']:.0f} MHz request "
               f"({ratio:.2f}x) — boost-table realization; reference := realized")
        notes.append(msg)
        print(f"WARN: {msg}")
    else:
        verdict, reference_sm = "PASS", requested["sm"]

    doc["reference_clock_mhz"] = {"sm": reference_sm, "mem": requested["mem"]}
    doc["verdict"] = verdict
    doc["verdict_causes"] = causes
    return causes


def verify_discrete(config, requested_arg, doc):
    requested = resolve_requested(config, requested_arg)
    doc["requested_clock_mhz"] = {"sm": requested["sm"], "mem": requested["mem"]}
    notes = doc.setdefault("notes", [])

    read_clocks, backend = make_clock_reader()
    if backend == "nvidia-smi":
        notes.append("pynvml unavailable — nvidia-smi subprocess sampling "
                     "(effective rate may be below 10 Hz)")

    idle = read_clocks()
    idle_block = None if idle is None else {"sm": idle[0], "mem": idle[1]}
    say(f"idle read-back (informational only): {idle_block}")

    proc = start_load_subprocess()
    sm_values, mem_values = sample_under_load(read_clocks, proc, notes)

    doc["discrete"] = {
        "idle_readback_mhz": idle_block,
        "idle_readback_note": ("informational only — idle discrete GPUs legitimately downclock "
                               "even when locked"),
        "sampler": backend,
        "load": {"kind": "torch-fp16-gemm", "n": GEMM_N, "duration_s": LOAD_S,
                 "warm_s": WARM_S, "sample_s": SAMPLE_S, "rate_hz": RATE_HZ},
    }
    if not sm_values or not mem_values:
        doc["reference_clock_mhz"] = {"sm": requested["sm"], "mem": requested["mem"]}
        doc["verdict"] = "FAIL"
        doc["verdict_causes"] = ["no clock samples collected under load — sampler broken"]
        return doc["verdict_causes"]
    return judge_discrete(doc, requested, sm_values, mem_values, notes)


def discrete_remediation(doc):
    requested = doc.get("requested_clock_mhz", {})
    sm = f"{requested['sm']:.0f}" if isinstance(requested.get("sm"), (int, float)) else "<mhz>"
    mem = f"{requested['mem']:.0f}" if isinstance(requested.get("mem"), (int, float)) else "<mhz>"
    return ("re-lock and re-verify before measuring:\n"
            "  sudo nvidia-smi -pm 1\n"
            f"  sudo nvidia-smi -lgc {sm}\n"
            f"  sudo nvidia-smi -lmc {mem}\n"
            "idle read-back proves nothing on a discrete GPU — only this under-load check does")


def parse_args():
    parser = argparse.ArgumentParser(description="verify a GPU clock lock actually took (lockverify/v1)")
    parser.add_argument("--device", required=True, help="device config JSON")
    parser.add_argument("--out", required=True, help="output lock_verified.json path")
    parser.add_argument("--requested", default=None, help="sm=<mhz>,mem=<mhz> override (discrete only)")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        config = json.loads(Path(args.device).resolve().read_text())
    except (OSError, ValueError) as error:
        die(f"cannot read device config {args.device}: {error}")

    platform = config.get("platform")
    doc = {
        "schema": "lockverify/v1",
        "device_tag": config.get("device_tag"),
        "device_id": config.get("device_id"),
        "platform": platform,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "notes": [],
    }

    if platform == "jetson":
        if args.requested:
            doc["notes"].append("--requested ignored on jetson (targets come from the device config)")
        causes = verify_jetson(config, doc)
        remediation = JETSON_REMEDIATION
    elif platform == "discrete":
        causes = verify_discrete(config, args.requested, doc)
        remediation = discrete_remediation(doc)
    else:
        die(f"unknown platform '{platform}' in {args.device} (want jetson|discrete)")

    out_path = write_json(args.out, doc)
    say(f"reference_clock_mhz: {doc['reference_clock_mhz']}")
    say(f"verdict: {doc['verdict']} — written to {out_path}")
    if doc["verdict"] == "FAIL":
        for cause in causes:
            print(f"  cause: {cause}", file=sys.stderr)
        die(f"clock lock NOT verified — do not measure.\n{remediation}")
    sys.exit(0)


if __name__ == "__main__":
    main()
