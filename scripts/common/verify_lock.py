#!/usr/bin/env python3
"""Post-lock clock verification — prove the lock actually took BEFORE measuring.

Why this exists: on the discrete box (2026-07) `-lgc/-lmc` silently failed — the log
claimed 3090/14001 MHz while the clocks read 180/405. And even a healthy SM lock
realizes at the boost-table tensor-load clock (~2570 observed for a 3090
request), never the requested number. An idle read-back therefore proves nothing
on a discrete GPU (idle cards legitimately downclock even when locked), so the
discrete check runs a real tensor load and judges the REALIZED clocks. On
Jetson, devfreq min==max==cur is authoritative, unprivileged, and needs no load.

Usage:
  verify_lock.py --device <cfg.json> --out <lock_verified.json> \
                 [--requested sm=<mhz>,mem=<mhz>]

Verdict thresholds (pre-declared, not post-hoc):
  jetson — locked iff, for both gpu and emc devfreq nodes, min_freq == max_freq
      == cur_freq (exact Hz equality) and the value equals the config
      lock_targets_mhz at 1 MHz granularity (guard against Hz->MHz rounding
      only). Anything else = FAIL. reference_clock_mhz = config targets.
      --requested is ignored on jetson (targets live in the device config).
  discrete — 5 s fp16 GEMM n=2048 subprocess load; 2 s warm, then 3 s sampled
      at 10 Hz (pynvml preferred, nvidia-smi subprocess fallback):
      FAIL  mem median off requested by > 1%
            (memory clocks lock exactly or not at all — no boost table on mem)
      FAIL  sm median < 0.75 x requested
            (the silent-lock-failure signature: clocks never left idle range)
      FAIL  sm p95 - p5 > 30 MHz
            (a locked SM is flat under load; wander means DVFS owns the clock)
      WARN  sm median in [0.75, 0.97) x requested — proceed, and
            reference_clock_mhz.sm := realized median
            (legitimate boost-table realization; drift downstream must be
            judged against what the silicon actually holds)
      PASS  otherwise; reference_clock_mhz := requested.

`reference_clock_mhz{sm,mem}` is ALWAYS written — drift_report.py judges
against it, never against the requested value. Schema: lockverify/v1.
Exit 0 on PASS/WARN, 1 on FAIL (re-lock remediation printed). stdlib only in
this process; torch runs only inside the load subprocess (import failure there
is fatal with the cu13x remediation — the sm_120 cuBLAS case).
"""
import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

# Pre-declared constants (rationale in the module docstring).
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


def say(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def die(msg):
    print(f'FATAL: {msg}', file=sys.stderr)
    sys.exit(1)


def write_json(path, doc):
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1) + "\n")
    return p


def smi_query(fields):
    """One nvidia-smi CSV query row as a list of strings, None on failure."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    return [v.strip() for v in out.stdout.strip().splitlines()[0].split(",")]


def pctile(sorted_vals, q):
    i = int(round(q * (len(sorted_vals) - 1)))
    return sorted_vals[max(0, min(len(sorted_vals) - 1, i))]


# --------------------------------------------------------------------------
# jetson path: devfreq min==max==cur==target, unprivileged, no load needed
# --------------------------------------------------------------------------
def verify_jetson(cfg, doc):
    cs = cfg["clock_sources"]
    targets = cfg["lock_targets_mhz"]
    readings, causes = {}, []
    for name, key in (("gpu", "gpu_mhz"), ("emc", "emc_mhz")):
        node = cs[key]
        scale = float(node.get("scale_hz_to_mhz", 1e-6))
        d = Path(node["dir"])
        raw = {}
        for attr in ("min_freq", "max_freq", "cur_freq"):
            try:
                raw[attr] = int((d / attr).read_text().strip())
            except (OSError, ValueError):
                raw[attr] = None
        mhz = {k: (None if v is None else round(v * scale, 3)) for k, v in raw.items()}
        target = float(targets[name])
        locked = (None not in raw.values()
                  and raw["min_freq"] == raw["max_freq"] == raw["cur_freq"]
                  and abs(mhz["cur_freq"] - target) < JETSON_TOL_MHZ)
        if not locked:
            causes.append(f"{name}: min/max/cur = {mhz['min_freq']}/{mhz['max_freq']}/"
                          f"{mhz['cur_freq']} MHz vs target {target:g} MHz")
        readings[name] = {"dir": str(d), "raw_hz": raw, "mhz": mhz,
                          "target_mhz": target, "locked": locked}

    doc["jetson"] = readings
    doc["reference_clock_mhz"] = {"sm": float(targets["gpu"]), "mem": float(targets["emc"])}
    doc["requested_clock_mhz"] = dict(doc["reference_clock_mhz"])
    doc["verdict"] = "PASS" if not causes else "FAIL"
    doc["verdict_causes"] = causes
    return causes


# --------------------------------------------------------------------------
# discrete path: idle read-back informational, verdict from under-load samples
# --------------------------------------------------------------------------
def resolve_requested(cfg, requested_arg):
    req = {"sm": None, "mem": None}
    if requested_arg:
        for part in requested_arg.split(","):
            if "=" not in part:
                die(f"bad --requested fragment '{part}' (want sm=<mhz>,mem=<mhz>)")
            k, v = part.split("=", 1)
            k = k.strip()
            if k not in req:
                die(f"bad --requested key '{k}' (want sm/mem)")
            req[k] = float(v)
    targets = cfg.get("lock_targets_mhz") or {}
    # A device-config mem_lock_mhz (the memory bin that actually holds under
    # load) outranks the nameplate lock target: requesting an unholdable bin
    # produces an honest-but-useless FAIL and a remediation that re-requests
    # the same unholdable bin.
    if req["mem"] is None and isinstance(cfg.get("mem_lock_mhz"), (int, float)):
        req["mem"] = float(cfg["mem_lock_mhz"])
    pending = []
    for k in ("sm", "mem"):
        if req[k] is not None:
            continue
        v = targets.get(k)
        if isinstance(v, (int, float)):
            req[k] = float(v)
        elif isinstance(v, str):
            # config points at an nvidia-smi field (e.g. clocks.max.graphics)
            pending.append((k, v))
        else:
            die(f"no requested {k} clock: pass --requested or set lock_targets_mhz.{k}")
    if pending:
        vals = smi_query(",".join(f for _, f in pending))
        if vals is None or len(vals) < len(pending):
            die("nvidia-smi query for max clocks failed — pass --requested sm=<mhz>,mem=<mhz>")
        for (k, _), v in zip(pending, vals):
            try:
                req[k] = float(v)
            except ValueError:
                die(f"nvidia-smi returned non-numeric max clock for {k}: '{v}'")
    return req


def make_sampler():
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)

        def sample():
            return (float(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)),
                    float(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM)))
        return sample, "pynvml"
    except Exception:
        def sample():
            vals = smi_query("clocks.sm,clocks.mem")
            if vals is None or len(vals) < 2:
                return None
            try:
                return (float(vals[0]), float(vals[1]))
            except ValueError:
                return None
        return sample, "nvidia-smi"


def verify_discrete(cfg, requested_arg, doc):
    req = resolve_requested(cfg, requested_arg)
    doc["requested_clock_mhz"] = {"sm": req["sm"], "mem": req["mem"]}
    notes = doc.setdefault("notes", [])

    sample, sampler_name = make_sampler()
    if sampler_name == "nvidia-smi":
        notes.append("pynvml unavailable — nvidia-smi subprocess sampling (effective rate may be below 10 Hz)")

    idle = sample()
    idle_block = None if idle is None else {"sm": idle[0], "mem": idle[1]}
    say(f"idle read-back (informational only): {idle_block}")

    say(f"launching {LOAD_S:.0f} s fp16 GEMM n={GEMM_N} load subprocess")
    proc = subprocess.Popen([sys.executable, "-c", LOAD_SRC, str(LOAD_S), str(GEMM_N)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    line = (proc.stdout.readline() or "").strip()
    if line != "LOAD_RUNNING":
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

    t_load = time.monotonic()
    time.sleep(WARM_S)
    sm_vals, mem_vals = [], []
    t0 = time.monotonic()
    n_target = int(SAMPLE_S * RATE_HZ)
    for i in range(n_target):
        tgt = t0 + i / RATE_HZ
        now = time.monotonic()
        # never sample past the load window — trailing idle reads would fake wander
        if now > t_load + LOAD_S - 0.05:
            notes.append(f"sampling stopped early at {len(sm_vals)} samples (load window expired)")
            break
        if tgt > now:
            time.sleep(tgt - now)
        try:
            v = sample()
        except Exception:
            v = None
        if v is not None:
            sm_vals.append(v[0])
            mem_vals.append(v[1])
    try:
        proc.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        notes.append("load subprocess had to be killed after the sampling window")
    if proc.returncode not in (0, None):
        notes.append(f"load subprocess exited rc={proc.returncode} — samples may include idle time")

    doc["discrete"] = {
        "idle_readback_mhz": idle_block,
        "idle_readback_note": "informational only — idle discrete GPUs legitimately downclock even when locked",
        "sampler": sampler_name,
        "load": {"kind": "torch-fp16-gemm", "n": GEMM_N, "duration_s": LOAD_S,
                 "warm_s": WARM_S, "sample_s": SAMPLE_S, "rate_hz": RATE_HZ},
    }

    if not sm_vals or not mem_vals:
        doc["reference_clock_mhz"] = {"sm": req["sm"], "mem": req["mem"]}
        doc["verdict"] = "FAIL"
        doc["verdict_causes"] = ["no clock samples collected under load — sampler broken"]
        return doc["verdict_causes"]

    sm_sorted = sorted(sm_vals)
    sm_med = statistics.median(sm_vals)
    sm_p5 = pctile(sm_sorted, 0.05)
    sm_p95 = pctile(sm_sorted, 0.95)
    mem_med = statistics.median(mem_vals)
    doc["discrete"]["realized"] = {
        "sm": {"median": round(sm_med, 1), "p5": round(sm_p5, 1), "p95": round(sm_p95, 1),
               "min": round(min(sm_vals), 1), "max": round(max(sm_vals), 1), "n": len(sm_vals)},
        "mem": {"median": round(mem_med, 1), "min": round(min(mem_vals), 1),
                "max": round(max(mem_vals), 1), "n": len(mem_vals)},
    }

    causes, checks = [], []
    mem_dev = abs(mem_med - req["mem"]) / req["mem"]
    mem_ok = mem_dev <= MEM_TOL_FRAC
    checks.append({"name": "mem_median_within_1pct_of_requested", "limit_pct": 1.0,
                   "value_pct": round(mem_dev * 100, 3), "pass": mem_ok})
    if not mem_ok:
        causes.append(f"mem median {mem_med:.0f} MHz is {mem_dev * 100:.1f}% off "
                      f"requested {req['mem']:.0f} MHz (limit 1%)")

    ratio = sm_med / req["sm"]
    floor_ok = ratio >= SM_FLOOR_FRAC
    checks.append({"name": "sm_median_floor", "limit_ratio": SM_FLOOR_FRAC,
                   "value_ratio": round(ratio, 4), "pass": floor_ok})
    if not floor_ok:
        causes.append(f"sm median {sm_med:.0f} MHz is {ratio:.2f}x requested "
                      f"{req['sm']:.0f} MHz (floor {SM_FLOOR_FRAC}x) — the silent-lock-failure signature")

    flat = sm_p95 - sm_p5
    flat_ok = flat <= SM_FLAT_MHZ
    checks.append({"name": "sm_flatness_p95_minus_p5", "limit_mhz": SM_FLAT_MHZ,
                   "value_mhz": round(flat, 1), "pass": flat_ok})
    if not flat_ok:
        causes.append(f"sm p95-p5 = {flat:.0f} MHz under load (limit {SM_FLAT_MHZ:.0f}) — DVFS owns the clock")

    boost_warn = not causes and floor_ok and ratio < SM_BOOST_OK_FRAC
    checks.append({"name": "sm_boost_table_band", "band_ratio": [SM_FLOOR_FRAC, SM_BOOST_OK_FRAC],
                   "value_ratio": round(ratio, 4), "in_band": boost_warn})
    doc["checks"] = checks

    if causes:
        verdict, ref_sm = "FAIL", req["sm"]
    elif boost_warn:
        verdict, ref_sm = "WARN", round(sm_med, 1)
        msg = (f"sm realized {sm_med:.0f} MHz for a {req['sm']:.0f} MHz request "
               f"({ratio:.2f}x) — boost-table realization; reference := realized")
        notes.append(msg)
        print(f"WARN: {msg}")
    else:
        verdict, ref_sm = "PASS", req["sm"]

    doc["reference_clock_mhz"] = {"sm": ref_sm, "mem": req["mem"]}
    doc["verdict"] = verdict
    doc["verdict_causes"] = causes
    return causes


def main():
    ap = argparse.ArgumentParser(description="verify a GPU clock lock actually took (lockverify/v1)")
    ap.add_argument("--device", required=True, help="device config JSON")
    ap.add_argument("--out", required=True, help="output lock_verified.json path")
    ap.add_argument("--requested", default=None, help="sm=<mhz>,mem=<mhz> override (discrete only)")
    args = ap.parse_args()

    try:
        cfg = json.loads(Path(args.device).resolve().read_text())
    except (OSError, ValueError) as e:
        die(f"cannot read device config {args.device}: {e}")

    platform = cfg.get("platform")
    doc = {
        "schema": "lockverify/v1",
        "device_tag": cfg.get("device_tag"),
        "device_id": cfg.get("device_id"),
        "platform": platform,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "notes": [],
    }

    if platform == "jetson":
        if args.requested:
            doc["notes"].append("--requested ignored on jetson (targets come from the device config)")
        causes = verify_jetson(cfg, doc)
        remediation = ("re-lock: sudo nvpmodel -m 0 (MAXN) if needed, then sudo jetson_clocks; "
                       "devfreq min==max==cur is the authoritative check — re-run this script after")
    elif platform == "discrete":
        causes = verify_discrete(cfg, args.requested, doc)
        rq = doc.get("requested_clock_mhz", {})
        rq_sm = f"{rq['sm']:.0f}" if isinstance(rq.get("sm"), (int, float)) else "<mhz>"
        rq_mem = f"{rq['mem']:.0f}" if isinstance(rq.get("mem"), (int, float)) else "<mhz>"
        remediation = ("re-lock and re-verify before measuring:\n"
                       "  sudo nvidia-smi -pm 1\n"
                       f"  sudo nvidia-smi -lgc {rq_sm}\n"
                       f"  sudo nvidia-smi -lmc {rq_mem}\n"
                       "idle read-back proves nothing on a discrete GPU — only this under-load check does")
    else:
        die(f"unknown platform '{platform}' in {args.device} (want jetson|discrete)")

    out_path = write_json(args.out, doc)
    say(f"reference_clock_mhz: {doc['reference_clock_mhz']}")
    say(f"verdict: {doc['verdict']} — written to {out_path}")
    if doc["verdict"] == "FAIL":
        for c in causes:
            print(f"  cause: {c}", file=sys.stderr)
        die(f"clock lock NOT verified — do not measure.\n{remediation}")
    sys.exit(0)


if __name__ == "__main__":
    main()
