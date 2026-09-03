#!/usr/bin/env bash
#=============================================================================
# Device ceilings runner — preflight, lock, verify, measure, drift, report
#
# Purpose: produce this device's measured capacity ceilings (tensor GEMM per
# precision, bandwidth kernel suite, CPU-load haircut, CUDA-core fp32,
# sustained-vs-burst) under the strict run regime: headless, exclusive
# GPU, clocks locked at max and PROVEN locked. The an earlier run discrete run showed clock
# locks can silently not take, so lock verification and under-load drift
# sampling are mandatory stages here, not optional extras.
#
# Stages:
#   1. preflight     — refuse to measure with a desktop or foreign GPU clients
#                      present (--require-maxn on jetson: ceilings need MAXN)
#   2. clock lock    — jetson_clocks store+apply | nvidia-smi -pm/-lgc/-lmc;
#                      an EXIT trap restores the pre-run state on normal exit,
#                      set -e, and Ctrl-C — but NOT on SIGKILL of the process
#                      group. If you hard-kill a run, the clocks stay pinned and
#                      silently bias whatever measures next: check and release by
#                      hand -- jetson:  sudo jetson_clocks --restore <prov>/jetson_clocks_saved.conf
#                                       (or reboot); discrete:  sudo nvidia-smi -rgc -rmc
#   3. verify_lock   — abort before measuring anything if the lock did not take
#   4. provenance    — environment snapshot + nvidia-smi -q (+ preflight.json)
#   5. measure       — measure_ceilings_thorough.py (~15 GPU-minutes) with
#                      clock_sampler.py attached to its PID for the duration
#   6. drift report  — quantitative under-load clock verdict from the samples
#   7. TXT report    — write_ceilings_report.py -> ceilings_report.txt
#
# Usage:
#   DEVICE_CFG=device_configs/<dev>.json ./run_device_ceilings.sh [out_dir]
#     out_dir default: <kit>/results_${DEVICE_TAG}_ceilings${MODE_SUFFIX}
#     (MODE_SUFFIX names the power mode when it is not MAXN, so a run never
#      overwrites another mode's reference)
#
# Env:
#   DEVICE_CFG    (required) device config json — schema for every reader
#   DEVICE_TAG    (optional) defaults to the config's device_tag field
#   SUSTAIN_PREC  (optional) forwarded to suite 6 (fp16 default, int8 option);
#                 recorded in provenance and meta.sustain_prec so downstream
#                 readers attribute the sustained samples correctly
#   CEIL_SMOKE=1  (optional) plumbing check: one GEMM size, one precision, short
#                 passes. Proves the whole pipeline runs in ~2 min; the numbers
#                 are NOT ceilings (too few sizes to find a peak) — for wiring,
#                 not measurement. Overrides: CEIL_GEMM_SIZES, TRT_GEMM_SIZES,
#                 TRT_GEMM_PRECISIONS give finer control.
#   ALLOW_DESKTOP=1  pass --allow-desktop to preflight (smoke runs only —
#                 stamps smoke_only:true, numbers invalid downstream)
#=============================================================================
set -euo pipefail

KIT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$PATH:/usr/src/tensorrt/bin"

if [ -n "${CEIL_SMOKE:-}" ]; then
  : "${TRT_GEMM_SIZES:=2048}" "${TRT_GEMM_PRECISIONS:=fp16}" "${CEIL_GEMM_SIZES:=2048}"
  : "${TRT_ITERATIONS:=50}" "${TRT_DURATION_S:=2}" "${SUSTAIN_SECONDS:=10}"
  export TRT_GEMM_SIZES TRT_GEMM_PRECISIONS CEIL_GEMM_SIZES TRT_ITERATIONS TRT_DURATION_S SUSTAIN_SECONDS
fi

die() { echo "[FATAL] $*" >&2; exit 1; }

# Auto-select the config from the machine when DEVICE_CFG is unset, so this entry
# point also takes no arguments (matches run_ceilings_only.sh). DEVICE_CFG still wins.
if [ -z "${DEVICE_CFG:-}" ]; then
  DEVICE_CFG="$(python3 "$KIT_DIR/common/pick_device_config.py" 2>/dev/null)" \
    || die "could not choose a device config for this machine. Fill one from
  configs/device_configs/device_config.template.json (set platform, device_name_match,
  the datasheet block, required_power_mode) and drop it in configs/device_configs/,
  or point DEVICE_CFG at it directly."
fi
[ -f "$DEVICE_CFG" ] || die "DEVICE_CFG does not exist: $DEVICE_CFG"

cfg_get() {
  python3 -c "import json,sys; v=json.load(open(sys.argv[1])).get(sys.argv[2]); print('' if v is None else v)" \
    "$DEVICE_CFG" "$1"
}
PLATFORM="$(cfg_get platform)"
[ -n "$PLATFORM" ] || die "device config has no 'platform' field: $DEVICE_CFG"
DEVICE_TAG="${DEVICE_TAG:-$(cfg_get device_tag)}"
[ -n "$DEVICE_TAG" ] || die "DEVICE_TAG unset and device config has no device_tag: $DEVICE_CFG"
# Resolved once here: suite 6 runs at this precision and every downstream
# reader (report writer, model-bench stage 5) attributes the sustained samples
# to it — an unrecorded operator override would silently mislabel them.
SUSTAIN_PREC="${SUSTAIN_PREC:-fp16}"

OUT="${1:-$KIT_DIR/results_${DEVICE_TAG}_ceilings${MODE_SUFFIX:-}}"
mkdir -p "$OUT"/{raw,provenance}
PROV="$OUT/provenance"
LOG="$OUT/ceilings_pipeline.log"
say() { echo -e "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

#-----------------------------------------------------------------------------
# Cleanup: ALWAYS release locked clocks and stop the sampler on exit — normal
# completion, error (set -e), or Ctrl-C. Leaving clocks pinned wastes power
# and invalidates whatever runs on this machine next.
#-----------------------------------------------------------------------------
CLOCKS_LOCKED=0
SAMPLER_PID=""
KEEPALIVE_PID=""
start_sudo_keepalive() {
  # A ceiling run can outlive the sudo timestamp (the sweep alone is ~2 h). Keep
  # it warm so the EXIT-trap restore never dead-ends on a password prompt. Needs
  # one interactive prime before the run; refreshes silently after that.
  ( while true; do sudo -n true 2>/dev/null || exit 0; sleep 60; done ) &
  KEEPALIVE_PID=$!
}
cleanup_clocks() {
  [ "$CLOCKS_LOCKED" = 1 ] || return 0
  say "releasing clock locks..."
  if [ "$PLATFORM" = jetson ]; then
    if [ -f "$PROV/jetson_clocks_saved.conf" ]; then
      # -n so a lapsed sudo cache fails fast with a message instead of blocking
      # forever on a password prompt no one can answer (headless, no tty).
      if sudo -n jetson_clocks --restore "$PROV/jetson_clocks_saved.conf" 2>/dev/null; then
        say "jetson clocks restored to pre-run state"
      else
        say "WARN: clocks still pinned at max — sudo cache lapsed during the run."
        say "      restore with:  sudo jetson_clocks --restore $PROV/jetson_clocks_saved.conf"
        say "      (or simply reboot; the lock does not survive a reboot)"
      fi
    fi
  else
    sudo -n nvidia-smi -rgc >/dev/null 2>&1 && say "SM clock lock released" \
      || say "WARN: SM clock lock NOT released (sudo cache lapsed) — run: sudo nvidia-smi -rgc"
    sudo -n nvidia-smi -rmc >/dev/null 2>&1 && say "MEM clock lock released"
  fi
  CLOCKS_LOCKED=0
}
cleanup_all() {
  if [ -n "$SAMPLER_PID" ]; then
    kill "$SAMPLER_PID" 2>/dev/null || true
    SAMPLER_PID=""
  fi
  [ -n "$KEEPALIVE_PID" ] && { kill "$KEEPALIVE_PID" 2>/dev/null || true; KEEPALIVE_PID=""; }
  cleanup_clocks
}
trap cleanup_all EXIT INT TERM

#=============================================================================
# STAGE 1 — PREFLIGHT
# Refuse to burn GPU-minutes on numbers the regime would invalidate: desktop
# compositor up, foreign GPU clients, or (jetson) a non-MAXN power model.
#=============================================================================
stage_preflight() {
  say "=== stage 1: preflight ==="
  local pf_args=(--device "$DEVICE_CFG" --out "$PROV/preflight.json")
  if [ "$PLATFORM" = jetson ]; then pf_args+=(--require-maxn); fi
  if [ "${ALLOW_DESKTOP:-0}" = 1 ]; then
    pf_args+=(--allow-desktop)
    say "WARN: ALLOW_DESKTOP=1 — smoke run only; results are stamped smoke_only and are not run numbers"
  fi
  local rc=0
  set +e
  python3 "$KIT_DIR/common/preflight.py" "${pf_args[@]}"
  rc=$?
  set -e
  if [ "$rc" -eq 2 ]; then
    die "preflight REFUSED to measure — fix the conditions it listed (headless, exclusive GPU$( [ "$PLATFORM" = jetson ] && echo ', MAXN')) and rerun"
  elif [ "$rc" -ne 0 ]; then
    die "preflight failed (exit $rc) — see its output above"
  fi
  say "preflight passed — evidence: $PROV/preflight.json"
}

#=============================================================================
# STAGE 2 — CLOCK LOCK
# Locked clocks are the fixed reference every ceiling is measured against.
#=============================================================================
MAXGC=""
MAXMC=""
stage_lock() {
  say "=== stage 2: clock lock ==="
  if [ "$PLATFORM" = jetson ]; then
    sudo jetson_clocks --store "$PROV/jetson_clocks_saved.conf" 2>/dev/null \
      || say "WARN: could not store pre-run clock config; exit-trap restore unavailable"
    sudo jetson_clocks || die "jetson_clocks failed — cannot lock clocks"
    CLOCKS_LOCKED=1
    say "jetson_clocks applied (clocks pinned to max)"
    start_sudo_keepalive
    sudo jetson_clocks --show > "$PROV/clocks_locked.txt" 2>/dev/null || true
  else
    # Lock targets. Some parts cannot hold the nameplate bins under tensor
    # load (a discrete Blackwell card: memory 14001 MHz sags to the 13365 bin; SM realizes
    # a workload-dependent DVFS clock at the power cap - 2437 under a light
    # GEMM, ~2167 under a heavy one - so a boost-clock reference makes every
    # heavy model fail drift). When the device config declares sm_lock_mhz /
    # mem_lock_mhz, lock to those sustained-supported bins so
    # requested==realized for every workload and the gates judge on honest
    # terms; otherwise use the hardware max. These feed both the locks and
    # verify_lock's --requested.
    MAXGC=$(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits | head -1)
    MAXMC=$(nvidia-smi --query-gpu=clocks.max.memory --format=csv,noheader,nounits | head -1)
    CFG_SMLOCK=$(cfg_get sm_lock_mhz)
    [ -n "$CFG_SMLOCK" ] && MAXGC="$CFG_SMLOCK"
    CFG_MEMLOCK=$(cfg_get mem_lock_mhz)
    [ -n "$CFG_MEMLOCK" ] && MAXMC="$CFG_MEMLOCK"
    sudo nvidia-smi -pm 1 >/dev/null
    sudo nvidia-smi -lgc "$MAXGC" >/dev/null || die "nvidia-smi -lgc $MAXGC failed — cannot lock SM clock"
    CLOCKS_LOCKED=1
    sudo nvidia-smi -lmc "$MAXMC" >/dev/null 2>&1 \
      || say "WARN: -lmc unsupported on this part; memory clock floats (verify_lock will judge)"
    # An idle read-back proves nothing on discrete parts — verification under
    # load in the next stage is the only evidence the lock took.
    say "requested SM=$MAXGC MHz MEM=$MAXMC MHz (verification under load follows)"
    start_sudo_keepalive
    nvidia-smi --query-gpu=clocks.gr,clocks.mem --format=csv > "$PROV/clocks_locked.txt" 2>/dev/null || true
  fi
}

#=============================================================================
# STAGE 3 — VERIFY LOCK
# Abort before measuring if the lock did not take (the silent-fail case).
#=============================================================================
stage_verify_lock() {
  say "=== stage 3: verify lock ==="
  local v_args=()
  if [ "$PLATFORM" != jetson ]; then v_args=(--requested "sm=${MAXGC},mem=${MAXMC}"); fi
  python3 "$KIT_DIR/common/verify_lock.py" --device "$DEVICE_CFG" --out "$PROV/lock_verified.json" \
    ${v_args[@]+"${v_args[@]}"} \
    || die "clock lock verification FAILED — the lock did not take; re-lock (see lock_verified.json remediation) before rerunning"
  say "lock verified — $PROV/lock_verified.json"
}

#=============================================================================
# STAGE 4 — PROVENANCE
# Every condition the numbers depend on, recorded as strict key: value lines
# so the report writer (and any later audit) can parse them mechanically.
#=============================================================================
stage_provenance() {
  say "=== stage 4: provenance capture ==="
  {
    echo "run_date: $(date -Iseconds)"
    echo "host: $(hostname)"
    echo "kernel: $(uname -r)"
    [ -f /proc/device-tree/model ] && echo "board: $(tr -d '\0' </proc/device-tree/model)"
    [ -f /etc/nv_tegra_release ] && echo "l4t: $(head -1 /etc/nv_tegra_release)"
    if command -v nvidia-smi >/dev/null; then
      SMI=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1)
      echo "gpu_name: $(echo "$SMI" | cut -d, -f1 | xargs)"
      echo "driver: $(echo "$SMI" | cut -d, -f2 | xargs)"
      echo "vram_total: $(echo "$SMI" | cut -d, -f3 | xargs)"
    fi
    command -v trtexec >/dev/null && echo "trtexec: $(trtexec --help 2>&1 | grep -m1 -i tensorrt || true)"
    python3 -c "import tensorrt; print('tensorrt_py:', tensorrt.__version__)" 2>/dev/null || true
    nvcc --version 2>/dev/null | grep release | sed 's/^/nvcc: /' || true
    python3 -c "import torch; print('torch:', torch.__version__)" 2>/dev/null || true
    echo "platform: $PLATFORM"
    echo "device_tag: $DEVICE_TAG"
    echo "device_cfg: $DEVICE_CFG"
    echo "sustain_prec: $SUSTAIN_PREC"
  } > "$PROV/environment.txt"
  command -v nvidia-smi >/dev/null && nvidia-smi -q > "$PROV/nvidia-smi-q.txt" 2>/dev/null || true
  say "provenance written to $PROV/"
}

#=============================================================================
# STAGE 5a — TRT ATTAINABLE-PEAK COMPUTE (opt level 5)
# The authoritative per-precision compute ceiling: trtexec-built GEMM engines
# at --builderOptimizationLevel=5, the same instrument the model half uses.
# Runs under the locked clocks. Non-fatal: if it fails, the torch cross-check
# in stage 5 still stands and the report falls back to it for section [2].
# Not drift-sampled on purpose — engine BUILD phases are GPU-idle and would
# pollute the sustained-load drift verdict; trtexec's own median-of-many-
# iterations timing is robust to transient clocks, and the stage-6 verdict
# (sampled over the continuous torch load) characterizes this box's clocks.
#=============================================================================
TRT_COMPUTE_JSON=""
stage_measure_trt() {
  say "=== stage 5a: TRT attainable-peak compute (opt level 5) ==="
  local log="$OUT/raw/trt_compute_log.txt"
  mkdir -p "$OUT/raw"
  if python3 "$KIT_DIR/device_ceilings/measure_compute_trt.py" "$OUT/raw" > "$log" 2>&1; then
    TRT_COMPUTE_JSON="$OUT/raw/trt_compute.json"
    say "TRT compute: $TRT_COMPUTE_JSON"
    grep -m1 '^\[.*\] DONE' "$log" | sed 's/^/    /' || true
  else
    say "WARN: TRT compute probe failed (exit $?) — see $log; report falls back to the torch cross-check for section [2]"
  fi
}

#=============================================================================
# STAGE 5 — MEASUREMENT (with drift sampling attached)
# The sampler rides on the measurement PID for the whole run; its samples are
# the evidence behind the stage-6 drift verdict.
#=============================================================================
MEAS_LOG=""
stage_measure() {
  say "=== stage 5: thorough ceiling suites (~15 GPU-minutes) ==="
  MEAS_LOG="$OUT/raw/measure_log.txt"
  : > "$MEAS_LOG"
  SUSTAIN_PREC="$SUSTAIN_PREC" \
    python3 "$KIT_DIR/device_ceilings/measure_ceilings_thorough.py" "$OUT/raw" \
    > "$MEAS_LOG" 2>&1 &
  local meas_pid=$!
  python3 "$KIT_DIR/common/clock_sampler.py" --device "$DEVICE_CFG" --out "$OUT/clock_samples.csv" \
    --pid "$meas_pid" --phase ceilings >> "$LOG" 2>&1 &
  SAMPLER_PID=$!
  say "measurement pid=$meas_pid, sampler pid=$SAMPLER_PID — live log follows"
  tail -f --pid="$meas_pid" "$MEAS_LOG" || true
  local meas_rc=0
  wait "$meas_pid" || meas_rc=$?
  wait "$SAMPLER_PID" || say "WARN: clock sampler exited nonzero — drift report may be partial"
  SAMPLER_PID=""
  [ "$meas_rc" -eq 0 ] || die "measure_ceilings_thorough.py failed (exit $meas_rc) — see $MEAS_LOG"
  [ -f "$OUT/raw/results.json" ] || die "no results.json produced — see $MEAS_LOG"
  # The frozen measurement script does not record which precision suite 6 ran;
  # stamp it into meta so the report writer and model-bench stage 5 attribute
  # the sustained samples (and the sustained/burst ratios) to the right pipe.
  python3 - "$OUT/raw/results.json" "$SUSTAIN_PREC" <<'PY'
import json, sys
path, prec = sys.argv[1], sys.argv[2]
d = json.load(open(path))
d.setdefault('meta', {})['sustain_prec'] = prec
with open(path, 'w') as f:
    json.dump(d, f, indent=1)
PY
  say "raw results: $OUT/raw/results.json (meta.sustain_prec=$SUSTAIN_PREC)"
}

#=============================================================================
# STAGE 6 — DRIFT REPORT
# Turns the under-load samples into the quantitative verdict that replaces the
# old idle-vs-idle cmp check. Verdict lives inside the JSON; the TXT report
# echoes it in section [6].
#=============================================================================
DRIFT_JSON=""
stage_drift() {
  say "=== stage 6: clock drift report ==="
  if python3 "$KIT_DIR/common/drift_report.py" "$OUT/clock_samples.csv" --device "$DEVICE_CFG" \
       --lock "$PROV/lock_verified.json" --out "$OUT/clock_drift.json" --phase ceilings; then
    DRIFT_JSON="$OUT/clock_drift.json"
    say "drift report: $DRIFT_JSON"
  else
    say "WARN: drift_report could not read the samples — report section [6] will say 'not sampled'"
  fi
}

#=============================================================================
# STAGE 7 — TXT REPORT
#=============================================================================
stage_report() {
  say "=== stage 7: ceilings report ==="
  local d_args=()
  if [ -n "$DRIFT_JSON" ] && [ -f "$DRIFT_JSON" ]; then d_args=(--drift "$DRIFT_JSON"); fi
  if [ -n "$TRT_COMPUTE_JSON" ] && [ -f "$TRT_COMPUTE_JSON" ]; then d_args+=(--trt "$TRT_COMPUTE_JSON"); fi
  python3 "$KIT_DIR/device_ceilings/write_ceilings_report.py" "$OUT/raw/results.json" \
    --device "$DEVICE_CFG" --provenance "$PROV" --out "$OUT/ceilings_report.txt" \
    ${d_args[@]+"${d_args[@]}"} \
    || die "write_ceilings_report.py failed"
}

say "=== device ceilings run: tag=$DEVICE_TAG platform=$PLATFORM ==="
say "device config: $DEVICE_CFG"
say "output dir:    $OUT"
stage_preflight
stage_lock
stage_verify_lock
stage_provenance
stage_measure_trt  # authoritative per-precision compute ceiling (TRT opt level 5)
stage_measure      # torch cross-check + bandwidth/sustained/shaped (drift-sampled)
stage_drift
cleanup_clocks     # report writing needs no GPU; do not keep clocks pinned for it
stage_report
say "DONE — ceilings report: $OUT/ceilings_report.txt"
