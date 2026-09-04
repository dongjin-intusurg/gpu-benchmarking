#!/usr/bin/env bash
#=============================================================================
# Per-model demand measurement pipeline (solo-exclusive regime)
#
# Measures the DEMAND side of every capacity budget for each model in the mix
# manifest: p99 latency, bytes/frame, VRAM footprint. The CAPACITY side comes
# from the thorough ceilings run (CEILINGS_JSON) — never re-measured here, so
# demand and capacity always share one provenance chain per device.
#
# Pipeline stages:
#   0. Provenance + preflight   — environment snapshot; common/preflight.py
#                                 refuses a non-headless or shared GPU (the
#                                 solo-exclusive regime is a precondition, not
#                                 an assumption)
#   1. Clock locking + verify   — lock, then common/verify_lock.py proves the
#                                 lock actually took (under load on discrete;
#                                 locks have silently failed before) — FAIL
#                                 aborts before any number is produced
#   3. Per-model profiling      — engines are ADOPTED only (built and hashed by
#                                 build_model_engines.sh):
#                                   3b. 1000-iter p99 with a per-PID clock
#                                       sampler + quantitative drift verdict
#                                   3b2. nsys timeline (also clock-sampled)
#                                   3c. NCU counters — counts only, never times
#   4. Clock-dial experiment    — optional memory-vs-SM sensitivity check
#                                 (RUN_CLOCK_DIAL=1, discrete only; Jetson EMC
#                                 dial lives in model_bench/measure_dram_demand.sh)
#   5. Post-processing          — budgets, U_max, C, L, N = min(L,C), Score;
#                                 plus regime/preflight/clock-integrity/accuracy
#                                 blocks and report.md
#   5b. Roofline reports        — per-kernel bound classification against the
#                                 measured ceilings (roofline_report.py)
# (there is no stage 2: the quick in-pipeline ceilings are gone — the thorough
#  suite's results.json is the capacity source of record)
#
# Usage:
#   DEVICE_CFG=<cfg.json> CEILINGS_JSON=<results.json> \
#     ./run_model_bench.sh [mix.csv] [output_dir]
#
#   mix.csv columns (header required):
#     name,onnx,precision,hz,deadline_ms,arch_gflops,extra
#       name        — model identifier (row label in the report)
#       onnx        — path to the prebuilt .engine (build_model_engines.sh
#                     writes this row; a raw .onnx here is an error)
#       precision   — fp16 | int8 | fp8 | fp32   (deployment precision)
#       hz          — required cycles/second from the frozen mix definition
#       deadline_ms — p99 latency bound (e.g. 33.3 for 30 Hz stages)
#       arch_gflops — architectural GFLOPs/frame from the mix document
#                     (naive graph count, FMA=2; NOT profiler-measured)
#       extra       — extra trtexec flags (plugins, shapes); may itself contain
#                     commas — the positional reader absorbs the remainder
#
# Env toggles:
#   DEVICE_CFG          (required) device config JSON from device_configs/
#   CEILINGS_JSON       (required) thorough ceilings run's raw results.json
#   ACCURACY_GATE_JSON  optional accuracy.json from run_accuracy_gate.py
#   DRAM_DEMAND_JSON    optional causal EMC-dial fit (trumps NCU byte proxies)
#   VRAM_CAPACITY_MB    budget-3 denominator override (default: device config
#                       vram_budget_cap_mb, else MemTotal on Jetson, else
#                       nvidia-smi memory.total)
#   SKIP_NCU=1          skip stage 3c        NCU_FULL=1        ncu --set full
#   RUN_CLOCK_DIAL=1    enable stage 4       P99_ALLOW_MEAN=1  degraded p99
#                                            escape hatch (recorded as such)
#   ALLOW_DESKTOP=1     pass --allow-desktop to preflight (smoke rehearsal only;
#                       stamps smoke_only:true — numbers invalid downstream)
#   REPORT_ONLY=1       run stages 5+5b only against an existing output dir
#=============================================================================
set -euo pipefail
export PATH="$PATH:/usr/src/tensorrt/bin"

MIX_FILE="${1:-mix.csv}"
OUT_DIR="${2:-ecc_results_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"/{provenance,models,report}
LOG="$OUT_DIR/pipeline.log"
say() { echo -e "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
die() { echo "[FATAL] $*" >&2; exit 1; }

KIT="$(cd "$(dirname "$0")/.." && pwd)"

[ -n "${DEVICE_CFG:-}" ] || die "DEVICE_CFG is required — point it at device_configs/<device>.json"
[ -f "$DEVICE_CFG" ] || die "DEVICE_CFG not found: $DEVICE_CFG"
[ -n "${CEILINGS_JSON:-}" ] || die "CEILINGS_JSON is required — the thorough ceilings run's raw results.json (device_ceilings/run_device_ceilings.sh writes it under <out>/raw/)"
[ -f "$CEILINGS_JSON" ] || die "CEILINGS_JSON not found: $CEILINGS_JSON"

# Platform from the device config (single source of truth); file-probe fallback
# keeps the script honest if the config predates the field.
PLATFORM=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("platform",""))' "$DEVICE_CFG")
if [ -z "$PLATFORM" ]; then
  if [ -f /etc/nv_tegra_release ]; then PLATFORM=jetson; else PLATFORM=discrete; fi
fi

#-----------------------------------------------------------------------------
# Clock-lock cleanup: ALWAYS release locked clocks on exit — normal completion,
# error (set -e), or Ctrl-C. Leaving clocks pinned wastes power and invalidates
# whatever runs on this machine next. Registered as an EXIT trap so it cannot
# be skipped.
#-----------------------------------------------------------------------------
CLOCKS_LOCKED=0
KEEPALIVE_PID=""
start_sudo_keepalive() {
  # model measurement can run for hours and outlive the sudo timestamp; keep it
  # warm so the EXIT-trap clock restore always succeeds.
  ( while true; do sudo -n true 2>/dev/null || exit 0; sleep 60; done ) &
  KEEPALIVE_PID=$!
}
cleanup_clocks() {
  [ "$CLOCKS_LOCKED" = 1 ] || return 0
  say "releasing clock locks..."
  if [ "${PLATFORM:-}" = jetson ]; then
    if [ -f "$OUT_DIR/provenance/jetson_clocks_saved.conf" ]; then
      # -n: a lapsed sudo cache must fail fast, never block on a password prompt
      # that a headless run cannot answer (that hang leaves clocks pinned).
      sudo -n jetson_clocks --restore "$OUT_DIR/provenance/jetson_clocks_saved.conf" 2>/dev/null \
        && say "jetson clocks restored to pre-run state" \
        || say "WARN: jetson_clocks --restore failed — restore manually or reboot"
    fi
  else
    sudo nvidia-smi -rgc >/dev/null 2>&1 && say "SM clock lock released"
    sudo nvidia-smi -rmc >/dev/null 2>&1 && say "MEM clock lock released"
  fi
  CLOCKS_LOCKED=0
  [ -n "$KEEPALIVE_PID" ] && { kill "$KEEPALIVE_PID" 2>/dev/null || true; KEEPALIVE_PID=""; }
}
trap cleanup_clocks EXIT INT TERM

#=============================================================================
# STAGE 0 — PROVENANCE + PREFLIGHT
#
# What we are doing: recording every condition under which the numbers are
# produced (device identity, driver/CUDA/TensorRT versions, clock settings,
# power mode), then letting common/preflight.py certify the regime: headless,
# no foreign GPU clients, believable idle baseline. A refusal (exit 2) is a
# hard stop — measuring on a shared or desktop-burdened GPU produces numbers
# that look plausible and are wrong.
#=============================================================================
stage0_provenance() {
  say "=== STAGE 0: provenance capture + preflight ==="
  {
    echo "run_date: $(date -Iseconds)"
    echo "host: $(hostname)"
    echo "kernel: $(uname -r)"
    [ -f /proc/device-tree/model ] && echo "board: $(tr -d '\0' </proc/device-tree/model)"
    [ -f /etc/nv_tegra_release ] && echo "l4t: $(head -1 /etc/nv_tegra_release)"
    command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    command -v trtexec >/dev/null && trtexec --help 2>&1 | grep -m1 -i tensorrt || true
    python3 -c "import tensorrt; print('tensorrt_py:', tensorrt.__version__)" 2>/dev/null || true
    nvcc --version 2>/dev/null | grep release || true
    echo "device_cfg: $DEVICE_CFG"
    python3 -c 'import json,sys; print("device_id:", json.load(open(sys.argv[1])).get("device_id","?"))' "$DEVICE_CFG"
    echo "ceilings_json: $CEILINGS_JSON"
    echo "platform: $PLATFORM"
  } > "$OUT_DIR/provenance/environment.txt"
  command -v nvidia-smi >/dev/null && nvidia-smi -q > "$OUT_DIR/provenance/nvidia-smi-q.txt" 2>/dev/null || true
  say "platform: $PLATFORM"

  RC=0
  PF_ARGS=(--device "$DEVICE_CFG" --out "$OUT_DIR/provenance/preflight.json")
  if [ "${ALLOW_DESKTOP:-0}" = 1 ]; then
    PF_ARGS+=(--allow-desktop)
    say "WARN: ALLOW_DESKTOP=1 — smoke rehearsal; preflight stamps smoke_only and every number downstream is invalid"
  fi
  python3 "$KIT/common/preflight.py" "${PF_ARGS[@]}" || RC=$?
  if [ "$RC" = 2 ]; then
    die "preflight REFUSED — device is not in a measurable state (see $OUT_DIR/provenance/preflight.json and the remediation above)"
  fi
  [ "$RC" = 0 ] || die "preflight crashed (exit $RC) — cannot certify measurement conditions"
  say "preflight passed — evidence in provenance/preflight.json"
}

#=============================================================================
# STAGE 1 — CLOCK LOCKING + VERIFICATION
#
# What we are doing: GPUs auto-adjust clocks continuously (boost, thermal
# DVFS). If clocks float, run-to-run timing differences reflect the clock
# governor, not the workload. Locking establishes the fixed reference all
# latencies are measured against — and because locks have silently failed on
# this suite's hardware before, common/verify_lock.py must PROVE the lock
# took (devfreq read-back on Jetson; under a GEMM load on discrete, where an
# idle read-back proves nothing). FAIL aborts the run. NOTE: lab numbers are
# locked-clock; the final certification run must use the production governor.
#=============================================================================
stage1_lock_clocks() {
  say "=== STAGE 1: clock locking + verification ==="
  if [ "$PLATFORM" = jetson ]; then
    # Jetson: save current clock config (for the exit-trap restore), select the
    # max-power model, then pin all clocks (GPU, EMC, CPU)
    # -n throughout: sudo must already be primed; a lock this script cannot apply
    # itself is a refusal, not a warning (a verified-but-inherited lock has no
    # restore path and no provenance of who pinned it)
    sudo -n nvpmodel -q | tee -a "$LOG" || say "WARN: nvpmodel query failed"
    sudo -n jetson_clocks --store "$OUT_DIR/provenance/jetson_clocks_saved.conf" 2>/dev/null \
      || die "cannot store the pre-run clock config (sudo not primed? run: sudo -v) — refusing to measure without a restore path"
    sudo -n jetson_clocks 2>/dev/null && { say "jetson_clocks applied (clocks pinned to max)"; CLOCKS_LOCKED=1; start_sudo_keepalive; } \
      || die "jetson_clocks failed — the lock must be applied by this run, not inherited"
    sudo -n jetson_clocks --show > "$OUT_DIR/provenance/clocks_locked.txt" 2>/dev/null || true
    python3 "$KIT/common/verify_lock.py" --device "$DEVICE_CFG" \
      --out "$OUT_DIR/provenance/lock_verified.json" \
      || die "clock-lock verification FAILED — devfreq is not pinned to the config targets; re-run jetson_clocks (or reboot) and retry. Evidence: $OUT_DIR/provenance/lock_verified.json"
  else
    # Discrete: read supported clocks, lock graphics and memory to max.
    # Device-config sm_lock_mhz / mem_lock_mhz override the nameplate bins —
    # some parts cannot hold the max under tensor load (RTX PRO 5000: memory
    # 14001 -> 13365; SM realizes a workload-dependent DVFS clock at the
    # power cap, so a boost reference makes every heavy model fail drift),
    # and requesting an unholdable bin makes verify_lock FAIL honestly but
    # uselessly. The config bins are the ones that hold for every workload.
    MAXGC=$(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits | head -1)
    MAXMC=$(nvidia-smi --query-gpu=clocks.max.memory   --format=csv,noheader,nounits | head -1)
    CFG_SMLOCK=$(python3 -c 'import json,sys;v=json.load(open(sys.argv[1])).get("sm_lock_mhz");print(v if v else "")' "$DEVICE_CFG")
    [ -n "$CFG_SMLOCK" ] && MAXGC="$CFG_SMLOCK"
    CFG_MEMLOCK=$(python3 -c 'import json,sys;v=json.load(open(sys.argv[1])).get("mem_lock_mhz");print(v if v else "")' "$DEVICE_CFG")
    [ -n "$CFG_MEMLOCK" ] && MAXMC="$CFG_MEMLOCK"
    sudo nvidia-smi -pm 1 >/dev/null
    sudo nvidia-smi -lgc "$MAXGC" && { say "SM clock locked at ${MAXGC} MHz"; CLOCKS_LOCKED=1; }
    sudo nvidia-smi -lmc "$MAXMC" 2>/dev/null && say "MEM clock locked at ${MAXMC} MHz" \
      || say "WARN: -lmc unsupported on this part; memory clock floats"
    nvidia-smi --query-gpu=clocks.gr,clocks.mem --format=csv > "$OUT_DIR/provenance/clocks_locked.txt"
    # verify under load: -lgc/-lmc have silently not taken on this hardware
    python3 "$KIT/common/verify_lock.py" --device "$DEVICE_CFG" \
      --out "$OUT_DIR/provenance/lock_verified.json" \
      --requested "sm=${MAXGC},mem=${MAXMC}" \
      || die "clock-lock verification FAILED — the lock did not take under load; re-lock (sudo nvidia-smi -pm 1; -lgc/-lmc) or reboot. Evidence: $OUT_DIR/provenance/lock_verified.json"
  fi
  say "lock verified — provenance/lock_verified.json"
}

#=============================================================================
# STAGE 3 — PER-MODEL PROFILING (demand side of every budget)
#
# What we are doing, per model in the mix manifest:
#   3a. ADOPT the pinned deployment artifact (built + hashed upstream by
#       build_model_engines.sh) and re-hash it here — the engine, not the
#       abstract model, is the unit of measurement; a different TRT version/
#       precision is a different artifact and gets re-measured.
#   3b. timing harness: 1000 timed iterations after warmup at locked clocks.
#       p99 latency x required Hz = the model's TIME SHARE (budget 1) and
#       p99 vs deadline = its contribution to L. A per-PID clock sampler
#       records the under-load clock evidence; drift_report.py turns it into
#       the quantitative verdict stage 5 gates measurement validity on.
#   3c. Nsight Compute over one full inference: exact DRAM bytes per kernel
#       (memory-controller counters -> budget 2), and per-pipe instruction
#       counters separating tensor-core work from CUDA-core work (budgets
#       3-4, and the per-kernel split used for cross-device projection).
#=============================================================================
stage3_models() {
  say "=== STAGE 3: per-model profiling ==="
  [ -f "$MIX_FILE" ] || die "mix manifest '$MIX_FILE' not found"

  tail -n +2 "$MIX_FILE" | while IFS=, read -r NAME ONNX PREC HZ DEADLINE AGF EXTRA; do
    [ -z "$NAME" ] && continue
    MDIR="$OUT_DIR/models/$NAME"; mkdir -p "$MDIR"
    say "--- model: $NAME  (prec=$PREC, hz=$HZ, deadline=${DEADLINE}ms) ---"

    # -- 3a. Adopt the pinned engine and record its identity -----------------
    # Engines are never built here: build_model_engines.sh owns builds (fp32
    # reference included) and never overlaps measurement. A raw .onnx in the
    # mix means that step was skipped — refuse rather than build ad hoc.
    if [[ "$ONNX" == *.engine ]]; then
      ENGINE="$ONNX"
      [ -f "$ENGINE" ] || { say "WARN: engine not found for $NAME: $ENGINE — skipping"; continue; }
      say "adopting prebuilt engine: $ENGINE"
    else
      die "mix row '$NAME' points at '$ONNX' — this pipeline adopts prebuilt engines only; run model_bench/build_model_engines.sh first (it writes the mix row with the engine path)"
    fi
    sha256sum "$ENGINE" > "$MDIR/engine.sha256"   # provenance: the artifact identity

    # -- 3b. Timing harness: p50/p99 at locked clocks -------------------------
    # --noDataTransfers isolates GPU execution (host I/O budgeted separately
    # via the link budgets). trtexec prints mean/median/percentile latencies.
    # NOTE: trtexec --warmUp is in MILLISECONDS (2000 = 2 s of warmup).
    say "timing: 1000 iterations..."
    trtexec --loadEngine="$ENGINE" $EXTRA --iterations=1000 --warmUp=2000 \
            --percentile=99 --noDataTransfers \
      > "$MDIR/timing.log" 2>&1 &
    TRT_PID=$!
    # Under-load clock evidence: the device-aware sampler follows the trtexec
    # PID and exits with it. This replaces the old inline nvidia-smi loop —
    # same discipline, but it also records power/thermal/throttle context and
    # feeds the quantitative drift verdict below.
    python3 "$KIT/common/clock_sampler.py" --device "$DEVICE_CFG" --pid "$TRT_PID" \
      --out "$MDIR/clock_samples.csv" --phase "timing" &
    SAMPLER_PID=$!
    # peak-VRAM sampler (per-process VRAM query returns nothing on Jetson
    # iGPUs — file stays empty there; stage 5 falls back to the TRT-managed
    # parse)
    : > "$MDIR/vram_samples.txt"
    while kill -0 "$TRT_PID" 2>/dev/null; do
      nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null \
        | awk -F', ' -v p="$TRT_PID" '$1==p {print $2}' >> "$MDIR/vram_samples.txt" || true
      sleep 0.1
    done
    wait "$TRT_PID" || say "WARN: timing failed for $NAME"
    wait "$SAMPLER_PID" 2>/dev/null || say "WARN: clock sampler exited abnormally for $NAME"
    # Quantitative drift verdict over the under-load samples — the evidence
    # the lock held THROUGH the measurement (start/end reads are idle-state
    # artifacts and prove nothing).
    python3 "$KIT/common/drift_report.py" "$MDIR/clock_samples.csv" \
      --device "$DEVICE_CFG" --lock "$OUT_DIR/provenance/lock_verified.json" \
      --out "$MDIR/drift_timing.json" --phase "timing" \
      || say "WARN: drift report failed for $NAME — clock integrity unrecorded for this model"

    # -- 3b2. Timeline capture via Nsight Systems (per-kernel durations) ------
    if command -v nsys >/dev/null; then
      say "nsys timeline (60 iterations)..."
      nsys profile -o "$MDIR/nsys_timeline" --force-overwrite=true \
        trtexec --loadEngine="$ENGINE" $EXTRA --iterations=60 --warmUp=500 --noDataTransfers \
        > "$MDIR/nsys.log" 2>&1 &
      NSYS_PID=$!
      python3 "$KIT/common/clock_sampler.py" --device "$DEVICE_CFG" --pid "$NSYS_PID" \
        --out "$MDIR/clock_samples_nsys.csv" --phase "nsys" &
      NSYS_SAMPLER=$!
      wait "$NSYS_PID" || say "WARN: nsys failed for $NAME"
      wait "$NSYS_SAMPLER" 2>/dev/null || true
    fi

    # -- 3c. Per-kernel counters via Nsight Compute ---------------------------
    # One inference, every kernel, hardware counters:
    #   dram__bytes.sum                         -> bytes/frame (budget 2)
    #   sm__inst_executed_pipe_tensor.sum       -> tensor-core work detected
    #   sm__sass_thread_inst_executed_op_*      -> CUDA-core FLOPs (budget 4)
    #   gpu__time_duration.sum                  -> per-kernel time weighting
    # Timing inside ncu is replay-distorted; the p99 of record stays 3b's.
    # NCU manages its own clocks — no sampler on this pass by design.
    if [ "${SKIP_NCU:-0}" != 1 ] && command -v ncu >/dev/null; then
      say "ncu counter pass (this is the slow step)..."
      # Warm-cache collection policy (pre-declared): NCU's default flushes
      # caches around every replay pass, so on big-cache parts the byte
      # counters describe an all-cold state no deployed frame ever sees —
      # observed 1291 MB/frame on a GEMM whose physical bound (p99 x bw) was
      # 996 MB and whose unique traffic is ~100 MB. --cache-control none
      # keeps caches live, and NCU_ITERS engine executions amortize the one
      # genuinely cold first frame; the parser divides by NCU_ITERS (recorded
      # in ncu_meta.json). Stage 5 additionally clamps to the physical bound.
      NCU_ITERS="${NCU_ITERS:-5}"
      NCU_ARGS=(--csv --target-processes all -f --cache-control none)
      if [ "${NCU_FULL:-0}" = 1 ]; then
        NCU_ARGS+=(--set full)
      else
        # dram__bytes is n/a on Jetson iGPUs (no FB counters); lts__t_bytes (L2<->
        # memory-side bytes from the SM view) is the byte proxy there — request
        # both, unsupported ones read n/a and the parser takes what's populated
        NCU_ARGS+=(--metrics dram__bytes.sum,lts__t_bytes.sum,lts__t_sectors_lookup_miss.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,gpu__time_duration.sum,sm__inst_executed_pipe_tensor.sum,sm__sass_thread_inst_executed_op_ffma_pred_on.sum,sm__sass_thread_inst_executed_op_fadd_pred_on.sum,sm__sass_thread_inst_executed_op_fmul_pred_on.sum,sm__sass_thread_inst_executed_op_hfma_pred_on.sum,sm__sass_thread_inst_executed_op_hadd_pred_on.sum,sm__sass_thread_inst_executed_op_hmul_pred_on.sum)
      fi
      # sudo resets PATH (secure_path): resolve both binaries to absolute paths
      NCU_BIN=$(command -v ncu); TRTEXEC_BIN=$(command -v trtexec)
      # sudo resets the environment: LD_LIBRARY_PATH must be re-passed or the
      # profiled trtexec can't load libnvinfer* from non-ldconfig TRT installs
      printf '{"iterations": %s, "cache_control": "none"}\n' "$NCU_ITERS" > "$MDIR/ncu_meta.json"
      sudo LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" "$NCU_BIN" "${NCU_ARGS[@]}" --log-file "$MDIR/ncu_kernels.csv" \
        "$TRTEXEC_BIN" --loadEngine="$ENGINE" $EXTRA --iterations="$NCU_ITERS" --duration=0 --warmUp=0 --noDataTransfers \
        > "$MDIR/ncu.log" 2>&1 || say "WARN: ncu failed for $NAME (driver perms? try: sudo, check $MDIR/ncu.log)"
    else
      say "ncu skipped for $NAME"
    fi
  done
}

#=============================================================================
# STAGE 4 — CLOCK-DIAL EXPERIMENT (optional; discrete GPUs only)
#
# What we are doing: the cheap causal cross-check on boundness. Downclock the
# MEMORY clock ~20% and re-time; then the SM clock ~20% and re-time. Whichever
# dial end-to-end latency tracks is the binding resource — integrated over
# every kernel, no profiler in the loop. Sensitivity ~1.0 = fully bound to
# that resource; both ~0 = latency-bound (launch overhead tracks no clock).
# $EXTRA rides on both invocations — plugin models refuse to load without it.
#=============================================================================
stage4_clock_dial() {
  [ "${RUN_CLOCK_DIAL:-0}" = 1 ] || { say "=== STAGE 4: clock-dial (skipped; RUN_CLOCK_DIAL=1 to enable) ==="; return; }
  if [ "$PLATFORM" = jetson ]; then
    say "=== STAGE 4: clock-dial — on Jetson the EMC devfreq dial lives in model_bench/measure_dram_demand.sh (own restore trap); skipping here ==="
    return
  fi
  say "=== STAGE 4: clock-dial experiment ==="
  MAXGC=$(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits | head -1)
  MAXMC=$(nvidia-smi --query-gpu=clocks.max.memory   --format=csv,noheader,nounits | head -1)
  # dial from (and restore to) the same holdable memory bin stage 1 locked
  CFG_MEMLOCK=$(python3 -c 'import json,sys;v=json.load(open(sys.argv[1])).get("mem_lock_mhz");print(v if v else "")' "$DEVICE_CFG")
  [ -n "$CFG_MEMLOCK" ] && MAXMC="$CFG_MEMLOCK"
  LOWGC=$(( MAXGC * 80 / 100 )); LOWMC=$(( MAXMC * 80 / 100 ))
  tail -n +2 "$MIX_FILE" | while IFS=, read -r NAME ONNX PREC HZ DEADLINE AGF EXTRA; do
    [ -z "$NAME" ] && continue
    [[ "$ONNX" == *.engine ]] || continue
    ENGINE="$ONNX"
    [ -f "$ENGINE" ] || continue
    say "--- clock dial: $NAME ---"
    sudo nvidia-smi -lmc "$LOWMC" >/dev/null   # memory dial down 20%
    trtexec --loadEngine="$ENGINE" $EXTRA --iterations=300 --warmUp=1000 --percentile=99 --noDataTransfers \
      > "$OUT_DIR/models/$NAME/timing_memdown.log" 2>&1 || true
    sudo nvidia-smi -lmc "$MAXMC" >/dev/null; sudo nvidia-smi -lgc "$LOWGC" >/dev/null  # SM dial down 20%
    trtexec --loadEngine="$ENGINE" $EXTRA --iterations=300 --warmUp=1000 --percentile=99 --noDataTransfers \
      > "$OUT_DIR/models/$NAME/timing_smdown.log" 2>&1 || true
    sudo nvidia-smi -lgc "$MAXGC" >/dev/null   # restore
  done
}

#=============================================================================
# STAGE 5 — POST-PROCESSING: budgets, U_max, C, L, N, Score
#
# What we are doing: turning raw logs into the report numbers.
#   per model : p99 (3b), bytes/frame + FLOPs split + time-weighted
#               tensor/CUDA/memory mix (3c), boundness verdicts, clock-drift
#               integrity (drift FAIL = numbers are throttle artifacts —
#               excluded from every device sum, run flagged invalid)
#   per device: budget 1 time occupancy    = sum(p99 x Hz)
#               budget 2 bandwidth         = sum(bytes x Hz) / bw_eff, where
#               bw_eff = IDLE copy_RW best from CEILINGS_JSON (solo-exclusive
#               regime: no CPU contention assumed, one model at a time)
#               U_max  = binding utilization,  C = 1/U_max
#               L      = min(deadline / p99) across models
#               N      = min(L, C)  with the throughput/latency cause tag
#               Score  = N x sum(arch_GFLOPs x Hz)   [workload constant from
#                        the manifest — architectural, never profiler FLOPs]
#=============================================================================
stage5_report() {
  say "=== STAGE 5: post-processing and report ==="
  python3 - "$MIX_FILE" "$OUT_DIR" "$DEVICE_CFG" "$CEILINGS_JSON" <<'PYEOF'
import csv, json, os, re, statistics, sys
mix_file, out_dir, cfg_path, ceilings_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

def pick(d, *keys, default=None):
    """First present, non-None value — tolerant reads across producer schemas."""
    for k in keys:
        if isinstance(d, dict) and d.get(k) is not None: return d[k]
    return default

def load_json(path):
    if not path or not os.path.exists(path): return None
    try: return json.load(open(path))
    except Exception as ex:
        print(f'WARN: unreadable JSON {path}: {ex}', file=sys.stderr); return None

def fin(x, nd):
    """Round for the report; infinities become null (empty/invalid-mix runs)."""
    try:
        if x is None or x != x or abs(x) == float('inf'): return None
        return round(x, nd)
    except TypeError:
        return None

cfg = load_json(cfg_path) or {}
platform = cfg.get('platform', 'discrete')

# ---- capacity side: mapped keys from the thorough ceilings suite ------------
# The pipeline never measures ceilings; CEILINGS_JSON (measure_ceilings_thorough
# output) is the source of record. bw_eff = IDLE best copy_RW — the
# solo-exclusive regime budgets against the uncontended bus, unlike the July
# run's CPU-loaded figure.
cj = load_json(ceilings_path)
if cj is None:
    print(f'FATAL: CEILINGS_JSON unreadable: {ceilings_path}', file=sys.stderr); sys.exit(1)
gs = cj.get('gemm_sweep') or {}
fp16_best = fp16_variant = None
int8_best = int8_at = None
for k, row in gs.items():
    if not isinstance(row, dict): continue          # fp8 scalar rides in the same dict
    for acc_key, tag in (('fp16_fp16acc_tflops', 'fp16acc'), ('fp16_fp32acc_tflops', 'fp32acc')):
        cell = row.get(acc_key)
        b = cell.get('best') if isinstance(cell, dict) else None
        if isinstance(b, (int, float)) and (fp16_best is None or b > fp16_best):
            fp16_best, fp16_variant = b, f'{tag}@n={k}'
    cell = row.get('int8_tops')
    b = cell.get('best') if isinstance(cell, dict) else None
    if isinstance(b, (int, float)) and (int8_best is None or b > int8_best):
        int8_best, int8_at = b, f'n={k}'
fp8_raw = gs.get('fp8_n4096_tflops')
fp8_best = fp8_raw if isinstance(fp8_raw, (int, float)) else None
fp8_note = fp8_raw if isinstance(fp8_raw, str) else None   # 'unavailable: ...'
cuda_fp32 = cj.get('cuda_fp32')
bw_best = None
for row in (cj.get('bandwidth') or {}).values():
    v = row.get('copy_RW') if isinstance(row, dict) else None
    if isinstance(v, (int, float)) and (bw_best is None or v > bw_best): bw_best = v
sust = [s.get('tflops') for s in (cj.get('sustained') or [])
        if isinstance(s, dict) and isinstance(s.get('tflops'), (int, float))]
sustained_med = round(statistics.median(sust), 1) if sust else None
cmeta = cj.get('meta') or {}
ceilings = {
    'tensor_ceiling_fp16_tflops': fp16_best,
    'tensor_fp16_best_variant': fp16_variant,     # which accumulate path + size won
    'tensor_ceiling_int8_tops': int8_best,
    'tensor_int8_best_at': int8_at,
    'tensor_ceiling_fp8_tflops': fp8_best,        # null when the probe was unavailable
    'cudacore_fp32_tflops': cuda_fp32,
    'bw_eff_gbps': bw_best,                       # budget-2 denominator (idle basis)
    'dram_gbps_idle': bw_best,
    'bw_regime': 'idle-exclusive',
    'sustained_tflops_median': sustained_med,
    # suite 6 runs fp16 unless the operator exported SUSTAIN_PREC; the raw json
    # does not record it, so the script default is the honest assumption
    'sustained_prec': cmeta.get('sustain_prec', 'fp16'),
    'ceilings_source': {'path': os.path.abspath(ceilings_path), 'start': cmeta.get('start')},
}
if fp8_note: ceilings['tensor_fp8_note'] = fp8_note
# The TRT GEMM-kernel probe (trt_compute.json, written next to results.json by
# measure_compute_trt.py) is the tensor ceiling of record when present: typed
# pure-GEMM engines with per-kernel time isolation and graph-form variant
# search. The torch sweep numbers stay recorded as cross-checks — on sm_120
# they understate badly (int8 128.8 vs 431.1 TOPS: the typed torch path picks
# weaker kernels than TRT's tactic search).
tj = load_json(os.path.join(os.path.dirname(os.path.abspath(ceilings_path)), 'trt_compute.json'))
if tj and isinstance(tj.get('precisions'), dict):
    for prec, key, vkey in (('fp16', 'tensor_ceiling_fp16_tflops', 'tensor_fp16_best_variant'),
                            ('int8', 'tensor_ceiling_int8_tops',   'tensor_int8_best_at'),
                            ('fp8',  'tensor_ceiling_fp8_tflops',  None)):
        blk = tj['precisions'].get(prec) or {}
        b = blk.get('best')
        if isinstance(b, (int, float)):
            if ceilings.get(key) is not None:
                ceilings['torch_sweep_' + key] = ceilings[key]
            ceilings[key] = b
            if vkey:
                ceilings[vkey] = 'trt@%s@n=%s' % (blk.get('best_method') or 'gemm', blk.get('at_n'))
    f32 = (tj['precisions'].get('fp32') or {}).get('best')
    if isinstance(f32, (int, float)):
        ceilings['fp32_trt_tflops'] = f32   # TRT --noTF32 GEMM; cudacore_fp32 stays the non-matmul roof
    ceilings['tensor_ceilings_source'] = 'trt-gemm-probe'
else:
    ceilings['tensor_ceilings_source'] = 'torch-gemm-sweep (no trt_compute.json beside CEILINGS_JSON)'
bw_eff = bw_best
if bw_eff is None:
    print('WARN: no bandwidth suite in CEILINGS_JSON — dram_bandwidth budget omitted', file=sys.stderr)

def parse_trtexec_latency(path):
    """p99 of record comes ONLY from the GPU Compute Time block. A log without
    it is a failed measurement, not a soft fallback — the old silent
    mean-as-p99 substitution shipped optimistic numbers once. P99_ALLOW_MEAN=1
    is the explicit escape hatch, and it is recorded as degraded."""
    if not os.path.exists(path):
        print(f'FATAL: timing log missing: {path} — the model was never timed (stage 3 failure?); '
              f'remove it from the mix or re-run', file=sys.stderr)
        sys.exit(1)
    txt = open(path, errors='ignore').read()
    m = re.search(r'GPU Compute Time.*?mean = ([\d.]+) ms.*?percentile\(99%\) = ([\d.]+) ms', txt, re.S)
    if m:
        return {'mean_ms': float(m.group(1)), 'p99_ms': float(m.group(2)),
                'p99_source': 'gpu-compute-time'}
    if os.environ.get('P99_ALLOW_MEAN') == '1':
        m = re.search(r'mean = ([\d.]+) ms', txt)
        if m:
            return {'mean_ms': float(m.group(1)), 'p99_ms': float(m.group(1)),
                    'p99_source': 'mean-fallback-degraded'}
    print(f'FATAL: no "GPU Compute Time" block in {path} — timing is invalid '
          f'(P99_ALLOW_MEAN=1 accepts mean-as-p99, recorded as degraded)', file=sys.stderr)
    sys.exit(1)

def parse_ncu_csv(path):
    """Sum per-kernel counters: DRAM bytes, tensor-pipe insts, CUDA-core FLOPs,
    and time-weighted tensor vs CUDA-core split. The NCU pass runs the engine
    ncu_meta.json['iterations'] times (warm-cache amortization) — every
    extensive total is divided by that count to yield per-frame values; a
    missing meta file means a legacy single-iteration artifact (divisor 1)."""
    if not os.path.exists(path): return None
    ncu_iters = 1
    meta_path = os.path.join(os.path.dirname(path), 'ncu_meta.json')
    if os.path.exists(meta_path):
        try: ncu_iters = max(1, int(json.load(open(meta_path)).get('iterations', 1)))
        except (ValueError, OSError): pass
    rows = []
    with open(path, errors='ignore') as f:
        for line in f:
            if line.startswith('"ID"') or line.startswith('ID,'):
                rows = list(csv.DictReader([line] + f.readlines())); break
    if not rows: return None
    agg = {'dram_bytes': 0.0, 'l2_bytes': 0.0, 'l2_miss_bytes': 0.0,
           'tensor_inst': 0.0, 'cuda_flops': 0.0,
           'time_tensor_ns': 0.0, 'time_cuda_ns': 0.0, 'kernels': 0}
    per_kernel = {}
    for r in rows:
        k = r.get('Kernel Name', '?'); metric = r.get('Metric Name', '');
        try: val = float(str(r.get('Metric Value', '0')).replace(',', ''))
        except ValueError: continue
        d = per_kernel.setdefault(k + '#' + r.get('ID', ''), {})
        d[metric] = d.get(metric, 0) + val
    for k, d in per_kernel.items():
        agg['kernels'] += 1
        # dram__bytes exists on discrete GPUs only. On Jetson iGPUs keep the L2
        # counters SEPARATE: lts__t_bytes counts hits too (SM<->L2 traffic, not
        # DRAM) — budgeting with it produced impossible >ceiling demand. The
        # miss-side sectors x32B approximate DRAM traffic; both stay labeled.
        agg['dram_bytes'] += d.get('dram__bytes.sum', 0)
        agg['l2_bytes'] += d.get('lts__t_bytes.sum', 0)
        agg['l2_miss_bytes'] += 32*d.get('lts__t_sectors_lookup_miss.sum', 0)
        t_inst = d.get('sm__inst_executed_pipe_tensor.sum', 0)
        agg['tensor_inst'] += t_inst
        # CUDA-core FLOPs: FMA counts twice; packed fp16 (half2) counts x2 lanes
        f32 = 2*d.get('sm__sass_thread_inst_executed_op_ffma_pred_on.sum',0) \
              + d.get('sm__sass_thread_inst_executed_op_fadd_pred_on.sum',0) \
              + d.get('sm__sass_thread_inst_executed_op_fmul_pred_on.sum',0)
        f16 = 2*(2*d.get('sm__sass_thread_inst_executed_op_hfma_pred_on.sum',0) \
              + d.get('sm__sass_thread_inst_executed_op_hadd_pred_on.sum',0) \
              + d.get('sm__sass_thread_inst_executed_op_hmul_pred_on.sum',0))
        agg['cuda_flops'] += f32 + f16
        dur = d.get('gpu__time_duration.sum', 0)
        if t_inst > 0: agg['time_tensor_ns'] += dur
        else:          agg['time_cuda_ns']   += dur
    if ncu_iters > 1:
        for k in agg: agg[k] /= ncu_iters
        agg['kernels'] = round(agg['kernels'])
    agg['ncu_iterations'] = ncu_iters
    return agg

# ---- VRAM capacity (budget 3 denominator) -----------------------------------
# Precedence: explicit env override, then the device config's policy cap
# (a 48 GB policy cap on a 72 GB card), then physical totals.
if os.environ.get('VRAM_CAPACITY_MB'):
    vram_capacity_mb = float(os.environ['VRAM_CAPACITY_MB']); vram_capacity_source = 'env-override'
elif isinstance(cfg.get('vram_budget_cap_mb'), (int, float)):
    vram_capacity_mb = float(cfg['vram_budget_cap_mb']); vram_capacity_source = 'device-config-cap'
elif platform == 'jetson':
    # unified memory: system MemTotal, shared with the OS — the budget
    # threshold does the safety-margin work
    vram_capacity_mb, vram_capacity_source = None, 'unavailable'
    try:
        for line in open('/proc/meminfo'):
            if line.startswith('MemTotal'):
                vram_capacity_mb = float(line.split()[1]) / 1024.0
                vram_capacity_source = 'meminfo-unified'; break
    except Exception: pass
else:
    vram_capacity_mb, vram_capacity_source = None, 'unavailable'
    try:
        out = os.popen('nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null').read().strip().splitlines()
        if out and out[0].strip().replace('.','').isdigit():
            vram_capacity_mb = float(out[0]); vram_capacity_source = 'nvidia-smi-total'
    except Exception: pass

# ---- provenance blocks: preflight + lock verification -----------------------
pf = load_json(f'{out_dir}/provenance/preflight.json')
lv = load_json(f'{out_dir}/provenance/lock_verified.json')
smoke_only = bool(pick(pf or {}, 'smoke_only', default=False))
pf_verdict = pick(pf or {}, 'verdict')
idle_mean = pick(pf or {}, 'idle_power_w_mean')
if idle_mean is None and isinstance((pf or {}).get('idle_baseline'), dict):
    idle_mean = pick(pf['idle_baseline'], 'power_w_mean', 'idle_power_w_mean', 'mean_w')
headless = pick(pf or {}, 'headless_verified', 'headless')
if headless is None and pf is not None:
    headless = (str(pf_verdict).upper() in ('PASS', 'WARN', 'OK')) and not smoke_only
preflight_block = {'verdict': pf_verdict, 'idle_power_w_mean': idle_mean, 'smoke_only': smoke_only}
lock_verdict = pick(lv or {}, 'verdict')
regime = {'headless_verified': headless, 'contention': 'solo-exclusive',
          'bw_ceiling_basis': 'idle', 'clock_policy': 'locked-max-except-emc-dial'}

models, work_const = [], 0.0
with open(mix_file) as f:
    for row in csv.DictReader(f):
        name = row['name'].strip()
        hz, dl = float(row['hz']), float(row['deadline_ms'])
        agf = float(row.get('arch_gflops') or 0)
        work_const += agf * hz  # GFLOP/s the mix inherently requires
        t = parse_trtexec_latency(f'{out_dir}/models/{name}/timing.log')
        n = parse_ncu_csv(f'{out_dir}/models/{name}/ncu_kernels.csv')
        m = {'name': name, 'hz': hz, 'deadline_ms': dl, 'arch_gflops': agf,
             'precision': (row.get('precision') or '').strip() or None}
        # VRAM footprint (budget 3 term): TRT-managed weights + activation
        # memory from the timing log — the steady-state per-engine residency
        tlog = f'{out_dir}/models/{name}/timing.log'
        if os.path.exists(tlog):
            txt = open(tlog, errors='ignore').read()
            w = re.search(r'Loaded engine size: ([\d.]+) MiB', txt)
            a = re.search(r'device memory size: ([\d.]+) MiB', txt)
            if w or a:
                m['vram_weights_mb'] = float(w.group(1)) if w else 0.0
                m['vram_activation_mb'] = float(a.group(1)) if a else 0.0
                m['vram_mb'] = round(m['vram_weights_mb'] + m['vram_activation_mb'], 1)
        # process-peak sampler output: budget with the peak (context + I/O +
        # scratch included) when available
        vs = f'{out_dir}/models/{name}/vram_samples.txt'
        if os.path.exists(vs):
            try:
                peaks = [float(x) for x in re.findall(r'([\d.]+)', open(vs).read())]
                if peaks:
                    m['vram_peak_mb'] = max(peaks)
                    m['vram_source'] = 'process-peak-sampled'
            except Exception: pass
        m['vram_budget_mb'] = m.get('vram_peak_mb') or m.get('vram_mb')
        # under-load clock evidence from common/clock_sampler.py: header row,
        # jetson columns gpu_mhz/emc_mhz, discrete sm_mhz/mem_mhz; blank cells
        # are unreadable-source samples and drop out here
        cspath = f'{out_dir}/models/{name}/clock_samples.csv'
        if os.path.exists(cspath):
            with open(cspath) as cf:
                rdr = csv.DictReader(cf)
                fields = rdr.fieldnames or []
                smc = 'sm_mhz' if 'sm_mhz' in fields else 'gpu_mhz'
                memc = 'mem_mhz' if 'mem_mhz' in fields else 'emc_mhz'
                sm_v, mem_v = [], []
                for cr in rdr:
                    for col, acc in ((smc, sm_v), (memc, mem_v)):
                        try: acc.append(float(cr.get(col) or ''))
                        except (TypeError, ValueError): pass
            if sm_v: m['sm_clock_mhz_during_run'] = {'min': min(sm_v), 'max': max(sm_v)}
            if mem_v: m['mem_clock_mhz_during_run'] = {'min': min(mem_v), 'max': max(mem_v)}
        # quantitative drift verdict for the timing pass (drift_report.py)
        ci = None
        d = load_json(f'{out_dir}/models/{name}/drift_timing.json')
        if d:
            ci = {'verdict': pick(d, 'verdict'),
                  'pct_at_target': pick(d, 'pct_at_target'),
                  'reference_clock_mhz': pick(d, 'reference_clock_mhz'),
                  'throttle_reasons_seen': pick(d, 'throttle_reasons_seen', 'throttle_reasons'),
                  'clamp_events': pick(d, 'clamp_events', 'clamp_event_count', 'oc_clamp_events')}
        m['clock_integrity'] = ci or {'verdict': None,
            'note': 'drift_timing.json missing — no under-load clock evidence recorded'}
        # drift FAIL invalidates THIS model's numbers: excluded from device
        # budget sums and from L (a throttled p99 is not evidence); WARN stands
        m['measurement_valid'] = not (ci and ci.get('verdict') == 'FAIL')
        if t:
            m.update(t)
            m['time_share'] = t['p99_ms'] / 1e3 * hz          # budget 1 term
            m['deadline_margin'] = dl / t['p99_ms']           # L term
        # causal EMC-dial bytes (fit_dram_demand.py) trump every profiler proxy:
        # measured by re-timing at multiple memory clocks, no replay artifacts
        causal = None
        dd_path = os.environ.get('DRAM_DEMAND_JSON', '')
        if dd_path and os.path.exists(dd_path):
            dd = json.load(open(dd_path))
            if name in dd and 'dram_bytes_per_frame_gb' in dd[name]:
                causal = dd[name]['dram_bytes_per_frame_gb'] * 1e9
        if causal is not None:
            m['bytes_per_frame_MB'] = causal / 1e6
            m['bw_demand_gbps'] = causal * hz / 1e9
            m['bytes_source'] = 'causal-emc-dial'
        elif n:
            # budget ONLY with true DRAM bytes (or the miss-side approximation,
            # flagged); raw L2 traffic is hit-inflated and never budgeted
            dram = n['dram_bytes'] or n['l2_miss_bytes']
            m['bytes_per_frame_MB'] = dram / 1e6
            m['bw_demand_gbps'] = dram * hz / 1e9             # budget 2 term
            m['bytes_source'] = 'dram' if n['dram_bytes'] else ('l2-miss-approx' if n['l2_miss_bytes'] else 'none')
        # Physical bound (pre-declared): a frame cannot move more DRAM bytes
        # than the bus carries in the frame's own duration — bytes <= p99 x
        # bw_eff (GB/s == MB/ms, so the product is directly MB). Counter
        # values above the bound are collection artifacts (replay/cold-cache;
        # observed 1291 MB against a 996 MB bound on a pure GEMM): the budget
        # uses the clamped bound — itself an overestimate, so still safe as
        # an upper bound — and the raw value stays recorded for the audit.
        if bw_eff and m.get('p99_ms') and m.get('bytes_per_frame_MB') is not None:
            bound_mb = m['p99_ms'] * bw_eff
            m['bytes_physical_bound_MB'] = round(bound_mb, 1)
            if m['bytes_per_frame_MB'] > bound_mb:
                m['bytes_counter_raw_MB'] = round(m['bytes_per_frame_MB'], 1)
                m['bytes_bound_violation'] = True
                m['bytes_per_frame_MB'] = round(bound_mb, 1)
                m['bw_demand_gbps'] = bound_mb * hz / 1e3
                m['bytes_source'] = m.get('bytes_source', 'none') + '-clamped-to-physical-bound'
                print(f"WARN: {name}: bytes/frame {m['bytes_counter_raw_MB']} MB exceeds the "
                      f"physical bound {round(bound_mb,1)} MB (p99 x bw_eff) — counter artifact; "
                      f"budgeting with the bound", file=sys.stderr)
        if n:
            m['l2_bytes_per_frame_MB'] = n['l2_bytes'] / 1e6  # proxy, labeled
            m['cuda_gflops_per_frame'] = n['cuda_flops'] / 1e9
            tt, tc = n['time_tensor_ns'], n['time_cuda_ns']
            m['pct_time_tensor_kernels'] = round(100*tt/(tt+tc), 1) if tt+tc else None
        models.append(m)

# ---- device-level budgets -> U_max, C, L, N, Score --------------------------
# per-model SOLO verdict: the model alone on the device, copies = 1
for m in models:
    b = {'time_occupancy': m.get('time_share', 0)}
    if bw_eff and m.get('bw_demand_gbps') is not None:
        b['dram_bandwidth'] = m['bw_demand_gbps'] / bw_eff
    if vram_capacity_mb and m.get('vram_budget_mb'):
        b['vram_footprint'] = m['vram_budget_mb'] / vram_capacity_mb
    if any(v > 0 for v in b.values()) or m.get('hz') == 0:
        # hz=0 (modal / on-demand): no sustained time or bandwidth bill; the
        # verdict is latency-driven (L vs the interaction deadline) plus any
        # resident-footprint axis. Contributes ~0 to the mix Score by design.
        um = max((v for v in b.values() if v > 0), default=0.0)
        c = (1/um) if um > 0 else float('inf')
        l = m.get('deadline_margin', float('inf'))
        n = min(l, c); w = m['arch_gflops'] * m['hz'] / 1e3
        m['solo'] = {'budgets': {k: round(v, 4) for k, v in b.items()},
                     'U_max': round(um, 4), 'C': round(c, 3) if c != float('inf') else None,
                     'L': round(l, 3), 'N': round(n, 3),
                     'cause': 'latency-limited' if l < c else 'throughput-limited',
                     'score_tflops': round(n * w, 3)}
        if m.get('hz') == 0:
            m['solo']['score_tflops'] = None   # modal: Score prices sustained work; this model is latency-gated, not scored
            m['solo']['modal'] = 'hz=0: no sustained time/bandwidth bill; N is the verdict (verify L against the CONTENDED p99: solo p99 / (1 - resident U_time), or measured against the live mix); resident VRAM still bills the mix budgets'

# drift-FAIL models are excluded from every device sum: their p99/bytes are
# throttle artifacts, not demand. They stay in models[] flagged invalid.
valid = [m for m in models if m.get('measurement_valid', True)]
excluded = [m['name'] for m in models if not m.get('measurement_valid', True)]
U_time = sum(m.get('time_share', 0) for m in valid)
U_bw   = (sum(m.get('bw_demand_gbps', 0) for m in valid) / bw_eff) if bw_eff else None
U_vram = (sum(m.get('vram_budget_mb') or 0 for m in valid) / vram_capacity_mb) if vram_capacity_mb else None
bw_upper_bound = any(m.get('bytes_source') == 'l2-miss-approx' for m in valid)
budgets = {'time_occupancy': U_time}
if U_bw is not None: budgets['dram_bandwidth'] = U_bw
if U_vram is not None: budgets['vram_footprint'] = U_vram
U_max, binding = max(((v, k) for k, v in budgets.items())), None
U_max, binding = U_max[0], U_max[1]
C = (1 / U_max) if U_max > 0 else float('inf')
margins = [m['deadline_margin'] for m in valid if 'deadline_margin' in m]
L = min(margins) if margins else float('inf')
N = min(L, C)
cause = 'latency-limited' if L < C else 'throughput-limited'
score = N * work_const / 1e3   # TFLOP/s of deadline-respecting useful work

# ---- clock-integrity rollup -------------------------------------------------
rank = {'PASS': 1, 'WARN': 2, 'FAIL': 3}
known = [m['clock_integrity'].get('verdict') for m in models
         if isinstance(m.get('clock_integrity'), dict)
         and m['clock_integrity'].get('verdict') in rank]
worst_model_verdict = max(known, key=lambda v: rank[v]) if known else None
run_valid = (not excluded) and (lock_verdict != 'FAIL') and (not smoke_only)
clock_integrity_top = {'lock_verify': lock_verdict,
                       'worst_model_verdict': worst_model_verdict,
                       'run_valid': run_valid,
                       'excluded_models': excluded}
json.dump({'lock_verify': lock_verdict, 'worst_model_verdict': worst_model_verdict,
           'run_valid': run_valid,
           'per_model': {m['name']: m.get('clock_integrity') for m in models}},
          open(f'{out_dir}/report/clock_drift.json', 'w'), indent=2)

# ---- accuracy gate ingestion (schema-tolerant) ------------------------------
acc_path = os.environ.get('ACCURACY_GATE_JSON', '')
acc = load_json(acc_path) if acc_path else None
if acc_path and acc is None:
    print(f'WARN: ACCURACY_GATE_JSON set but unreadable: {acc_path}', file=sys.stderr)
acc_block, acc_rows, acc_fail = None, [], set()
if acc:
    def norm_gate(x):
        if isinstance(x, bool): return 'PASS' if x else 'FAIL'
        return str(x).upper() if x is not None else None
    def agree_hits(node, label, hits, pinned=False):
        # collect every agree_frac with its variant label; an explicit
        # 'variant' field pins the label for its whole subtree (so per-output
        # keys below it do not overwrite it), otherwise dict keys are the
        # best-effort labels — the walk is schema-agnostic on purpose
        if isinstance(node, dict):
            v = node.get('variant')
            if isinstance(v, str):
                label, pinned = v, True
            elif not pinned:
                label = pick(node, 'label', 'name', default=label)
            if isinstance(node.get('agree_frac'), (int, float)):
                hits.append((node['agree_frac'], label))
            for k, val in node.items():
                child = label if pinned else (k if isinstance(val, (dict, list)) else label)
                agree_hits(val, child, hits, pinned)
        elif isinstance(node, list):
            for val in node: agree_hits(val, label, hits, pinned)
    def summarize(node, cand):
        hits = []
        agree_hits(node, cand, hits)
        worst = min(hits, key=lambda h: h[0]) if hits else (None, None)
        return {'candidate': cand,
                'gate': norm_gate(pick(node, 'gate', 'gate_verdict', 'verdict', 'gate_pass')),
                'n_variants': len({h[1] for h in hits}) or None,
                'worst_variant': worst[1], 'worst_agree_frac': worst[0]}
    cands = acc.get('candidates')
    # accuracy-gate/v1 keeps the per-candidate rollup under summary{}, not
    # candidates{} (which carries only engine+sha) — prefer it; the recursive
    # walker survives as the fallback for foreign schemas only
    summ_map = acc.get('summary') if isinstance(acc.get('summary'), dict) else {}
    if isinstance(cands, dict) and cands:
        for cname, sub in cands.items():
            sm = summ_map.get(cname)
            if isinstance(sm, dict) and 'variants_total' in sm:
                s = {'candidate': str(cname),
                     'gate': norm_gate(sm.get('gate')),
                     'n_variants': sm.get('variants_total'),
                     'worst_variant': sm.get('worst_variant'),
                     'worst_agree_frac': sm.get('worst_agree_frac')}
            else:
                s = summarize(sub, str(cname))
            s['gate'] = s['gate'] or norm_gate(pick(acc, 'gate', 'gate_verdict', 'verdict', 'gate_pass'))
            acc_rows.append(s)
    else:
        cname = pick(acc, 'candidate', 'model', 'name', default='candidate')
        acc_rows.append(summarize(acc, str(cname)))
    for s in acc_rows:
        if s['gate'] == 'FAIL': acc_fail.add(s['candidate'])
    acc_block = {'gate': norm_gate(pick(acc, 'gate', 'gate_verdict', 'verdict', 'gate_pass')),
                 'source': os.path.abspath(acc_path),
                 'per_candidate': acc_rows, 'detail': acc}

result = {'suite': 'per-model-solo', 'convention': 'solo',
          'regime': regime,
          'preflight': preflight_block,
          'clock_integrity': clock_integrity_top,
          'ceilings': ceilings}
if acc_block is not None: result['accuracy'] = acc_block
result.update({'budgets': budgets, 'binding_budget': binding,
          'bw_ceiling_source': 'thorough-ceilings idle copy_RW best' if bw_eff else None,
          'bw_demand_is_upper_bound': bw_upper_bound or None,
          'vram_capacity_mb': vram_capacity_mb,
          'vram_capacity_source': vram_capacity_source,
          'U_max': round(U_max, 4), 'C': fin(C, 3), 'L': fin(L, 3),
          'N': fin(N, 3), 'shortfall_cause_if_N_lt_1': cause,
          'workload_constant_gflops_per_s': round(work_const, 1),
          'score_tflops_at_deadline': fin(score, 2), 'models': models})
json.dump(result, open(f'{out_dir}/report/results.json', 'w'), indent=2)

# ---- human-readable report --------------------------------------------------
def fmt(x, p):
    return 'n/a' if (x is None or x != x or abs(x) == float('inf')) else f'{x:.{p}f}'
with open(f'{out_dir}/report/report.md', 'w') as r:
    r.write('# Measurement report (this device)\n\n')
    r.write(f"Regime: suite=per-model-solo | contention={regime['contention']} | "
            f"bw ceiling basis={regime['bw_ceiling_basis']} | clock policy={regime['clock_policy']} | "
            f"headless_verified={regime['headless_verified']}\n\n")
    if not run_valid:
        reasons = []
        if excluded: reasons.append(f"clock drift FAIL on: {', '.join(excluded)} (excluded from budget sums)")
        if lock_verdict == 'FAIL': reasons.append('clock-lock verification FAILED')
        if smoke_only: reasons.append('preflight ran smoke-only (desktop up) — numbers not certifiable')
        r.write(f"> **RUN INVALID** — {'; '.join(reasons) or 'see clock_integrity block'}. "
                f"Numbers below are not evidence; fix the condition and re-run.\n\n")
    r.write('## Measured ceilings (capacity side, from CEILINGS_JSON)\n\n')
    r.write('| ceiling | measured | notes |\n|---|---|---|\n')
    # tensor rows come from the ceilings dict — the TRT GEMM probe when
    # trt_compute.json was present, else the torch sweep — never the raw locals
    c_src = ceilings.get('tensor_ceilings_source', 'torch-gemm-sweep')
    c_fp16 = ceilings.get('tensor_ceiling_fp16_tflops')
    c_int8 = ceilings.get('tensor_ceiling_int8_tops')
    c_fp8 = ceilings.get('tensor_ceiling_fp8_tflops')
    r.write(f"| Tensor fp16 | {c_fp16 and f'{c_fp16:.1f} TFLOPS' or 'not in ceilings json'} "
            f"| {c_src} ({ceilings.get('tensor_fp16_best_variant') or '—'}) |\n")
    r.write(f"| Tensor int8 | {c_int8 and f'{c_int8:.1f} TOPS' or 'not in ceilings json'} "
            f"| {c_src} ({ceilings.get('tensor_int8_best_at') or '—'}) |\n")
    r.write(f"| Tensor fp8 | {c_fp8 and f'{c_fp8:.1f} TFLOPS' or 'unavailable'} "
            f"| {fp8_note or c_src} |\n")
    r.write(f"| CUDA-core FP32 | {cuda_fp32 and f'{cuda_fp32:.2f} TFLOPS' or 'not in ceilings json'} "
            f"| ceiling for all non-matmul kernels |\n")
    r.write(f"| DRAM bandwidth (idle) | {bw_best and f'{bw_best:.0f} GB/s' or 'not in ceilings json'} "
            f"| bw_eff basis — best copy_RW over the size sweep; solo-exclusive regime |\n")
    r.write(f"| Sustained tensor ({ceilings['sustained_prec']}) | "
            f"{sustained_med and f'{sustained_med:.1f} TFLOPS median' or 'not in ceilings json'} "
            f"| 3-min continuous GEMM — plan against sustained |\n\n")
    r.write(f"Ceilings source: `{ceilings['ceilings_source']['path']}` "
            f"(run start {ceilings['ceilings_source']['start'] or 'unknown'})\n\n")
    r.write(f"Clock integrity: lock_verify={lock_verdict or 'n/a'} | "
            f"worst model drift={worst_model_verdict or 'n/a'} | run_valid={run_valid}\n\n")
    if acc_block is not None:
        r.write('## Accuracy gate\n\n')
        r.write('| model | gate | variants | worst variant | worst agree |\n|---|---|---|---|---|\n')
        for s in acc_rows:
            wa = s['worst_agree_frac']
            r.write(f"| {s['candidate']} | {s['gate'] or '—'} | {s['n_variants'] or '—'} "
                    f"| {s['worst_variant'] or '—'} | {fmt(wa, 4) if wa is not None else '—'} |\n")
        if acc_fail:
            r.write('\nGate FAIL flags the affected demand rows below; capacity numbers stand.\n')
        r.write('\n')
    if not models:
        r.write('## Demand (empty — no models in the manifest)\n\n'
                'U_max / C / L / N / Score are not meaningful for this run.\n')
    else:
        r.write('## Demand (per model)\n\n')
        r.write('| model | p99 ms | time share | MB/frame | BW GB/s | VRAM MB (peak proc) | deadline margin | % time in tensor kernels |\n|---|---|---|---|---|---|---|---|\n')
        for m in models:
            label = m['name']
            if not m.get('measurement_valid', True): label += ' **EXCLUDED (drift FAIL)**'
            if m['name'] in acc_fail: label += ' **accuracy FAIL**'
            if m.get('p99_source') == 'mean-fallback-degraded': label += ' *(p99=mean, degraded)*'
            if m.get('bytes_bound_violation'): label += ' *(bytes clamped to physical bound)*'
            r.write(f"| {label} | {m.get('p99_ms','—')} | {round(m.get('time_share',0),3)} "
                    f"| {round(m.get('bytes_per_frame_MB',0),1)} | {round(m.get('bw_demand_gbps',0),2)} "
                    f"| {m.get('vram_budget_mb','—')} "
                    f"| {round(m.get('deadline_margin',0),2)} | {m.get('pct_time_tensor_kernels','—')} |\n")
        r.write(f"\n**Budgets:** {budgets}  → binding: **{binding}**\n\n")
        r.write(f"**U_max = {fmt(U_max,3)}   C = {fmt(C,2)}   L = {fmt(L,2)}   →   N = {fmt(N,2)}** "
                f"({cause if (N == N and N < 1) else 'fits'})\n\n")
        r.write(f"**Score = N × Σ(arch FLOPs × Hz) = {fmt(score,2)} TFLOP/s** of deadline-respecting work\n")
print(open(f'{out_dir}/report/report.md').read())
PYEOF
}

#=============================================================================
# STAGE 5b — PER-KERNEL ROOFLINE REPORTS
#
# What we are doing: classifying every profiled kernel (memory- vs compute- vs
# latency-bound) against THIS run's measured ceilings — the roofline
# reader takes them from the results.json stage 5 just wrote, so the
# classification and the budgets always share denominators. A refusal from
# roofline_report.py (missing ceiling, uncalibrated ops_per_tinst) downgrades
# to a WARN and a skipped block: the capacity verdict never depends on it.
#=============================================================================
stage5b_roofline() {
  say "=== STAGE 5b: per-kernel roofline reports ==="
  RESULTS="$OUT_DIR/report/results.json"
  [ -f "$RESULTS" ] || { say "WARN: $RESULTS missing — run stage 5 first; roofline skipped"; return 0; }
  [ -f "$MIX_FILE" ] || die "mix manifest '$MIX_FILE' not found"
  tail -n +2 "$MIX_FILE" | while IFS=, read -r NAME ONNX PREC HZ DEADLINE AGF EXTRA; do
    [ -z "$NAME" ] && continue
    MDIR="$OUT_DIR/models/$NAME"
    [ -f "$MDIR/ncu_kernels.csv" ] || { say "roofline: no ncu_kernels.csv for $NAME — skipped"; continue; }
    say "roofline: $NAME"
    python3 "$KIT/model_bench/roofline_report.py" "$MDIR/ncu_kernels.csv" \
      --ceilings "$RESULTS" --device "$DEVICE_CFG" --out "$MDIR/roofline.json" \
      || { say "WARN: roofline_report failed for $NAME (see message above) — block skipped"; rm -f "$MDIR/roofline.json"; continue; }
  done
  # merge per-model summaries into results.json and append the report sections
  python3 - "$MIX_FILE" "$OUT_DIR" <<'PYEOF'
import csv, json, os, sys
mix_file, out_dir = sys.argv[1], sys.argv[2]
res_path = f'{out_dir}/report/results.json'
res = json.load(open(res_path))
by_name = {m.get('name'): m for m in res.get('models', [])}
sections = []
with open(mix_file) as f:
    for row in csv.DictReader(f):
        name = row['name'].strip()
        rl_path = f'{out_dir}/models/{name}/roofline.json'
        if not os.path.exists(rl_path): continue
        try:
            rj = json.load(open(rl_path))
        except Exception as ex:
            print(f'WARN: unreadable roofline.json for {name}: {ex} — skipped', file=sys.stderr)
            continue
        summ = rj.get('summary') if isinstance(rj.get('summary'), dict) else rj
        top = (rj.get('top_kernels') or summ.get('top_kernels')
               or rj.get('kernels') or summ.get('kernels') or [])
        block = {}
        for k in ('time_pct_by_bound', 'time_pct_by_pipe', 'byte_source', 'ceilings_used', 'detail'):
            if summ.get(k) is not None: block[k] = summ[k]
            elif isinstance(rj, dict) and rj.get(k) is not None: block[k] = rj[k]
        block['top_kernels'] = top[:10]
        block.setdefault('detail', f'models/{name}/roofline.json')
        if name in by_name: by_name[name]['roofline'] = block
        sections.append((name, block))
json.dump(res, open(res_path, 'w'), indent=2)
with open(f'{out_dir}/report/report.md', 'a') as r:
    if sections:
        r.write('\n## Roofline (per model)\n')
    for name, block in sections:
        tb = block.get('time_pct_by_bound') or {}
        r.write(f"\n### {name}\n\n")
        r.write(f"{tb.get('memory', '?')}% memory / {tb.get('compute', '?')}% compute / "
                f"{tb.get('latency', '?')}% latency bound by time")
        if block.get('byte_source'): r.write(f" (bytes via {block['byte_source']})")
        r.write('\n\n| kernel | what it does | time us | pipe | bound | % of roof |\n|---|---|---|---|---|---|\n')
        for k in block['top_kernels'][:10]:
            if not isinstance(k, dict): continue
            r.write(f"| {str(k.get('name', '?'))[:48]} | {k.get('label', '—')} | {k.get('us', '—')} "
                    f"| {k.get('pipe', '—')} | {k.get('bound', '—')} | {k.get('pct_of_roof', '—')} |\n")
for name, _ in sections:
    print(f'roofline block merged: {name}')
PYEOF
}

#=============================================================================
# MAIN
#=============================================================================
if [ "${REPORT_ONLY:-0}" = 1 ]; then
  say "REPORT_ONLY=1 — stages 5+5b only, against existing logs in $OUT_DIR"
  stage5_report
  stage5b_roofline
  say "DONE. Report: $OUT_DIR/report/report.md  |  JSON: $OUT_DIR/report/results.json"
  exit 0
fi
stage0_provenance
stage1_lock_clocks
stage3_models
stage4_clock_dial
stage5_report
stage5b_roofline
say "DONE. Report: $OUT_DIR/report/report.md  |  JSON: $OUT_DIR/report/results.json"
