#!/usr/bin/env bash
#=============================================================================
# Stage 4c - measure every row build_models.sh registered, LOCKED, and score N.
#
#   ./measure_models.sh                 every row in engines/<tag>/registry_rows/
#   ./measure_models.sh --only <model>  one model (repeatable)
#   ./measure_models.sh --list          what would be measured, then exit
#
# Three row kinds, three harnesses, ONE scorer (compute_budgets.py):
#   engine      trtexec 1000-iter p99 + nsys + NCU bytes + VRAM, through
#               run_model_bench.sh unchanged (the numbers validated against the
#               reference runs come from that path and must not move)
#   e2e         the model's own C++ driver over all its engines, compiled here:
#                 <bin> <engine_dir> <prec> <inputs> <out.jsonl> <repeats> 0 [args]
#               latency of record = p99 of gpu_total_ms; plus a <model>_decode
#               row (decode-step p99 at the measured token rate)
#   generative  jetson: Edge-LLM llm_bench (visual, prefill, KV-reuse prefill,
#               decode) + llm_inference battery with --dumpProfile
#               discrete: TensorRT-LLM trtllm_vlm_step.py sweep (RequestPerfMetrics)
#               latency of record = one control step: visual + prefill + chunk x decode
#
# Everything runs under a verified clock lock that is released on exit; the
# ceilings run is discovered (newest results_<tag>_ceilings*), never re-measured.
# Output: results/solo_<tag>_<stamp>/{results.json, report.md, engine_rows/,
#         e2e/, generative/, provenance/, measure.log}
#=============================================================================
set -uo pipefail
. "$(dirname "$0")/stage4_common.sh"
parse_only "$@"
discover_device
discover_ceilings
[ -n "${CEILINGS_JSON:-}" ] && [ -f "$CEILINGS_JSON" ] \
  || die "no completed ceilings run for '$DEVICE_TAG' - run scripts/device_ceilings/run_device_ceilings.sh first (N needs the capacity side)"

ROWS_DIR="$ENGINE_ROOT/$DEVICE_TAG/registry_rows"
MR="$MB_DIR/measure_rows.py"
ls "$ROWS_DIR"/*.jsonl >/dev/null 2>&1 || die "no registered rows under $ROWS_DIR - run build_models.sh first"

refuse_if_building(){
  local p
  p=$(pgrep -f "build_models.sh|build_model_engines.sh|trtexec .*--saveEngine|llm_build|visual_build" | grep -v "^$$\$" | head -3 || true)
  [ -z "$p" ] || die "an engine build is running (pids: $(echo $p)) - never measure while anything builds on this GPU"
}

if [ "$LIST" = 1 ]; then
  python3 "$MR" select --rows-dir "$ROWS_DIR" "${ONLY[@]}" --table
  exit 0
fi
refuse_if_building

STAMP=$(date +%Y%m%d_%H%M%S)
OUT="${OUT_DIR:-$RESULTS_ROOT/solo_${DEVICE_TAG}_$STAMP}"
mkdir -p "$OUT/provenance"; LOG="$OUT/measure.log"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "=== stage 4c: measure ($DEVICE_TAG / $PLATFORM) -> $OUT"
say "ceilings: $CEILINGS_JSON"

python3 - "$OUT/provenance/provenance.json" "$DEVICE_CFG" "$PLATFORM" "$DEVICE_TAG" "$CEILINGS_JSON" "$ROWS_DIR" <<'PY'
import json, subprocess, sys, datetime, os
cfg = json.load(open(sys.argv[2]))
trt = ''
try:
    trt = subprocess.run(['trtexec', '--version'], capture_output=True, text=True, timeout=30).stdout.strip().split('\n')[-1]
except Exception:
    pass
json.dump({'device': cfg.get('name') or cfg.get('device_name_match'), 'platform': sys.argv[3], 'device_tag': sys.argv[4],
           'device_config': sys.argv[2], 'ceilings_json': sys.argv[5], 'rows_dir': sys.argv[6],
           'date': datetime.datetime.now().isoformat(timespec='seconds'), 'trtexec_version': trt,
           'host': os.uname().nodename, 'kernel': os.uname().release}, open(sys.argv[1], 'w'), indent=1)
PY

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# lock first, before any row: preflight, store pre-run clocks, pin, verify,
# keep sudo warm, restore on exit. A run that cannot lock refuses here, at
# minute 0, rather than after the engine rows. run_model_bench.sh re-applies
# the same (idempotent) lock for its own provenance and restores to this
# pinned state; the true pre-run state is restored by this script's trap.
#-----------------------------------------------------------------------------
CLOCKS_LOCKED=0; KEEPALIVE_PID=""; SAMPLER_PID=""; CLK_ANCHOR=""; CLK_SAMPLER=""
start_keepalive(){ ( while true; do sudo -n true 2>/dev/null || exit 0; sleep 60; done ) & KEEPALIVE_PID=$!; }
cleanup(){
  [ -n "$KEEPALIVE_PID" ] && { kill "$KEEPALIVE_PID" 2>/dev/null || true; }
  [ -n "$SAMPLER_PID" ] && { kill "$SAMPLER_PID" 2>/dev/null || true; }
  [ -n "$CLK_ANCHOR" ] && { kill "$CLK_ANCHOR" 2>/dev/null || true; }
  [ "$CLOCKS_LOCKED" = 1 ] || return 0
  say "releasing clock locks..."
  if [ "$PLATFORM" = jetson ]; then
    sudo -n jetson_clocks --restore "$OUT/provenance/jetson_clocks_saved.conf" 2>/dev/null \
      && say "clocks restored to pre-run state" \
      || say "WARN: clocks still pinned - restore: sudo jetson_clocks --restore $OUT/provenance/jetson_clocks_saved.conf"
  else
    sudo -n nvidia-smi -rgc >/dev/null 2>&1; sudo -n nvidia-smi -rmc >/dev/null 2>&1; say "clock locks released"
  fi
  CLOCKS_LOCKED=0
}
trap cleanup EXIT INT TERM

python3 "$KIT/common/preflight.py" --device "$DEVICE_CFG" --out "$OUT/provenance/preflight.json" \
  ${ALLOW_DESKTOP:+--allow-desktop} 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || die "preflight refused"
say "=== locking clocks for the whole run (engine, driver and generative rows)"
if [ "$PLATFORM" = jetson ]; then
  sudo -n jetson_clocks --store "$OUT/provenance/jetson_clocks_saved.conf" 2>/dev/null \
    || die "cannot store pre-run clocks (is sudo primed on this tty? run: sudo -v)"
  sudo -n jetson_clocks 2>/dev/null || die "jetson_clocks failed"
else
  MAXGC=$(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits | head -1)
  # Lock the bins the device config declares, not the nameplate max. Stage 3 and
  # run_model_bench.sh already do this; locking to a boost bin the part cannot
  # hold makes verify_lock adopt the realized clock as the reference, and every
  # heavy row then adjudicates against a clock that was never actually held.
  MAXMC=$(nvidia-smi --query-gpu=clocks.max.memory --format=csv,noheader,nounits | head -1)
  CFG_SMLOCK=$(python3 -c 'import json,sys;v=json.load(open(sys.argv[1])).get("sm_lock_mhz");print(v if v else "")' "$DEVICE_CFG")
  [ -n "$CFG_SMLOCK" ] && MAXGC="$CFG_SMLOCK"
  CFG_MEMLOCK=$(python3 -c 'import json,sys;v=json.load(open(sys.argv[1])).get("mem_lock_mhz");print(v if v else "")' "$DEVICE_CFG")
  [ -n "$CFG_MEMLOCK" ] && MAXMC="$CFG_MEMLOCK"
  sudo -n nvidia-smi -pm 1 >/dev/null 2>&1; sudo -n nvidia-smi -lgc "$MAXGC" >/dev/null 2>&1 || die "-lgc failed"
  sudo -n nvidia-smi -lmc "$MAXMC" >/dev/null 2>&1 \
    || say "WARN: -lmc unsupported on this part; memory clock floats"
fi
CLOCKS_LOCKED=1; start_keepalive; say "clocks pinned"
  # MAXGC/MAXMC are set only on the discrete branch above; on jetson nothing is
  # passed and verify_lock resolves its targets from the device config as before.
  python3 "$KIT/common/verify_lock.py" --device "$DEVICE_CFG" --out "$OUT/provenance/lock_verified.json" \
  ${MAXGC:+--requested "sm=${MAXGC},mem=${MAXMC}"} 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || die "clock-lock verification FAILED - evidence: $OUT/provenance/lock_verified.json"

# engine rows -> run_model_bench.sh (re-locks under the outer lock, verifies, measures)
#-----------------------------------------------------------------------------
MIX="$OUT/mix_engine_rows.csv"
n_engine=$(python3 "$MR" mix --rows-dir "$ROWS_DIR" "${ONLY[@]}" --out "$MIX")
ENGINE_RESULTS=""
if [ "$n_engine" -gt 0 ]; then
  say "=== engine rows: $n_engine (trtexec p99 / nsys / NCU / VRAM via run_model_bench.sh)"
  DEVICE_CFG="$DEVICE_CFG" CEILINGS_JSON="$CEILINGS_JSON" "$MB_DIR/run_model_bench.sh" "$MIX" "$OUT/engine_rows" \
    > "$OUT/engine_rows.log" 2>&1
  rc=$?
  tail -25 "$OUT/engine_rows.log" | tee -a "$LOG"
  [ $rc -eq 0 ] && [ -f "$OUT/engine_rows/report/results.json" ] \
    || die "engine rows failed (rc=$rc) - see $OUT/engine_rows.log"
  ENGINE_RESULTS="$OUT/engine_rows/report/results.json"
  # run_model_bench.sh's exit trap released the clocks, and nvidia-smi -rgc is
  # not nested: it dropped the lock taken above as well. The e2e and generative
  # rows that follow must run under the same lock, so re-pin it. (jetson_clocks
  # --restore in the same trap restores the pinned state it stored, so jetson
  # keeps its lock without this.)
  if [ "$PLATFORM" != jetson ] && [ -n "${MAXGC:-}" ]; then
    sudo -n nvidia-smi -lgc "$MAXGC" >/dev/null 2>&1 || die "re-lock -lgc $MAXGC failed after the engine rows"
    sudo -n nvidia-smi -lmc "$MAXMC" >/dev/null 2>&1 || true
    say "clocks re-pinned (sm=$MAXGC mem=$MAXMC) for the e2e and generative rows"
  fi
else
  say "=== engine rows: none selected"
fi

#-----------------------------------------------------------------------------
# non-engine rows need the lock held by this script
#-----------------------------------------------------------------------------
E2E_ROWS=$(python3 "$MR" select --rows-dir "$ROWS_DIR" "${ONLY[@]}" --kind e2e --shell)
GEN_ROWS=$(python3 "$MR" select --rows-dir "$ROWS_DIR" "${ONLY[@]}" --kind generative --shell)
NONENGINE_JSONS=()

if [ -n "$E2E_ROWS$GEN_ROWS" ]; then
  # Under-load clock evidence for the whole row window (every sub-benchmark of
  # the row, as the engine rows have for their timing pass): the device-aware
  # sampler follows an anchor process that lives from row_clock_start to
  # row_clock_stop, and drift_report.py adjudicates the samples against the
  # run's verified lock. A FAIL marks the row measurement_valid:false.
  row_clock_start(){ # dir phase
    sleep 2147483 & CLK_ANCHOR=$!
    python3 "$KIT/common/clock_sampler.py" --device "$DEVICE_CFG" --pid "$CLK_ANCHOR" \
      --out "$1/clock_samples.csv" --phase "$2" >/dev/null 2>&1 & CLK_SAMPLER=$!
  }
  row_clock_stop(){ # dir phase
    kill "$CLK_ANCHOR" 2>/dev/null; wait "$CLK_SAMPLER" 2>/dev/null; CLK_ANCHOR=""; CLK_SAMPLER=""
    python3 "$KIT/common/drift_report.py" "$1/clock_samples.csv" --device "$DEVICE_CFG" \
      --lock "$OUT/provenance/lock_verified.json" --out "$1/drift.json" --phase "$2" >/dev/null 2>&1 \
      || say "  WARN: drift report failed for $2 - clock integrity unrecorded for this row"
  }
  # run one measured process with the peak-memory sampler on its pid
  sampled(){ # vram_out log cmd...
    local vout="$1" log="$2"; shift 2
    # the memory baseline must predate the process (jetson: MemAvailable delta)
    local base; base=$(awk '/MemAvailable/{printf "%.1f", $2/1024}' /proc/meminfo)
    "$@" > "$log" 2>&1 & local pid=$!
    python3 "$MB_DIR/vram_sampler.py" --pid $pid --platform "$PLATFORM" --out "$vout" --baseline-mb "$base" & SAMPLER_PID=$!
    wait $pid; local rc=$?
    wait $SAMPLER_PID 2>/dev/null; SAMPLER_PID=""
    return $rc
  }
fi

#--- e2e driver rows ----------------------------------------------------------
if [ -n "$E2E_ROWS" ]; then
  mkdir -p "$OUT/e2e/bin"
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    unset ARGS
    eval "$line"        # ROW MODEL KIND ENGINE_DIR PRECISION SRC INPUTS ARGS REPEATS HZ DEADLINE_MS MODEL_DEADLINE_MS ARCH_GFLOPS
    D="$OUT/e2e/$ROW"; mkdir -p "$D"
    BIN="$OUT/e2e/bin/${MODEL}_e2e"
    say "=== e2e: $ROW  ($(basename "$SRC") over $ENGINE_DIR, $PRECISION, repeats $REPEATS)"
    if [ ! -x "$BIN" ]; then
      # A discrete box normally carries TensorRT as a tarball beside the trtexec on
      # PATH, so its headers are not under the multiarch include dir. Add that
      # root when it exists (setup.sh compiles the same source the same way).
      _tx=$(command -v trtexec 2>/dev/null || true); _tr=""
      [ -n "$_tx" ] && _tr=$(cd "$(dirname "$_tx")/.." 2>/dev/null && pwd)
      g++ -O2 -std=c++17 "$SRC" -I"/usr/include/$(gcc -dumpmachine)" -I/usr/local/cuda/include \
          ${_tr:+$([ -d "$_tr/include" ] && echo "-I$_tr/include")} ${_tr:+$([ -d "$_tr/lib" ] && echo "-L$_tr/lib")} \
          -L/usr/local/cuda/lib64 -lnvinfer -lnvinfer_plugin -lcudart -ldl -lpthread -o "$BIN" > "$D/compile.log" 2>&1 \
        || { tail -15 "$D/compile.log" | tee -a "$LOG"; say "  $ROW: driver failed to compile - $D/compile.log"; continue; }
    fi
    # relative paths inside the input list resolve against the list's directory
    row_clock_start "$D" "e2e:$ROW"
    ( cd "$(dirname "$INPUTS")" && sampled "$D/vram.json" "$D/driver.log" \
        "$BIN" "$ENGINE_DIR" "$PRECISION" "$INPUTS" "$D/out.jsonl" "$REPEATS" 0 $ARGS )
    rc=$?
    row_clock_stop "$D" "e2e:$ROW"
    [ $rc -eq 0 ] && [ -s "$D/out.jsonl" ] || { tail -10 "$D/driver.log" | tee -a "$LOG"; say "  $ROW: driver failed (rc=$rc) - $D/driver.log"; continue; }
    python3 "$MR" select --rows-dir "$ROWS_DIR" --kind e2e | python3 -c "import json,sys; [json.dump(r, open('$D/row.json','w')) for r in json.load(sys.stdin) if r['row']=='$ROW']"
    python3 "$MR" e2e --row "$D/row.json" --jsonl "$D/out.jsonl" --vram "$D/vram.json" --drift "$D/drift.json" --out "$D/rows.json" 2>&1 | tee -a "$LOG" \
      && NONENGINE_JSONS+=("$D/rows.json")
  done <<< "$E2E_ROWS"
fi

#--- generative rows ----------------------------------------------------------
if [ -n "$GEN_ROWS" ]; then
  declare -A VISLOG=()
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    unset LLM_DIR VISUAL_DIR SERVING_DIR BATTERY IMAGE
    eval "$line"        # ROW MODEL RUNTIME PRECISION LLM_DIR VISUAL_DIR | SERVING_DIR, BATTERY IMAGE CHUNK CONTEXT_LEN REUSE_LEN HZ DEADLINE_MS ...
    D="$OUT/generative/$ROW"; mkdir -p "$D"
    python3 "$MR" select --rows-dir "$ROWS_DIR" --kind generative | python3 -c "import json,sys; [json.dump(r, open('$D/row.json','w')) for r in json.load(sys.stdin) if r['row']=='$ROW']"
    case "$RUNTIME" in
      edgellm)
        [ "$PLATFORM" = jetson ] || die "$ROW is an Edge-LLM row on a $PLATFORM device - the registry should have refused it"
        EDGELLM_ROOT="${EDGELLM_ROOT:-$HOME/tools/TensorRT-Edge-LLM}"
        BENCH="$EDGELLM_ROOT/build/examples/llm/llm_bench"; INFER="$EDGELLM_ROOT/build/examples/llm/llm_inference"
        [ -x "$BENCH" ] || die "llm_bench missing at $BENCH - build the Edge-LLM runtime first"
        export EDGELLM_PLUGIN_PATH="${EDGELLM_PLUGIN_PATH:-$EDGELLM_ROOT/build/libNvInfer_edgellm_plugin.so}"
        say "=== generative (Edge-LLM): $ROW  chunk $CHUNK, context $CONTEXT_LEN, reuse $REUSE_LEN"
        row_clock_start "$D" "generative:$ROW"
        if [ -n "${VISUAL_DIR:-}" ]; then
          # the visual engine is shared by every precision of a model: measure once, link
          if [ -n "${VISLOG[$MODEL]:-}" ]; then
            ln -sf "${VISLOG[$MODEL]}" "$D/visual.log"
          else
            say "  visual 448x448"
            ( cd "$EDGELLM_ROOT" && "$BENCH" --engineDir "$VISUAL_DIR" --mode visual --imageSize 448x448 ) > "$D/visual.log" 2>&1
            VISLOG[$MODEL]="$D/visual.log"
          fi
        fi
        say "  prefill inputLen=$CONTEXT_LEN"
        ( cd "$EDGELLM_ROOT" && "$BENCH" --engineDir "$LLM_DIR" --mode prefill --inputLen "$CONTEXT_LEN" ) > "$D/prefill.log" 2>&1
        NEW=$((CONTEXT_LEN - REUSE_LEN))
        say "  prefill context=$CONTEXT_LEN reuse=$REUSE_LEN (new $NEW; run twice, keep the second)"
        ( cd "$EDGELLM_ROOT" && "$BENCH" --engineDir "$LLM_DIR" --mode prefill --inputLen "$NEW" --reuseKVLen "$REUSE_LEN" ) > "$D/reuse_warm.log" 2>&1
        ( cd "$EDGELLM_ROOT" && "$BENCH" --engineDir "$LLM_DIR" --mode prefill --inputLen "$NEW" --reuseKVLen "$REUSE_LEN" ) > "$D/reuse.log" 2>&1
        say "  decode pastKV=$CONTEXT_LEN"
        ( cd "$EDGELLM_ROOT" && "$BENCH" --engineDir "$LLM_DIR" --mode decode --pastKVLen "$CONTEXT_LEN" ) > "$D/decode.log" 2>&1
        PROFILE=""
        if [ -n "${BATTERY:-}" ] && [ -f "$BATTERY" ] && [ -x "$INFER" ]; then
          # hybrid models refuse context reuse without the snapshot pools, and the
          # runtime sizes the pool in whole slots: slot = linear-attention layers x
          # (recurrent state + conv state). A fixed 64 MiB is one slot for a 4B-class
          # model and ZERO for a 27B-class one (~148 MiB/slot) -> "requires at least
          # one recurrent snapshot slot". Size it from the engine config: one slot,
          # never below 64 MiB, so every model runs with the same one-slot policy.
          SNAP=$(python3 - "$LLM_DIR/config.json" <<'PYSNAP'
import json, sys
c = json.load(open(sys.argv[1]))
n = int(c.get('num_linear_attn_layers') or 0)
rec = (int(c.get('recurrent_state_num_heads') or 0) * int(c.get('recurrent_state_head_dim') or 0)
       * int(c.get('recurrent_state_size') or 0) * (4 if c.get('recurrent_state_dtype', 'fp32') == 'fp32' else 2))
conv = int(c.get('conv_dim') or 0) * int(c.get('conv_kernel') or 0) * (4 if c.get('conv_state_dtype') == 'fp32' else 2)
slot = n * (rec + conv)
print(max(64 << 20, (-(-slot // (1 << 20))) << 20))
PYSNAP
)
          say "  llm_inference battery $(basename "$BATTERY") (context reuse on; recurrent snapshot pool $((SNAP >> 20)) MiB = one slot)"
          ( cd "$EDGELLM_ROOT" && sampled "$D/vram.json" "$D/e2e.log" "$INFER" \
              --engineDir "$LLM_DIR" ${VISUAL_DIR:+--multimodalEngineDir "$(dirname "$VISUAL_DIR")"} \
              --inputFile "$BATTERY" --outputFile "$D/e2e_out.json" \
              --dumpProfile --profileOutputFile "$D/e2e_profile.json" \
              --enableContextReuse --contextCacheRecurrentSnapshotPoolBytes "$SNAP" \
              --contextCachePartialKVSnapshotPoolBytes 67108864 ) \
            && PROFILE="$D/e2e_profile.json" \
            || { tail -8 "$D/e2e.log" | tee -a "$LOG"; say "  $ROW: llm_inference failed - bench numbers only"; }
        fi
        row_clock_stop "$D" "generative:$ROW"
        python3 "$MR" edgellm --row "$D/row.json" --bench-dir "$D" ${PROFILE:+--profile "$PROFILE"} --vram "$D/vram.json" --drift "$D/drift.json" --out "$D/rows.json" 2>&1 | tee -a "$LOG" \
          && NONENGINE_JSONS+=("$D/rows.json")
        ;;
      trtllm)
        [ "$PLATFORM" = discrete ] || die "$ROW is a TensorRT-LLM row on a $PLATFORM device - the registry should have refused it"
        STEP="$HANDOFF_ROOT/trtllm_vlm_step.py"; [ -f "$STEP" ] || STEP="$KIT/model_bench/trtllm/trtllm_vlm_step.py"
        [ -f "$STEP" ] || die "trtllm_vlm_step.py not found (looked in $HANDOFF_ROOT and $KIT/model_bench/trtllm)"
        for _e in "${BENCH_ENV_SH:-}" "$HANDOFF_ROOT/trtllm_env.sh" "$KIT/model_bench/trtllm/trtllm_env.sh"; do
          [ -n "$_e" ] && [ -r "$_e" ] && { . "$_e"; break; }
        done
        TRT_PY="${BENCH_PY:-${TRTLLM_PY:-python3}}"
        "$TRT_PY" -c 'import tensorrt_llm' >/dev/null 2>&1 \
          || die "tensorrt_llm is not importable with '$TRT_PY' - point BENCH_PY at the TensorRT-LLM venv interpreter (Edge-LLM is not a substitute on this platform)"
        say "=== generative (TensorRT-LLM): $ROW  chunk $CHUNK on $(basename "$SERVING_DIR")"
        # this loop reads its rows from a here-string; the sweep must not consume it
        row_clock_start "$D" "generative:$ROW"
        sampled "$D/vram.json" "$D/sweep.log" "$TRT_PY" "$STEP" sweep --model "$SERVING_DIR" --image "$IMAGE" \
            --out "$D" --tag "$ROW" --state locked --chunks "$CHUNK" < /dev/null
        rc=$?; row_clock_stop "$D" "generative:$ROW"
        [ $rc -eq 0 ] || { tail -10 "$D/sweep.log" | tee -a "$LOG"; say "  $ROW: sweep failed - $D/sweep.log"; continue; }
        python3 "$MR" trtllm --row "$D/row.json" --sweep "$D/sweep_${ROW}_locked.json" --vram "$D/vram.json" --drift "$D/drift.json" --out "$D/rows.json" 2>&1 | tee -a "$LOG" \
          && NONENGINE_JSONS+=("$D/rows.json")
        ;;
      *) die "$ROW: unknown runtime '$RUNTIME'" ;;
    esac
  done <<< "$GEN_ROWS"
fi

#-----------------------------------------------------------------------------
# score + merge
#-----------------------------------------------------------------------------
BUDGETS=""
if [ ${#NONENGINE_JSONS[@]} -gt 0 ]; then
  python3 - "$OUT/rows_nonengine.json" "${NONENGINE_JSONS[@]}" <<'PY'
import json, sys
rows = []
for f in sys.argv[2:]:
    rows += json.load(open(f))
json.dump(rows, open(sys.argv[1], 'w'), indent=1)
PY
  say "=== scoring ${#NONENGINE_JSONS[@]} non-engine row sets"
  python3 "$MB_DIR/compute_budgets.py" --rows "$OUT/rows_nonengine.json" --ceilings "$CEILINGS_JSON" \
    --device "$DEVICE_CFG" --out "$OUT/budgets_nonengine.json" --json-only > /dev/null 2>>"$LOG" \
    || die "compute_budgets.py failed on $OUT/rows_nonengine.json"
  BUDGETS="$OUT/budgets_nonengine.json"
fi
[ -n "$ENGINE_RESULTS$BUDGETS" ] || die "nothing was measured"
python3 "$MR" merge ${BUDGETS:+--budgets "$BUDGETS"} ${ENGINE_RESULTS:+--engine-results "$ENGINE_RESULTS"} \
  --provenance "$OUT/provenance/provenance.json" --out "$OUT/results.json" --report "$OUT/report.md" 2>&1 | tee -a "$LOG"
say "=== done: $OUT/report.md"
python3 - "$OUT/results.json" <<'PY'
import json, sys
R = json.load(open(sys.argv[1]))
print(f"{'row':30} {'kind':10} {'prec':7} {'latency ms':>10}  {'U_max':>6} {'C':>6} {'L':>6} {'N':>6}  cause")
for r in R['rows']:
    s = r.get('solo') or {}
    if not s:
        print(f"{r['name']:30} {r.get('kind',''):10} {str(r.get('precision','')):7} {'-':>10}  {r.get('error','unscored')}"); continue
    print(f"{r['name']:30} {r.get('kind',''):10} {str(r.get('precision','')):7} {r['latency_ms']:10.3f}  {s['U_max']:6.4f} {str(s['C']):>6} {s['L']:6.3f} {s['N']:6.3f}  {s['cause']}"
          + ('' if s.get('budget_complete', True) else '  [incomplete: ' + ','.join(s['budgets_not_measured']) + ']'))
PY
