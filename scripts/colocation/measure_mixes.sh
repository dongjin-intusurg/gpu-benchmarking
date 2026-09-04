#!/usr/bin/env bash
#=============================================================================
# Stage 5 - co-location: measure every registered mix under every arm this
# platform supports, LOCKED, and issue one verdict per (mix, arm) cell.
#
#   ./measure_mixes.sh                    every mix x every arm
#   ./measure_mixes.sh --mix <m> [...]    one mix (repeatable)
#   ./measure_mixes.sh --arm <a> [...]    one arm (repeatable): plain mps streams mig
#   ./measure_mixes.sh --skip-solo        reuse the newest run's paced solos (arms only)
#   ./measure_mixes.sh --list             mixes, resolved rows, arms - no GPU
#   RUN_SECONDS=<s>                       run length of every paced solo and arm (default 90)
#
# Per mix (configs/colocation/<mix>.csv, rows by stage-4 row name):
#   resolve   engines/flags/solo numbers from the registry rows + the newest
#             stage-4 results.json (mixes.py) - nothing re-typed
#   compose   the arithmetic budget of the mix from stage-4 numbers
#             (compose_mix.py): U_time/U_bw/U_vram, C, L, N_predicted, per-row
#             contended p99 prediction
#   paced solo  every frame row ALONE at its mix rate with the same driver
#             (row_loop) for RUN_SECONDS; every side row alone with its own
#             runtime - the contention reference of every arm
#   arms      arms/arm_<arm>.sh -> concurrent_mix.json (rows, side loads),
#             then period_makespan.py over the per-frame traces
#   verdict   verdict.py: valid / fits / contention factors / N_measured per
#             cell, the matrix table in report.md
#
# Output: results/coloc_<tag>_<stamp>/{resolved/, composed/, paced_solo/,
#         <mix>/<arm>/, verdict.json, report.md, provenance/, measure.log}
#=============================================================================
set -uo pipefail
. "$(dirname "$0")/stage5_common.sh"
parse_coloc_args "$@"
discover_device
discover_ceilings
[ -n "${CEILINGS_JSON:-}" ] && [ -f "$CEILINGS_JSON" ] || die "no completed ceilings run for '$DEVICE_TAG' - run scripts/device_ceilings/run_device_ceilings.sh first"
discover_solo_results
[ "${#ARMS[@]}" -gt 0 ] || ARMS=($(platform_arms))

MIXPY="$COLOC_DIR/mixes.py"
if [ "$LIST" = 1 ]; then
  echo "stage-4 solo results: $SOLO_RESULTS"
  echo "arms ($PLATFORM): ${ARMS[*]}   run length: ${RUN_SECONDS}s"
  python3 "$MIXPY" list --mix-dir "$MIX_DIR" --rows-dir "$ROWS_DIR" --results "$SOLO_RESULTS" --platform "$PLATFORM" $(mix_args)
  exit $?
fi
p=$(pgrep -f "build_models.sh|build_model_engines.sh|trtexec .*--saveEngine|llm_build|visual_build|measure_models.sh|run_model_bench.sh" | grep -v "^$$\$" | head -3 || true)
[ -z "$p" ] || die "a build or a stage-4 measurement is running (pids: $(echo $p)) - never co-locate over it"

STAMP=$(date +%Y%m%d_%H%M%S)
OUT="${OUT_DIR:-$RESULTS_ROOT/coloc_${DEVICE_TAG}_$STAMP}"
mkdir -p "$OUT/provenance" "$OUT/resolved" "$OUT/composed" "$OUT/bin"; LOG="$OUT/measure.log"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "=== stage 5: co-location ($DEVICE_TAG / $PLATFORM) -> $OUT"
say "stage-4 solo: $SOLO_RESULTS"; say "ceilings: $CEILINGS_JSON"; say "arms: ${ARMS[*]}   run length ${RUN_SECONDS}s"

# ---- resolve + compose (no GPU): a mix that does not resolve is skipped with its errors recorded
mapfile -t NAMES < <(mix_names)
[ "${#NAMES[@]}" -gt 0 ] || { python3 "$MIXPY" list --mix-dir "$MIX_DIR" --rows-dir "$ROWS_DIR" --results "$SOLO_RESULTS" --platform "$PLATFORM" $(mix_args); die "no valid mix to run"; }
GOOD=()
for m in "${NAMES[@]}"; do
  R="$OUT/resolved/$m.json"
  if python3 "$MIXPY" resolve --mix "$MIX_DIR/$m.csv" --rows-dir "$ROWS_DIR" --results "$SOLO_RESULTS" --platform "$PLATFORM" --out "$R" 2>&1 | tee -a "$LOG" \
     && [ "${PIPESTATUS[0]}" -eq 0 ]; then
    python3 "$COLOC_DIR/compose_mix.py" --resolved "$R" --device "$DEVICE_CFG" --ceilings "$CEILINGS_JSON" --out "$OUT/composed/$m.json" 2>&1 | tee -a "$LOG" \
      && [ "${PIPESTATUS[0]}" -eq 0 ] && GOOD+=("$m") || say "$m: compose failed - skipped"
  else
    say "$m: does not resolve - skipped"
  fi
done
[ "${#GOOD[@]}" -gt 0 ] || die "no mix resolved"

python3 - "$OUT/provenance/provenance.json" "$DEVICE_CFG" "$PLATFORM" "$DEVICE_TAG" "$CEILINGS_JSON" "$SOLO_RESULTS" "$RUN_SECONDS" "${ARMS[*]}" "${GOOD[*]}" <<'PY'
import json, sys, datetime, os, subprocess
cfg = json.load(open(sys.argv[2]))
trt = ''
try: trt = subprocess.run(['trtexec', '--version'], capture_output=True, text=True, timeout=30).stdout.strip().split('\n')[-1]
except Exception: pass
json.dump({'device': cfg.get('name') or cfg.get('device_name_match'), 'platform': sys.argv[3], 'device_tag': sys.argv[4], 'device_config': sys.argv[2],
           'ceilings_json': sys.argv[5], 'stage4_results': sys.argv[6], 'run_seconds': float(sys.argv[7]), 'arms': sys.argv[8].split(), 'mixes': sys.argv[9].split(),
           'date': datetime.datetime.now().isoformat(timespec='seconds'), 'trtexec_version': trt, 'host': os.uname().nodename, 'kernel': os.uname().release},
          open(sys.argv[1], 'w'), indent=1)
PY

ROW_LOOP="$OUT/bin/row_loop"; build_row_loop "$ROW_LOOP"; export ROW_LOOP
export COLOC_BIN_DIR="$OUT/bin"

# ---- lock for the whole run, release on exit
cleanup(){ "$COLOC_DIR/mps_ctl.sh" status >/dev/null 2>&1 || "$COLOC_DIR/mps_ctl.sh" stop >/dev/null 2>&1; release_clocks "$OUT/provenance"; return 0; }
trap cleanup EXIT INT TERM
python3 "$KIT/common/preflight.py" --device "$DEVICE_CFG" --out "$OUT/provenance/preflight.json" ${ALLOW_DESKTOP:+--allow-desktop} 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || die "preflight refused"
"$COLOC_DIR/mps_ctl.sh" status >/dev/null || die "an MPS control daemon is already running - stop it first: $COLOC_DIR/mps_ctl.sh stop"
say "=== locking clocks for the whole run (paced solos and every arm)"
lock_clocks "$OUT/provenance" "$LOG"

# sampled <phase> <samples.csv> <drift.json> cmd... : run one measured process
# under the clock sampler and adjudicate drift over its samples
sampled(){
  local phase="$1" csv="$2" drift="$3"; shift 3
  "$@" & local pid=$!
  python3 "$KIT/common/clock_sampler.py" --device "$DEVICE_CFG" --pid "$pid" --out "$csv" --phase "$phase" >/dev/null 2>&1 & local sp=$!
  wait "$pid"; local rc=$?
  wait "$sp" 2>/dev/null
  python3 "$KIT/common/drift_report.py" "$csv" --device "$DEVICE_CFG" --lock "$OUT/provenance/lock_verified.json" --out "$drift" --phase "$phase" >/dev/null 2>&1 \
    || say "WARN: drift report failed for $phase"
  return $rc
}

# ---- paced solo: the contention reference (same driver, same duty, alone)
PS="$OUT/paced_solo"
if [ "$SKIP_SOLO" = 1 ]; then
  PREV="${PACED_SOLO:-$(ls -1dt "$RESULTS_ROOT"/coloc_${DEVICE_TAG}_*/paced_solo 2>/dev/null | grep -v "^$OUT/" | head -1)}"
  [ -n "$PREV" ] && [ -d "$PREV" ] || die "--skip-solo: no earlier paced_solo dir under $RESULTS_ROOT (or set PACED_SOLO=<dir>)"
  ln -s "$(cd "$PREV" && pwd)" "$PS"; say "paced solo reused from $PREV"
else
  mkdir -p "$PS"
fi
export PACED_SOLO="$PS"
say "=== paced solo (${RUN_SECONDS}s per row)"
for m in "${GOOD[@]}"; do
  R="$OUT/resolved/$m.json"
  while IFS='|' read -r name role runtime engine hz dl flags; do
    [ -n "$name" ] || continue
    if [ "$role" = frame ]; then
      key="$name@$hz"; J="$PS/$key.json"
      [ -s "$J" ] && { say "  $key: exists"; continue; }
      [ "$SKIP_SOLO" = 1 ] && { say "  $key: MISSING in the reused paced_solo dir - the cell will carry no contention factor"; continue; }
      say "  $key (solo, row_loop)"
      mkdir -p "$PS/$key"
      # shellcheck disable=SC2086
      ROW_TRACE_DIR="$PS/$key" sampled "solo:$key" "$PS/$key/clock_samples.csv" "$PS/$key/clock_drift.json" \
        "$ROW_LOOP" "$engine" "$name" "$hz" "$dl" "$RUN_SECONDS" "$J" $flags 2> "$PS/$key/stderr.txt" \
        || { rc=$?; tail -3 "$PS/$key/stderr.txt" | tee -a "$LOG"; say "  $key: row_loop failed (rc $rc) - see $PS/$key/stderr.txt"; }
      [ -s "$J" ] && say "    $(python3 -c 'import json,sys; j=json.load(open(sys.argv[1])); print("p50 %.2f / p99 %.2f ms  miss %.2f%%  at %.2f Hz" % (j["p50_ms"], j["p99_ms"], 100*j["miss_frac"], j["achieved_hz"]))' "$J")"
      python3 "$COLOC_DIR/period_makespan.py" "$PS/$key" --mix "$R" --out "$PS/$key/period_makespan.json" >/dev/null 2>&1 || true
    else
      SD="$PS/side_$name"; [ -s "$SD/side_result.json" ] && { say "  side $name: exists"; continue; }
      [ "$SKIP_SOLO" = 1 ] && { say "  side $name: MISSING in the reused paced_solo dir"; continue; }
      case "$runtime" in asr_driver) SS="$COLOC_DIR/side_asr.sh" ;; edgellm|trtllm) SS="$COLOC_DIR/side_$runtime.sh" ;; *) say "  side $name: no side harness for runtime '$runtime'"; continue ;; esac
      say "  side $name (solo, $(basename "$SS"))"; mkdir -p "$SD"
      sampled "solo:side:$name" "$SD/clock_samples.csv" "$SD/clock_drift.json" "$SS" "$R" "$name" "$SD" "$RUN_SECONDS" > "$SD/side.log" 2>&1 \
        || { rc=$?; tail -3 "$SD/side.log" | tee -a "$LOG"; say "  side $name: harness rc $rc - see $SD/side.log"; }
      [ -s "$SD/side_result.json" ] && say "    $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("summary") or json.load(open(sys.argv[1])).get("why"))' "$SD/side_result.json")"
    fi
  done < <(python3 -c '
import json, sys
for r in json.load(open(sys.argv[1]))["rows"]:
    print("|".join([r["name"], r["role"], r.get("runtime") or "", r.get("engine") or "", "X" if r.get("hz_is_x") else "%g" % r["hz"], "%g" % r["deadline_ms"], " ".join(r.get("run_flags") or [])]))' "$R")
done

# ---- arms
for m in "${GOOD[@]}"; do
  R="$OUT/resolved/$m.json"
  for a in "${ARMS[@]}"; do
    AO="$OUT/$m/$a"; mkdir -p "$AO"
    say "=== $m / $a (${RUN_SECONDS}s)"
    sampled "arm:$m/$a" "$AO/clock_samples.csv" "$AO/clock_drift.json" "$COLOC_DIR/arms/arm_$a.sh" "$R" "$AO" > "$AO/arm.log" 2>&1
    rc=$?; tail -n 12 "$AO/arm.log" | tee -a "$LOG"
    [ $rc -eq 0 ] || say "  $m/$a: arm rc $rc"
    if [ -s "$AO/concurrent_mix.json" ] && ls "$AO"/trace_*.csv >/dev/null 2>&1; then
      python3 "$COLOC_DIR/period_makespan.py" "$AO" --mix "$R" --out "$AO/period_makespan.json" 2>&1 | tee -a "$LOG"
    fi
    "$COLOC_DIR/mps_ctl.sh" status >/dev/null || { say "  WARN: MPS daemon left running after $a - stopping it"; "$COLOC_DIR/mps_ctl.sh" stop >> "$LOG" 2>&1; }
  done
done

release_clocks "$OUT/provenance"
say "=== verdict"
python3 "$COLOC_DIR/verdict.py" --run "$OUT" 2>&1 | tee -a "$LOG"
say "done: $OUT/report.md"
