#!/usr/bin/env bash
#=============================================================================
# Stage 6 - power: the same measurements at every operating point of this
# platform, one point at a time, each judged against the baseline point.
#
#   ./measure_points.sh                     the default points (configs/power/points_<platform>.csv, default=1)
#   ./measure_points.sh --point <p> [...]   one point (repeatable); ungated points always run before gated ones
#   ./measure_points.sh --mix <m> [...]     one mix (repeatable); default: every registered mix without side rows
#   ./measure_points.sh --skip-tops         no per-watt sweeps;  --skip-mixes: no paced solos / co-location
#   ./measure_points.sh --list              points, baseline, mixes, sweep arguments - no GPU
#   RUN_SECONDS=<s>   run length of every paced solo and arm (default 90)
#   SWEEP_ARGS="..."  per_watt_sweep.py arguments (default: --n 4096 jetson / 8192 discrete, --seconds 15)
#
# Per point:
#   knob      apply the point (nvpmodel mode / power limit / clock-cap recipe),
#             verified by READ-BACK (mode name, power.limit) - never by exit code
#   config    derive the point's device config: required mode, lock targets at
#             the point's own caps (derive_device_config.py) - every downstream
#             tool judges the point against itself
#   mixes     colocation/measure_mixes.sh at the point (preflight, lock verified
#             at the point's caps, paced solo per frame row under the power
#             sampler, every arm, release, stage-5 verdict) -> <point>/coloc/
#   per-watt  per_watt_sweep.py, clocks unlocked (DVFS is the measurement; a
#             clock-cap recipe keeps its cap) -> <point>/per_watt/
#   restore   the EXIT trap puts the baseline knob and the saved clocks back
# Then power_verdict.py: J/frame, ratios vs baseline, per-watt fits, report.md.
#
# A gated point (Jetson TPC power-gating mask, applied at boot only) poisons
# every ungated point of the same boot: the stage records it (boot-id marker)
# and refuses ungated points until a reboot.
#
# Output: results/power_<tag>_<stamp>/{provenance/, <point>/{readback.json,
#         device_config.json, coloc/, per_watt/, status.json}, verdict.json, report.md}
#=============================================================================
set -uo pipefail
. "$(dirname "$0")/stage6_common.sh"
parse_power_args "$@"
discover_device
discover_ceilings
[ -n "${CEILINGS_JSON:-}" ] && [ -f "$CEILINGS_JSON" ] || die "no completed ceilings run for '$DEVICE_TAG' - run scripts/device_ceilings/run_device_ceilings.sh first"
discover_solo_results
BASE_CFG="$DEVICE_CFG"
[ -f "$(points_file)" ] || die "no point table for $PLATFORM: $(points_file)"
# point_rows validates the table and the --point names; its FATAL line is the
# error message (mapfile hides the exit status, an empty selection is the tell)
mapfile -t PTS < <(point_rows "${POINTS[@]}" 2>&1)
[ "${#PTS[@]}" -gt 0 ] && [ "${PTS[0]#FATAL}" = "${PTS[0]}" ] || { echo "${PTS[0]:-}" >&2; die "no point selected"; }
BASELINE=$(baseline_point); [ -n "$BASELINE" ] || die "$(points_file): no default=1 row (the first one is the baseline)"
BASE_KNOB=""; BASE_VAL=""
while IFS='|' read -r p k v d g n; do [ "$p" = "$BASELINE" ] && { BASE_KNOB="$k"; BASE_VAL="$v"; }; done < <(point_rows)
[ -n "$BASE_KNOB" ] || die "baseline $BASELINE has no knob"
printf '%s\n' "${PTS[@]}" | grep -q "^$BASELINE|" \
  || say "NOTE: baseline $BASELINE is not among the selected points - the verdict's point/baseline ratios will be undefined (add --point $BASELINE)"
# default mixes: the registered mixes without side rows (frame rows only - the
# paced, deadline-judged residents a power point is priced on)
if [ "${#MIXES[@]}" -eq 0 ]; then
  mapfile -t MIXES < <(python3 "$COLOC_DIR/mixes.py" list --mix-dir "$MIX_DIR" --rows-dir "$ROWS_DIR" --results "$SOLO_RESULTS" --platform "$PLATFORM" 2>/dev/null \
                        | sed -n 's/^mix \([^ ]*\) *(.*: [0-9]* frame, 0 side)$/\1/p')
fi
[ "${#MIXES[@]}" -gt 0 ] || [ "$SKIP_MIXES" = 1 ] || die "no frame-only mix registered under $MIX_DIR (or pass --mix)"
[ "$PLATFORM" = jetson ] && SWEEP_N=4096 || SWEEP_N=8192
SWEEP_ARGS="${SWEEP_ARGS:---n $SWEEP_N --seconds 15}"

if [ "$LIST" = 1 ]; then
  echo "device: $DEVICE_TAG ($PLATFORM)  base config: $BASE_CFG"
  echo "points ($(points_file)); baseline $BASELINE:"
  printf '  %-10s %-9s %-8s %-7s %-5s %s\n' point knob value default gated note
  while IFS='|' read -r p k v d g n; do printf '  %-10s %-9s %-8s %-7s %-5s %s\n' "$p" "$k" "$v" "$d" "$g" "$n"; done < <(point_rows "${POINTS[@]}")
  echo "mixes per point: ${MIXES[*]:-(none)}   arms: $(platform_arms)   run length ${RUN_SECONDS}s"
  echo "per-watt sweep: per_watt_sweep.py $SWEEP_ARGS   $([ "$SKIP_TOPS" = 1 ] && echo '(skipped)')"
  echo "stage-4 solo results: $SOLO_RESULTS"
  [ -f "$GATE_MARK" ] && echo "NOTE: a gated point ran in this boot ($(cat "$GATE_MARK")) - ungated points refuse until reboot"
  exit 0
fi
p=$(pgrep -f "build_models.sh|build_model_engines.sh|trtexec .*--saveEngine|llm_build|visual_build|measure_models.sh|run_model_bench.sh|measure_mixes.sh|run_colocation.sh" | grep -v "^$$\$" | head -3 || true)
[ -z "$p" ] || die "a build or a measurement is running (pids: $(echo $p)) - never change the operating point under it"
if [ "$PLATFORM" = jetson ]; then
  sudo -n true 2>/dev/null || die "sudo is not primed on this tty (run: sudo -v) - the knobs and the clock lock need it"
else
  # the discrete knob and lock only ever call nvidia-smi: accept a passwordless rule scoped to it
  sudo -n nvidia-smi -L >/dev/null 2>&1 || die "sudo -n nvidia-smi fails - the knobs and the clock lock need it (run: sudo -v, or a NOPASSWD rule for nvidia-smi)"
fi

STAMP=$(date +%Y%m%d_%H%M%S)
OUT="${OUT_DIR:-$RESULTS_ROOT/power_${DEVICE_TAG}_$STAMP}"
mkdir -p "$OUT/provenance"; LOG="$OUT/measure.log"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "=== stage 6: power ($DEVICE_TAG / $PLATFORM) -> $OUT"
say "points: $(for r in "${PTS[@]}"; do echo -n "${r%%|*} "; done)  baseline $BASELINE   mixes: ${MIXES[*]:-(none)}   run length ${RUN_SECONDS}s"

# headless / exclusivity gate once, at the baseline, BEFORE any knob is touched
python3 "$KIT/common/preflight.py" --device "$BASE_CFG" --out "$OUT/provenance/preflight_baseline.json" --baseline-seconds 3 ${ALLOW_DESKTOP:+--allow-desktop} 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || die "preflight refused"
knob readback "$BASE_KNOB" "$BASE_VAL" "$OUT/provenance/state_before.json" >/dev/null || die "cannot read the current operating point"
python3 - "$OUT/provenance/provenance.json" "$BASE_CFG" "$PLATFORM" "$DEVICE_TAG" "$CEILINGS_JSON" "$SOLO_RESULTS" "$RUN_SECONDS" "$BASELINE" "$SWEEP_ARGS" "$(points_file)" "${MIXES[*]:-}" "${PTS[@]}" <<'PY'
import json, sys, datetime, platform
a = sys.argv
json.dump(dict(schema='power-run/v1', date=datetime.datetime.now().isoformat(timespec='seconds'), host=platform.node(), base_device_config=a[2], platform=a[3],
               device_tag=a[4], ceilings=a[5], stage4_results=a[6], run_seconds=float(a[7]), baseline=a[8], sweep_args=a[9], points_file=a[10],
               mixes=a[11].split(), points=[r.split('|')[0] for r in a[12:]], point_rows=[dict(zip(('point', 'knob', 'value', 'default', 'gated', 'note'), r.split('|'))) for r in a[12:]]),
          open(a[1], 'w'), indent=1)
PY
JC_SAVED=""
if [ "$PLATFORM" = jetson ]; then
  JC_SAVED="$OUT/provenance/jetson_clocks_saved.conf"
  sudo -n jetson_clocks --store "$JC_SAVED" </dev/null 2>/dev/null || die "cannot store the pre-run clock state"
fi
RESTORED=0
restore_all(){
  [ "$RESTORED" = 1 ] && return 0; RESTORED=1
  "$COLOC_DIR/mps_ctl.sh" status >/dev/null 2>&1 || "$COLOC_DIR/mps_ctl.sh" stop >/dev/null 2>&1
  say "restoring the baseline point ($BASELINE) and the saved clock state..."
  knob restore "$BASE_KNOB" "$BASE_VAL" >/dev/null 2>&1 && say "  $BASELINE restored" || say "  WARN: could not restore $BASELINE - do it by hand: $POWER_DIR/knob_$PLATFORM.sh restore $BASE_KNOB $BASE_VAL"
  if [ -n "$JC_SAVED" ] && [ -s "$JC_SAVED" ]; then
    sudo -n jetson_clocks --restore "$JC_SAVED" </dev/null 2>/dev/null && say "  clocks restored to pre-run state" \
      || say "  WARN: clocks still pinned - restore: sudo jetson_clocks --restore $JC_SAVED"
  fi
  [ -f "$GATE_MARK" ] && say "  NOTE: a gated point ran in this boot - the power-gating mask stays applied under $BASELINE until reboot; reboot before any ungated measurement"
  [ -n "$KEEPALIVE_PID" ] && { kill "$KEEPALIVE_PID" 2>/dev/null || true; KEEPALIVE_PID=""; }
  return 0
}
trap restore_all EXIT INT TERM
start_keepalive

status(){ python3 -c 'import json,sys; json.dump(dict(status=sys.argv[2], why=(sys.argv[3] or None)), open(sys.argv[1],"w"), indent=1)' "$1" "$2" "${3:-}"; }
for r in "${PTS[@]}"; do
  IFS='|' read -r point knob_name value _d gated note <<<"$r"
  PD="$OUT/$point"; mkdir -p "$PD"
  say "=== point $point ($knob_name $value)${note:+ - $note}"
  gate_check "$point" "$gated"
  mode=$(knob apply "$knob_name" "$value" 2> >(sed 's/^/  /' | tee -a "$LOG" >&2)); rc=$?
  if [ $rc -eq 3 ]; then status "$PD/status.json" SKIPPED "knob cannot apply $knob_name $value on this device"; say "  $point: SKIPPED"; continue; fi
  [ $rc -eq 0 ] || { status "$PD/status.json" FAIL "knob apply failed (rc $rc)"; say "  $point: knob FAILED"; continue; }
  gate_mark "$point" "$gated"
  knob readback "$knob_name" "$value" "$PD/readback.json" >/dev/null || { status "$PD/status.json" FAIL "read-back failed"; continue; }
  say "  applied: $mode"
  python3 "$POWER_DIR/derive_device_config.py" --base "$BASE_CFG" --point "$point" --readback "$PD/readback.json" --out "$PD/device_config.json" 2>&1 | sed 's/^/  /' | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { status "$PD/status.json" FAIL "derived config"; continue; }
  export DEVICE_CFG="$PD/device_config.json"
  why=""
  if [ "$SKIP_MIXES" = 0 ]; then
    say "  mixes (${MIXES[*]}) at $point -> $PD/coloc"
    # measure_mixes: preflight at the point, lock verified at the point's caps, paced solos + arms under the sampler, release, verdict
    OUT_DIR="$PD/coloc" RUN_SECONDS="$RUN_SECONDS" SOLO_RESULTS="$SOLO_RESULTS" CEILINGS_JSON="$CEILINGS_JSON" DEVICE_CFG="$DEVICE_CFG" \
      "$COLOC_DIR/measure_mixes.sh" $(mix_args) > "$PD/coloc.log" 2>&1; rc=$?
    grep -E '^\[.*\] (=== |  [^ ].*(p50|exists|MISSING|failed|rc )|done:|FATAL)' "$PD/coloc.log" | sed 's/^/  /' | tail -n 40 | tee -a "$LOG"
    [ $rc -eq 0 ] && [ -s "$PD/coloc/verdict.json" ] || why="mixes rc $rc - see $PD/coloc.log"
    [ -s "$PD/coloc/provenance/preflight.json" ] || die "preflight did not run at $point - see $PD/coloc.log"
  fi
  if [ "$SKIP_TOPS" = 0 ]; then
    say "  per-watt sweep at $point (clocks unlocked${knob_name:+; $knob_name $value held}) -> $PD/per_watt"
    [ "$knob_name" = lgc ] && sudo -n nvidia-smi -lgc "$value" >/dev/null 2>&1
    python3 "$POWER_DIR/per_watt_sweep.py" --device "$DEVICE_CFG" --out "$PD/per_watt" $SWEEP_ARGS > "$PD/per_watt.log" 2>&1; rc=$?
    [ "$knob_name" = lgc ] && sudo -n nvidia-smi -rgc >/dev/null 2>&1
    grep -E '^(idle|== |  fit|  power-cap|wrote)' "$PD/per_watt.log" | sed 's/^/  /' | tee -a "$LOG"
    [ $rc -eq 0 ] && [ -s "$PD/per_watt/power_tops_sweep.json" ] || { tail -5 "$PD/per_watt.log" | tee -a "$LOG"; why="${why:+$why; }per-watt sweep rc $rc - see $PD/per_watt.log"; }
  fi
  if [ -n "$why" ]; then status "$PD/status.json" FAIL "$why"; say "  $point: FAIL - $why"; else status "$PD/status.json" OK; say "  $point: OK"; fi
  export DEVICE_CFG="$BASE_CFG"
done
restore_all
say "=== verdict"
python3 "$POWER_DIR/power_verdict.py" --run "$OUT" 2>&1 | tail -n 30 | tee -a "$LOG"
say "done: $OUT/report.md"
