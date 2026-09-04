#!/usr/bin/env bash
#=============================================================================
# Stage 6 - power: the stage-5 measurements (paced solos, co-location arms) and
# a per-watt sweep at every operating point of this platform, judged against
# the baseline point.
#
#   ./run_power.sh                    the default points (configs/power/points_<platform>.csv)
#   ./run_power.sh --list             points, baseline, mixes, sweep arguments - no GPU
#   ./run_power.sh --point <p>        one point (repeatable); --mix <m>: one mix (repeatable)
#   ./run_power.sh --skip-tops        no per-watt sweeps;  --skip-mixes: no paced solos / arms
#   RUN_SECONDS=<s> ./run_power.sh    run length per paced solo / arm (default 90)
#   SWEEP_ARGS="..."                  per_watt_sweep.py arguments (smoke: "--seconds 2 --targets 50,100 --precisions fp16,copy --idle-seconds 2")
#
# No other arguments. Device config, ceilings, registry rows, the newest
# stage-4 solo run and the mixes are discovered as in stage 5. The baseline
# point is the first default row of the point table (Jetson: the required
# power mode; discrete: the default power limit); every other point is an
# excursion from it and every number is re-measured there - nothing is scaled.
# The baseline knob and the saved clock state are restored on exit.
#
#   0. validate   colocation/validate_mixes.sh   no GPU; the selected mixes
#   1. measure    power/measure_points.sh        per point: knob -> derived
#                 config -> mixes (locked at the point's caps) -> per-watt
#                 sweep (unlocked) -> restore; then power_verdict.py
#=============================================================================
set -uo pipefail
. "$(dirname "$0")/power/stage6_common.sh"
parse_power_args "$@"
if [ "$LIST" = 1 ]; then
  exec "$POWER_DIR/measure_points.sh" "$@"
fi
if [ "$SKIP_MIXES" = 0 ]; then
  say "=========== stage 6.0: validate the mixes ==========="
  "$COLOC_DIR/validate_mixes.sh" $(mix_args) || die "mixes invalid - fix configs/colocation/<mix>.csv per the errors above, then rerun"
fi
say "=========== stage 6.1: measure every point -> verdict ==========="
"$POWER_DIR/measure_points.sh" "$@"
