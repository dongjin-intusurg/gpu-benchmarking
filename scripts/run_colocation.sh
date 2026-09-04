#!/usr/bin/env bash
#=============================================================================
# Stage 5 - co-location of registered mixes: validate -> compose -> paced
# solo -> arms -> verdict.
#
#   ./run_colocation.sh                every mix x every arm this platform supports
#   ./run_colocation.sh --list         mixes, resolved rows, arms - no GPU
#   ./run_colocation.sh --mix <m>      one mix (repeatable); --arm <a>: one arm (repeatable)
#   ./run_colocation.sh --skip-solo    reuse the newest run's paced solos: arms only
#   RUN_SECONDS=<s> ./run_colocation.sh   run length per paced solo / arm (default 90)
#   SOLO_RESULTS=<results.json>            a stage-4 run other than the newest
#
# No other arguments. Device config, ceilings run, registry rows and the newest
# stage-4 solo run are discovered from the machine (env.sh path contract);
# mixes are configs/colocation/<mix>.csv (see template.csv).
#
#   0. validate   colocation/validate_mixes.sh   no GPU; every error at once
#   1. measure    colocation/measure_mixes.sh    LOCKED + verified, released on
#                 exit: compose the budget, paced solo per row, every arm
#                 (plain / mps / streams / mig), period makespan, verdict
#=============================================================================
set -uo pipefail
. "$(dirname "$0")/colocation/stage5_common.sh"
parse_coloc_args "$@"
if [ "$LIST" = 1 ]; then
  exec "$COLOC_DIR/validate_mixes.sh" "$@"
fi
say "=========== stage 5.0: validate the mixes ==========="
"$COLOC_DIR/validate_mixes.sh" "$@" || die "mixes invalid - fix configs/colocation/<mix>.csv per the errors above, then rerun"
say "=========== stage 5.1: measure (locked) -> verdict ==========="
"$COLOC_DIR/measure_mixes.sh" "$@"
