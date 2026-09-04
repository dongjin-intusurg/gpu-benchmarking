#!/usr/bin/env bash
#=============================================================================
# Stage 5a - validate every registered mix against this device (no GPU).
#
#   ./validate_mixes.sh                 every configs/colocation/<mix>.csv
#   ./validate_mixes.sh --mix <m> ...   one mix (repeatable)
#   ./validate_mixes.sh --list          the same, printed as a table
#
# Every error at once: unknown row, row absent from the newest stage-4
# results.json, frame role on a non-engine row, side role on an engine row,
# X on a frame row, prio outside the device's stream priority range, bad
# hz/deadline/mps_pct, duplicate rows, no frame row. The stream priority range
# comes from the compiled row_loop (built here into the kit's scratch bin).
#=============================================================================
set -uo pipefail
. "$(dirname "$0")/stage5_common.sh"
parse_coloc_args "$@"
discover_device
discover_solo_results
BIN="${COLOC_BIN_DIR:-${TMPDIR:-/tmp}/coloc_bin_$USER}/row_loop"
PR=""
if command -v g++ >/dev/null && build_row_loop "$BIN" 2>/dev/null; then
  PR=$("$BIN" --prio-range 2>/dev/null | awk '{print $1","$2}')
fi
echo "stage-4 solo results: $SOLO_RESULTS"
echo "registry rows: $ROWS_DIR"
echo "arms ($PLATFORM): $(platform_arms)${PR:+   stream priority range $PR}"
python3 "$COLOC_DIR/mixes.py" $([ "$LIST" = 1 ] && echo list || echo validate) --mix-dir "$MIX_DIR" --rows-dir "$ROWS_DIR" \
    --results "$SOLO_RESULTS" --platform "$PLATFORM" ${PR:+--prio-range "$PR"} $(mix_args)
