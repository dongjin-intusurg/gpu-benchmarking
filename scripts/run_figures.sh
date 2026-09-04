#!/usr/bin/env bash
#=============================================================================
# Stage 7 - figures and report: one derivation file, a fixed figure set and a
# consolidated report regenerated from this device's newest stage 3-6 runs.
#
#   ./run_figures.sh              newest runs of this device -> figs/<tag>/{data.json, report.md, fig_*.png}
#   ./run_figures.sh --list       the resolved inputs, their counts and the output directory - writes nothing
#   ./run_figures.sh --check      regenerate into a scratch directory and diff data.json + report.md
#                                 against the output directory; exit 1 on any difference
#   SOLO_RESULTS=<results.json> COLOC_RUN=<dir> POWER_RUN=<dir> CEILINGS_JSON=<raw/results.json>
#                                 pin an input (empty string = treat the stage as not run)
#   OUT_DIR=<dir>                 output directory (default figs/<device tag>)
#   DEVICE_TAG=<tag>              select the device by tag: regenerate on a machine without the GPU
#
# No other arguments, no GPU, no clock lock. Each measuring stage is optional:
# an absent run skips its figures and the report names the stage to run. The
# derivation file is deterministic (sorted keys, no timestamps outside its
# sources block), so --check tells whether a regeneration changed a number.
#
#   0. collect   figures/collect.py   stage 3-6 results -> data.json (every number the figures use)
#   1. render    figures/render.py    data.json -> fig_*.png
#   2. report    figures/report.py    data.json -> report.md
#=============================================================================
set -uo pipefail
. "$(dirname "$0")/figures/stage7_common.sh"
parse_figures_args "$@"
device_by_tag
discover_solo_run; discover_coloc_run; discover_power_run
discover_ceilings_of_record
OUT_DIR="${OUT_DIR:-$FIGS_ROOT/$DEVICE_TAG}"
[ -n "${CEILINGS_JSON}${SOLO_RESULTS}${COLOC_RUN}${POWER_RUN}" ] \
  || die "no stage 3-6 run of device '$DEVICE_TAG' under $RESULTS_ROOT - run the measuring stages first (or pin an input)"

say "device $DEVICE_TAG ($DEVICE_CFG)"
describe_inputs
say "output    $OUT_DIR"
[ "$LIST" = 1 ] && exit 0

generate(){   # generate <dir> [collect-only]: the three steps into one directory
  local out="$1" only="${2:-}"
  mkdir -p "$out"
  say "=========== stage 7.0: collect -> data.json ==========="
  python3 "$FIG_DIR/collect.py" --device "$DEVICE_CFG" \
    ${CEILINGS_JSON:+--ceilings "$CEILINGS_JSON"} ${SOLO_RESULTS:+--solo "$SOLO_RESULTS"} \
    ${COLOC_RUN:+--coloc "$COLOC_RUN"} ${POWER_RUN:+--power "$POWER_RUN"} \
    --root "$KIT/.." --out "$out/data.json" || die "collect failed"
  if [ "$only" != "no-figures" ]; then
    say "=========== stage 7.1: render -> fig_*.png ==========="
    python3 "$FIG_DIR/render.py" "$out/data.json" --out "$out" || die "render failed"
  fi
  say "=========== stage 7.2: report -> report.md ==========="
  python3 "$FIG_DIR/report.py" "$out/data.json" --out "$out/report.md" || die "report failed"
}

if [ "$CHECK" = 1 ]; then
  [ -f "$OUT_DIR/data.json" ] || die "$OUT_DIR/data.json does not exist - nothing to check against; run without --check first"
  scratch=$(mktemp -d "${WORK_ROOT:-${TMPDIR:-/tmp}}/figures_check.XXXXXX")
  generate "$scratch" no-figures
  rc=0
  for f in data.json report.md; do
    if diff -u "$OUT_DIR/$f" "$scratch/$f"; then say "$f: unchanged"; else say "$f: DIFFERS"; rc=1; fi
  done
  rm -rf "$scratch"
  [ "$rc" = 0 ] && say "check PASS: a regeneration reproduces $OUT_DIR" || say "check FAIL: a regeneration changes $OUT_DIR (diff above)"
  exit "$rc"
fi

generate "$OUT_DIR"
say "figures and report in $OUT_DIR"
