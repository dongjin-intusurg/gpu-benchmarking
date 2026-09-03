#!/usr/bin/env bash
#=============================================================================
# Silicon issue-rate probe runner — builds and runs peak_issue_probe.cu
# a public-toolchain issue-rate probe: mma.sync loops on zero, register-
# resident operands — the top tier of the ceiling ladder
# (silicon bound > runtime-attainable > sustained).
#
# Interpretation rules (pre-declared; revised after the NCU arbiter run):
#   - These numbers NEVER replace the runtime-attainable ceilings in budgets
#     or floors.
#   - sm_120: warp-MMA is the full-rate path -> results are the silicon bound.
#   - sm_110: warp-MMA is a compatibility path — results (~58 TF fp16 /
#     ~113 TOPS int8) bound what mma.sync codegen can issue and explain a low
#     hand-written-MatMul throughput; the rated tensor path (used by the
#     library conv kernels) is not reachable from public PTX warp-MMA.
#   - fp8 on sm_110: public ptxas emits no fp8 MMA (verified via SASS) —
#     the row is auto-marked UNSUPPORTED; int8 is the same-rung proxy.
#   - A sustained pass (3 s) alongside the burst pass isolates the DATA
#     dependence of throttling: zero operands barely draw current, so
#     sustained =~ burst here even where real-operand loads derate.
#
# Usage:  ./run_peak_issue_probe.sh [out_dir]   (default: results_<dev>_ceilings/raw)
# Env:    CUDA_PATH (default /usr/local/cuda)
#=============================================================================
set -euo pipefail
KIT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$(cd "$(dirname "$0")" && pwd)/peak_issue_probe.cu"
say(){ echo "[$(date +%H:%M:%S)] $*"; }
die(){ echo "[FATAL] $*" >&2; exit 1; }
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"

if [ -z "${DEVICE_TAG:-}" ]; then
  if [ -f /etc/nv_tegra_release ]; then DEVICE_TAG=jetson; else DEVICE_TAG=discrete; fi
fi
case "$DEVICE_TAG" in
  jetson) ARCH=sm_110a ;;
  discrete) ARCH=sm_120a ;;
  *) die "unknown DEVICE_TAG=$DEVICE_TAG" ;;
esac

OUT="${1:-$KIT_DIR/results_${DEVICE_TAG}_ceilings/raw}"
mkdir -p "$OUT"
BIN="$OUT/peak_issue_probe"

say "building $ARCH -> $BIN"
"$CUDA_PATH/bin/nvcc" -O3 -gencode "arch=compute_${ARCH#sm_},code=$ARCH" "$SRC" -o "$BIN" || die "nvcc build failed"

say "burst pass (1 s per precision)"
"$BIN" 1.0 | tee "$OUT/peak_issue_burst.txt"
say "sustained pass (3 s per precision)"
"$BIN" 3.0 | tee "$OUT/peak_issue_sustained.txt"

say "done — silicon-bound results in $OUT/peak_issue_{burst,sustained}.txt"
say "reminder: tier-1 silicon bounds; budgets keep using the TRT-attainable ceilings"
