#!/usr/bin/env bash
#=============================================================================
# build_model_engines.sh — candidate engine + fp32 reference from one ONNX
#
# Purpose: turn the operator's model manifest into the two pinned TensorRT
# artifacts the suite measures: the candidate engine at the deployment
# precision, and a naive fp32 reference engine from the SAME ONNX (the
# accuracy gate's ground truth). The engine — not the abstract model — is
# the unit of measurement: a different ONNX, precision, or TensorRT version
# is a different artifact, so both engines are sha256-hashed and the build is
# stamped; a stamp mismatch on rerun is a hard stop, never a silent reuse.
#
# Also emits: per-engine layer censuses (--profilingVerbosity=detailed
# --exportLayerInfo — the realized per-layer precision record), the 7-column
# mix manifest consumed by run_model_bench.sh, and — when MODEL_ARCH_GFLOPS
# is left empty — the architectural GFLOPs computed from the ONNX.
#
# Usage:
#   DEVICE_CFG=<device_configs/*.json> DEVICE_TAG=<device tag> \
#     ./build_model_engines.sh <model_manifest.env> <out_dir>
#
# Idempotency: <out_dir>/build_stamp records onnx sha256 + precision + TRT
# version. Matching stamp = builds skipped (mix regenerated if missing);
# mismatch = die with instructions. rm the stamp (or the out dir) to rebuild.
#
# Env: DEVICE_CFG (device config json), DEVICE_TAG (names the mix csv),
#      MIX_DIR (mix csv destination), MODEL_BUILD_ARGS (build-only trtexec flags, e.g. min/opt/max shape
#      profiles; enter the stamp, never a timing run), SKIP_FP32_REF=1 (a
#      secondary precision whose primary build dir already holds the fp32 ref).
# Clock locks are NOT needed for builds — but builds must never overlap a
# measurement on this GPU (contaminates both).
#=============================================================================
set -euo pipefail
say(){ echo "[$(date +%H:%M:%S)] $*"; }
die(){ echo "[FATAL] $*" >&2; exit 1; }

KIT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$PATH:/usr/src/tensorrt/bin"

[ $# -ge 2 ] || die "usage: build_model_engines.sh <model_manifest.env> <out_dir>"
MANIFEST="$1"
OUT_DIR="$2"
[ -f "$MANIFEST" ] || die "model manifest not found: $MANIFEST"
[ -n "${DEVICE_CFG:-}" ] || die "DEVICE_CFG not set — export the device config json path (or run via run_model_solo.sh, which selects it)"
[ -f "$DEVICE_CFG" ] || die "DEVICE_CFG points at a missing file: $DEVICE_CFG"
[ -n "${DEVICE_TAG:-}" ] || die "DEVICE_TAG not set — the device config tag; it names the generated mix csv"
command -v trtexec >/dev/null || die "trtexec missing — install TensorRT or fix PATH (usually /usr/src/tensorrt/bin)"

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"   # absolute: engine paths land in the mix csv

#-----------------------------------------------------------------------------
# Manifest validation, field by field. Each die names the field AND why the
# pipeline needs it — the operator fixes the manifest, not the script.
#-----------------------------------------------------------------------------
# shellcheck disable=SC1090
. "$MANIFEST"

req(){ # $1 = field, $2 = why it is required
  [ -n "${!1:-}" ] || die "model manifest field $1 is empty or missing — $2. Edit $MANIFEST (the template documents every field)."
}
req MODEL_NAME        "it identifies the model: names both engines, the mix row, and the results directory"
req MODEL_ONNX        "it is the single source ONNX both the candidate and the fp32 reference are built from (same-source is what makes the gate comparison valid)"
req MODEL_PRECISION   "deployment precision selects the trtexec builder flags (fp16|int8|fp8|fp32)"
req MODEL_HZ          "the mix-defined rate is the demand side of the time and bandwidth budgets"
req MODEL_DEADLINE_MS "the p99 latency bound is the L term of N = min(L, C)"
[ -f "$MODEL_ONNX" ] || die "MODEL_ONNX not found: $MODEL_ONNX — check the path in $MANIFEST"

case "$MODEL_PRECISION" in
  fp16) PREC_FLAGS="--fp16" ;;
  int8)
    PREC_FLAGS="--int8 --fp16"
    if [ -n "${MODEL_CALIB_CACHE:-}" ]; then
      [ -f "$MODEL_CALIB_CACHE" ] || die "MODEL_CALIB_CACHE not found: $MODEL_CALIB_CACHE — fix the path or clear the field"
      PREC_FLAGS="$PREC_FLAGS --calib=$MODEL_CALIB_CACHE"
    else
      say "WARN: int8 build without MODEL_CALIB_CACHE — trtexec self-calibrates on synthetic data (timing unaffected; accuracy fidelity is judged by the gate)"
    fi
    ;;
  fp8)  PREC_FLAGS="--fp8 --fp16" ;;
  fp32) PREC_FLAGS="" ;;
  *) die "MODEL_PRECISION '$MODEL_PRECISION' is not one of fp16|int8|fp8|fp32" ;;
esac
if [ -n "${MODEL_CALIB_CACHE:-}" ] && [ "$MODEL_PRECISION" != int8 ]; then
  say "WARN: MODEL_CALIB_CACHE is set but precision is $MODEL_PRECISION — calibration caches only apply to int8; field ignored"
fi

# EXTRA rides EVERY trtexec invocation — build, timing, nsys, NCU, gate — so
# plugin-model behavior is identical across all of them (the single-model dial
# stage bug was exactly this string missing from one invocation).
EXTRA=""
if [ -n "${MODEL_PLUGINS:-}" ]; then
  [ -f "$MODEL_PLUGINS" ] || die "MODEL_PLUGINS not found: $MODEL_PLUGINS — build the plugin .so first or clear the field"
  EXTRA="--staticPlugins=$MODEL_PLUGINS"
fi
if [ -n "${MODEL_SHAPES:-}" ]; then
  EXTRA="$EXTRA --shapes=$MODEL_SHAPES"
fi
if [ -n "${MODEL_EXTRA_ARGS:-}" ]; then
  EXTRA="$EXTRA $MODEL_EXTRA_ARGS"
fi
EXTRA="${EXTRA# }"
# MODEL_BUILD_ARGS are BUILD-ONLY trtexec flags (min/opt/max shape profiles for
# a dynamic decoder, say). They shape the artifact, so they enter the stamp,
# but they never ride a timing run - the built engine's own profile does.
BUILD_ARGS="${MODEL_BUILD_ARGS:-}"

CAND="$OUT_DIR/${MODEL_NAME}_${MODEL_PRECISION}.engine"
REF="$OUT_DIR/${MODEL_NAME}_fp32_ref.engine"
MIX="${MIX_DIR:-$KIT_DIR}/mix_${DEVICE_TAG}_${MODEL_NAME}.csv"   # MIX_DIR: where the mix csv lands (default: the kit dir)
STAMP_FILE="$OUT_DIR/build_stamp"

#-----------------------------------------------------------------------------
# Build stamp: onnx sha256 + precision + TRT version + builder opt level. Any
# change means the artifacts on disk no longer correspond to the manifest —
# reusing them would attach measurements to the wrong artifact identity.
#-----------------------------------------------------------------------------
BUILD_OPT_LEVEL="${BUILD_OPT_LEVEL:-5}"
ONNX_SHA="$(sha256sum "$MODEL_ONNX" | awk '{print $1}')"
TRT_VER="$(python3 -c 'import tensorrt; print(tensorrt.__version__)' 2>/dev/null || true)"
if [ -z "$TRT_VER" ]; then
  TRT_VER="$(trtexec --help 2>&1 | grep -m1 -i tensorrt | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
fi
[ -n "$TRT_VER" ] || TRT_VER=unknown
STAMP="onnx_sha256=$ONNX_SHA precision=$MODEL_PRECISION trt=$TRT_VER opt_level=$BUILD_OPT_LEVEL"
[ -z "$BUILD_ARGS" ] || STAMP="$STAMP build_args_sha256=$(printf '%s' "$BUILD_ARGS" | sha256sum | awk '{print $1}')"
# SKIP_FP32_REF=1: a secondary precision of a model whose primary build dir
# already holds the fp32 reference - do not build (and hash) it twice.
SKIP_FP32_REF="${SKIP_FP32_REF:-0}"

SKIP_BUILD=0
if [ -f "$STAMP_FILE" ]; then
  OLD_STAMP="$(cat "$STAMP_FILE")"
  if [ "$OLD_STAMP" = "$STAMP" ]; then
    SKIP_BUILD=1
    say "build_stamp matches — builds skipped (rm $STAMP_FILE to force a rebuild)"
    [ -f "$CAND" ] && { [ "$SKIP_FP32_REF" = 1 ] || [ -f "$REF" ]; } || die "build_stamp present but engines missing in $OUT_DIR — rm $STAMP_FILE and rerun to rebuild"
  else
    die "build_stamp mismatch — the engines in $OUT_DIR were built from different inputs:
  on disk : $OLD_STAMP
  manifest: $STAMP
A changed ONNX, precision, or TensorRT version is a different measurement
artifact. rm -r $OUT_DIR (or just $STAMP_FILE) to rebuild from the current
manifest — the old engines and their measurements must not be mixed with new ones."
  fi
fi

#-----------------------------------------------------------------------------
# Builds. Both engines carry the layer census flags: the census is the
# realized per-layer precision record ('int8' engines are mixed precision by
# design — the census, not the flag, is the truth).
#
# Build policy (suite rule): naive TensorRT at the manifest precision with
# MAXIMUM tactic search (--builderOptimizationLevel=5) and nothing else — no
# QDQ surgery, no format pinning, no plugin rework. The suite prices what
# the TRT builder finds on its own at the provided quantization level; level 5
# makes builds noticeably slower but the cost is paid once, unlocked, and
# never overlaps a measurement. BUILD_OPT_LEVEL=<0-5> overrides for debugging.
#-----------------------------------------------------------------------------
build_engine(){ # $1 = engine path, $2 = precision flags, $3 = log path
  local census="${1%.engine}_profile.json"
  # word splitting of $2/$EXTRA is intentional: they are flag strings
  trtexec --onnx="$MODEL_ONNX" $2 $EXTRA $BUILD_ARGS \
    --builderOptimizationLevel="$BUILD_OPT_LEVEL" \
    --saveEngine="$1" \
    --profilingVerbosity=detailed --exportLayerInfo="$census" \
    > "$3" 2>&1
}

if [ "$SKIP_BUILD" = 0 ]; then
  say "REMINDER: never build while anything else measures on this GPU — a build contaminates concurrent measurements and vice versa. The GPU must be otherwise idle."
  CAND_LOG="$OUT_DIR/build_${MODEL_PRECISION}.log"
  say "building candidate engine ($MODEL_PRECISION): $CAND"
  build_engine "$CAND" "$PREC_FLAGS" "$CAND_LOG" \
    || die "candidate engine build failed — see $CAND_LOG"
  REF_LOG="$OUT_DIR/build_fp32_ref.log"
  if [ "$SKIP_FP32_REF" = 1 ]; then
    say "fp32 reference skipped (SKIP_FP32_REF=1: the primary-precision build dir carries it)"
  else
  say "building fp32 reference engine (same ONNX, no precision flags): $REF"
  build_engine "$REF" "" "$REF_LOG" || die "fp32 reference build failed — see $REF_LOG
The accuracy gate requires an fp32 TensorRT reference built on-device from the
same ONNX. If this device genuinely cannot build it (e.g. a plugin without an
fp32 path), supply a known-good fp32 engine yourself:
  python3 $KIT_DIR/model_bench/run_accuracy_gate.py <acc_manifest> --reference <fp32.engine> ...
An operator-supplied reference is recorded as such in accuracy.json — there is
no silent precision fallback."
  fi
  if [ "$SKIP_FP32_REF" = 1 ]; then sha256sum "$CAND" > "$OUT_DIR/engines.sha256"
  else sha256sum "$CAND" "$REF" > "$OUT_DIR/engines.sha256"; fi
  echo "$STAMP" > "$STAMP_FILE"
  say "engines hashed: $OUT_DIR/engines.sha256"
fi

#-----------------------------------------------------------------------------
# Mix manifest. Column 2 is the BUILT engine (run_model_bench adopts it, never
# rebuilds); column 7 is the assembled extra-args string — it may contain
# commas, which is fine because the positional reader absorbs the remainder.
#-----------------------------------------------------------------------------
if [ "$SKIP_BUILD" = 1 ] && [ -f "$MIX" ]; then
  say "mix manifest already present: $MIX — nothing to do"
  exit 0
fi

AGF="${MODEL_ARCH_GFLOPS:-}"
if [ -z "$AGF" ]; then
  # Architectural GFLOPs are the Score's workload constant — naive graph count
  # at pinned shapes, never profiler-measured executed FLOPs.
  say "MODEL_ARCH_GFLOPS empty — computing architectural GFLOPs from the ONNX"
  SHAPE_FLAGS=()
  if [ -n "${MODEL_SHAPES:-}" ]; then
    # trtexec --shapes syntax 'a:1x2x3,b:4x5' -> repeated --shape flags
    IFS=',' read -ra _specs <<< "$MODEL_SHAPES"
    for _s in "${_specs[@]}"; do
      SHAPE_FLAGS+=(--shape "$_s")
    done
  fi
  AGF_LOG="$OUT_DIR/arch_gflops.log"
  python3 "$KIT_DIR/model_bench/compute_arch_gflops.py" "$MODEL_ONNX" "${SHAPE_FLAGS[@]}" \
    > "$AGF_LOG" 2>&1 \
    || die "compute_arch_gflops.py failed — see $AGF_LOG (needs the onnx python package; alternatively fill MODEL_ARCH_GFLOPS in $MANIFEST by hand)"
  AGF="$(grep -m1 '^arch_gflops:' "$AGF_LOG" | awk '{print $2}')"
  [ -n "$AGF" ] || die "could not parse 'arch_gflops:' from $AGF_LOG"
  say "arch_gflops: $AGF (skipped-node notes, if any, are in $AGF_LOG)"
fi

{
  echo "name,onnx,precision,hz,deadline_ms,arch_gflops,extra"
  echo "$MODEL_NAME,$CAND,$MODEL_PRECISION,$MODEL_HZ,$MODEL_DEADLINE_MS,$AGF,$EXTRA"
} > "$MIX"
say "wrote $MIX"

# ---------------------------------------------------------------------------
# Validate what we just built, before anything measures with it. A build that
# silently fell back to another precision, or an engine that will not
# deserialize, must fail HERE - not three stages later inside a timing run.
# SKIP_ENGINE_VALIDATION=1 bypasses it (debugging only).
# ---------------------------------------------------------------------------
if [ -z "${SKIP_ENGINE_VALIDATION:-}" ]; then
  say "=== validating built engines ==="
  # The candidate carries the manifest precision; the fp32 reference is built
  # with NO precision flags by design, so it must be judged as fp32 - validating
  # both against one requested precision fails the reference every time.
  # plugin engines only deserialize with their .so loaded: the load gate gets
  # the same EXTRA string every measurement will use
  python3 "$(dirname "$0")/validate_engines.py" "$CAND" \
      --precision "$MODEL_PRECISION" ${EXTRA:+--extra="$EXTRA"} 2>&1 | tee    "$OUT_DIR/engine_validation.txt"
  vrc=${PIPESTATUS[0]}
  if [ "$SKIP_FP32_REF" = 1 ]; then rrc=0; else
  python3 "$(dirname "$0")/validate_engines.py" "$REF" \
      --precision fp32 ${EXTRA:+--extra="$EXTRA"} 2>&1 | tee -a "$OUT_DIR/engine_validation.txt"
  rrc=${PIPESTATUS[0]}
  fi
  [ "$vrc" -eq 0 ] || vrc=$vrc
  [ "$rrc" -eq 0 ] || vrc=$rrc
  [ "$vrc" -eq 0 ] || die "engine validation failed (exit $vrc) — the engines in $OUT_DIR are not fit to measure with.
  Fix the build (precision fallback? wrong calibration cache? corrupt engine?) and rerun.
  To bypass for debugging only: SKIP_ENGINE_VALIDATION=1"
fi

say "done:"
say "  candidate : $CAND"
say "  fp32 ref  : $REF"
say "  censuses  : ${CAND%.engine}_profile.json, ${REF%.engine}_profile.json"
say "  hashes    : $OUT_DIR/engines.sha256"
say "  mix       : $MIX"
