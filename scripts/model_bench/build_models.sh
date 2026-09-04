#!/usr/bin/env bash
#=============================================================================
# build_models.sh - build (or adopt) every artifact the registry declares.
#
#   ./build_models.sh                 every registered model
#   ./build_models.sh --only <model>  one model (repeatable)
#   ./build_models.sh --list          what would be built, nothing runs
#
# Runs UNLOCKED: builds are untimed. It refuses to start while anything is
# measuring on this GPU (a build contaminates a live measurement and vice
# versa). Per model x builder x precision it dispatches to the builder the
# registration names and validates what came out before anything measures it:
#
#   trt      trtexec via build_model_engines.sh (candidate + fp32 reference,
#            layer census, sha256, build_stamp). A MODEL_ENGINE_SET model
#            builds one engine per member. A stamped build is reused; an
#            engine already present in the model's engine dir
#            (<name>_<prec>.engine, or <member>_<prec>.engine) is load-gated
#            and taken as is - no rebuild while the file is there.
#   adopt    a prebuilt .engine supplied by its repo: linked, load-gated,
#            recorded as adopted (no census unless the repo shipped one).
#   edgellm  Edge-LLM quantize -> export -> llm_build (+ visual_build once);
#            an engine dir that already validates is reused and stamped.
#   trtllm   modelopt quantization to a serving checkpoint dir (the runtime
#            loads it directly; no engine file is the normal state).
#
# A builder that fails falls back to `trt` ONLY when the registration lists
# trt after it and supplies MODEL_ONNX; the row then records
# builder_requested vs builder_used. Anything else is a hard stop.
#
# Output: one JSON line per measurable row in
#   $ENGINE_ROOT/<device_tag>/registry_rows/<model>.jsonl
# which measure_models.sh consumes. Engines land in the model's engine dir
# (default $ENGINE_ROOT/<device_tag>/<model>/).
#=============================================================================
# shellcheck disable=SC1091
. "$(cd "$(dirname "$0")" && pwd)/stage4_common.sh"
parse_only "$@"
discover_device
[ -d "$MANIFEST_DIR" ] || die "no $MANIFEST_DIR - run ./configure.sh first (it expands configs/manifests/ into it)"

ROWS_DIR="$ENGINE_ROOT/$DEVICE_TAG/registry_rows"
BUILD_LOG_DIR="$ENGINE_ROOT/$DEVICE_TAG/build_logs"
mkdir -p "$ROWS_DIR" "$BUILD_LOG_DIR"
STAMP_DATE="$(date +%Y-%m-%dT%H:%M:%S)"
TRT_VER="$(python3 -c 'import tensorrt; print(tensorrt.__version__)' 2>/dev/null || true)"
[ -n "$TRT_VER" ] || TRT_VER="$(trtexec --help 2>&1 | grep -m1 -i tensorrt | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
[ -n "$TRT_VER" ] || TRT_VER=unknown

say "device   : $(basename "$DEVICE_CFG")  platform=$PLATFORM  tag=$DEVICE_TAG  trt=$TRT_VER"
say "registry : $MANIFEST_DIR"
say "engines  : $ENGINE_ROOT/$DEVICE_TAG"

# every registration must parse before any GPU minute is spent
_v=$(python3 "$REGISTRY" validate $(registry_args) "${ONLY[@]}" 2>&1) || { echo "$_v"; die "registry has errors - fix the manifests above, then rerun"; }
[ "$LIST" = 1 ] || echo "$_v"
NAMES=$(registry_names "${ONLY[@]}")
[ -n "$NAMES" ] || die "nothing registered"

#-----------------------------------------------------------------------------
# helpers
#-----------------------------------------------------------------------------
# one JSON object per line, typed from key=value pairs (numbers/bools/JSON
# literals parse, everything else is a string)
jrow(){
  python3 - "$@" <<'PY' >> "$ROWS"
import json, sys
d = {}
for a in sys.argv[1:]:
    k, _, v = a.partition('=')
    try:
        d[k] = json.loads(v) if v and v[0] in '0123456789-[{tfn' else v
    except Exception:
        d[k] = v
print(json.dumps(d))
PY
}
sha(){ sha256sum "$1" | awk '{print $1}'; }
fsize(){ stat -c %s "$1" 2>/dev/null || echo 0; }
# the run-time trtexec string for an engine row: plugins + shapes + extra (+ member flags)
run_flags(){
  local f=""
  [ -z "$R_PLUGINS" ] || f="--staticPlugins=$R_PLUGINS"
  [ -z "$R_SHAPES" ] || f="$f --shapes=$R_SHAPES"
  [ -z "$R_EXTRA_ARGS" ] || f="$f $R_EXTRA_ARGS"
  [ -z "${1:-}" ] || f="$f $1"
  echo "${f# }"
}
in_list(){ local x=$1; shift; for y in "$@"; do [ "$x" = "$y" ] && return 0; done; return 1; }

#-----------------------------------------------------------------------------
# trt: one engine from one ONNX, through build_model_engines.sh
#   $1 row name (build dir + rows entry)  $2 engine base name (MODEL_NAME)
#   $3 onnx  $4 precision  $5 primary(1|0)  $6 build args  $7 shapes
#   $8 arch_gflops  $9 flat link name (without _<prec>.engine)
#   $10 builder_requested  $11 extra member run flags
#-----------------------------------------------------------------------------
build_trt_one(){
  local row=$1 base=$2 onnx=$3 prec=$4 primary=$5 bargs=$6 shapes=$7 agf=$8 link=$9 breq=${10} mflags=${11:-}
  local bdir="$R_ENGINE_DIR/build/${row}" man="$R_ENGINE_DIR/build/manifest_${row}.env"
  mkdir -p "$R_ENGINE_DIR/build"
  # a synthesized single-engine manifest: the existing builder's contract
  {
    echo "MODEL_NAME=$base"
    echo "MODEL_ONNX=$onnx"
    echo "MODEL_PRECISION=$prec"
    echo "MODEL_HZ=$R_HZ"
    echo "MODEL_DEADLINE_MS=$R_DEADLINE_MS"
    [ "$prec" = int8 ] && echo "MODEL_CALIB_CACHE=$R_CALIB_CACHE"
    echo "MODEL_PLUGINS=$R_PLUGINS"
    echo "MODEL_SHAPES=$shapes"
    echo "MODEL_EXTRA_ARGS=$R_EXTRA_ARGS"
    echo "MODEL_BUILD_ARGS=$bargs"
    echo "ACC_MANIFEST=$R_ACC_MANIFEST"
    echo "MODEL_ARCH_GFLOPS=$agf"
  } > "$man"
  local log="$BUILD_LOG_DIR/${row}.log" skip=0
  [ "$primary" = 1 ] || skip=1
  # An engine already present in the model's engine dir that this kit did not
  # build (not a link into build/) is taken as is: load-gated at the declared
  # precision, recorded as pre-existing, never rebuilt. Drop or relink the file
  # to force a build.
  local pre="$R_ENGINE_DIR/${link}_${prec}.engine"
  if [ -e "$pre" ] && [[ "$(readlink -f "$pre")" != "$R_ENGINE_DIR/build/"* ]]; then
    local rf; rf=$(run_flags "$mflags")
    say "  [trt] $row  pre-existing $(readlink -f "$pre") - load gate, build skipped"
    if ! python3 "$MB_DIR/validate_engines.py" "$pre" --precision "$prec" ${rf:+--extra="$rf"} > "$log" 2>&1; then
      tail -12 "$log"; say "  [trt] $row: pre-existing engine failed the load gate - see $log"; return 1
    fi
    local preref="$R_ENGINE_DIR/${link}_fp32_ref.engine"; [ -e "$preref" ] || preref=""
    jrow row="$row" model="$R_NAME" kind=engine engine="$pre" precision="$prec" \
         builder_requested="$breq" builder_used=trt adopted=true source_engine="$(readlink -f "$pre")" \
         run_flags="$rf" hz="$R_HZ" deadline_ms="$R_DEADLINE_MS" arch_gflops="${agf:-0}" \
         engine_bytes="$(fsize "$pre")" stamp="preexisting sha256=$(sha "$pre") precision=$prec" fp32_ref="$preref" \
         built_at="$STAMP_DATE" trt="$TRT_VER"
    return 0
  fi
  say "  [trt] $row  <- $(basename "$onnx")  ($prec$([ $skip = 1 ] && echo ', no fp32 ref'))"
  if ! DEVICE_CFG="$DEVICE_CFG" DEVICE_TAG="$DEVICE_TAG" SKIP_FP32_REF=$skip MIX_DIR="$bdir" \
       "$MB_DIR/build_model_engines.sh" "$man" "$bdir" > "$log" 2>&1; then
    tail -15 "$log"; say "  [trt] $row FAILED - see $log"; return 1
  fi
  local eng="$bdir/${base}_${prec}.engine"
  [ -f "$eng" ] || { say "  [trt] $row: builder exited 0 but $eng is missing"; return 1; }
  ln -sfn "$eng" "$R_ENGINE_DIR/${link}_${prec}.engine"
  [ -f "$bdir/${base}_${prec}_profile.json" ] && ln -sfn "$bdir/${base}_${prec}_profile.json" "$R_ENGINE_DIR/${link}_${prec}_profile.json"
  # the fp32 reference lives with the primary precision's build
  local ref="$R_ENGINE_DIR/build/${base}/${base}_fp32_ref.engine"
  [ $skip = 1 ] || ln -sfn "$ref" "$R_ENGINE_DIR/${link}_fp32_ref.engine"
  # arch gflops may have been computed by the builder: read it back from its mix
  local a="$agf"
  [ -n "$a" ] || a=$(tail -1 "$bdir/mix_${DEVICE_TAG}_${base}.csv" 2>/dev/null | cut -d, -f6)
  jrow row="$row" model="$R_NAME" kind=engine engine="$eng" precision="$prec" \
       builder_requested="$breq" builder_used=trt adopted=false \
       run_flags="$(run_flags "$mflags")" hz="$R_HZ" deadline_ms="$R_DEADLINE_MS" arch_gflops="${a:-0}" \
       engine_bytes="$(fsize "$eng")" stamp="$(cat "$bdir/build_stamp" 2>/dev/null)" fp32_ref="$ref" \
       built_at="$STAMP_DATE" trt="$TRT_VER"
}

build_trt(){ # $1 builder_requested
  local breq=$1 p
  for p in $R_PRECISIONS; do
    local primary=0; [ "$p" = "$R_PRIMARY_PRECISION" ] && primary=1
    if [ -z "$R_MEMBERS" ]; then
      local row="$R_NAME"; [ $primary = 1 ] || row="${R_NAME}_$p"
      build_trt_one "$row" "$R_NAME" "$R_ONNX" "$p" $primary "$R_BUILD_ARGS" "$R_SHAPES" "$R_ARCH_GFLOPS" "$R_NAME" "$breq" || return 1
    else
      local m
      for m in $R_MEMBERS; do
        local onnx bargs shapes agf rflags
        local v
        v="R_ONNX_$m"; onnx="${!v:-}";  v="R_BUILD_FLAGS_$m"; bargs="${!v:-}"
        v="R_SHAPES_$m"; shapes="${!v:-}"; v="R_ARCH_GFLOPS_$m"; agf="${!v:-}"
        v="R_RUN_FLAGS_$m"; rflags="${!v:-}"
        local base="${R_NAME}_$m" row="${R_NAME}_$m"; [ $primary = 1 ] || row="${R_NAME}_${m}_$p"
        # member engines are linked as <member>_<prec>.engine: the layout an
        # e2e driver opens (encoder_fp16.engine, decoder_first_fp16.engine ...)
        build_trt_one "$row" "$base" "$onnx" "$p" $primary "$bargs" "$shapes" "$agf" "$m" "$breq" "$rflags" || return 1
      done
    fi
  done
}

#-----------------------------------------------------------------------------
# adopt: a repo-built engine; we link it, load-gate it, and say so in the row
#-----------------------------------------------------------------------------
build_adopt(){
  local p="$R_PRIMARY_PRECISION" src="$R_ENGINE"
  local link="$R_ENGINE_DIR/${R_NAME}_${p}.engine"
  say "  [adopt] $R_NAME  <- $src"
  [ -f "$src" ] || { say "  [adopt] engine missing: $src"; return 1; }
  ln -sfn "$src" "$link"
  local prof="${src%.engine}_profile.json"
  [ -f "$prof" ] && ln -sfn "$prof" "${link%.engine}_profile.json"
  local log="$BUILD_LOG_DIR/${R_NAME}_adopt.log" rf; rf=$(run_flags)
  if ! python3 "$MB_DIR/validate_engines.py" "$link" --precision "$p" ${rf:+--extra="$rf"} > "$log" 2>&1; then
    tail -12 "$log"; say "  [adopt] $R_NAME failed the load gate - see $log"; return 1
  fi
  jrow row="$R_NAME" model="$R_NAME" kind=engine engine="$link" precision="$p" \
       builder_requested=adopt builder_used=adopt adopted=true source_engine="$src" \
       run_flags="$rf" hz="$R_HZ" deadline_ms="$R_DEADLINE_MS" arch_gflops="${R_ARCH_GFLOPS:-0}" \
       engine_bytes="$(fsize "$src")" stamp="adopted sha256=$(sha "$src")" built_at="$STAMP_DATE" trt="$TRT_VER"
}

#-----------------------------------------------------------------------------
# edgellm: the precision-ladder chain (quantize -> export -> llm_build), one
# llm engine dir per precision, one shared visual engine. Layout is the
# workspace layout the runtime tools already use:
#   fp16   -> <dir>/onnx/llm            <dir>/engines/llm
#   <p>    -> <dir>/ckpt_<p> onnx_<p>   <dir>/engines_<p>/llm
#   visual -> <dir>/onnx/visual         <dir>/engines/visual
#-----------------------------------------------------------------------------
edgellm_py(){ PYTHONPATH="${EDGELLM_PYLIB:-}${PYTHONPATH:+:$PYTHONPATH}" python3 "$@"; }
edgellm_ready(){
  [ -x "$EDGELLM_ROOT/build/examples/llm/llm_build" ] \
    || { say "  Edge-LLM binaries not built at $EDGELLM_ROOT/build/examples/llm (setup.sh builds them)"; return 1; }
  edgellm_py -c 'import tensorrt_edgellm' 2>/dev/null \
    || { say "  the tensorrt_edgellm python frontend does not import (EDGELLM_PYLIB=${EDGELLM_PYLIB:-unset}); setup.sh installs it"; return 1; }
}
llm_stamp(){ echo "ckpt_config_sha256=$(sha "$R_CHECKPOINT/config.json") precision=$1 build_flags='$2' trt=$TRT_VER"; }

build_edgellm(){
  local WS="$R_ENGINE_DIR"; mkdir -p "$WS"
  edgellm_ready || return 1
  local p
  for p in $R_PRECISIONS; do
    local base="${p%_nvflags}" llm_dir onnx_dir flags stamp
    if [ "$p" = fp16 ]; then llm_dir="$WS/engines/llm"; onnx_dir="$WS/onnx"; else llm_dir="$WS/engines_$p/llm"; onnx_dir="$WS/onnx_$base"; fi
    local v="R_LLM_BUILD_FLAGS_${p^^}"; flags="${!v:-}"; [ -n "$flags" ] || flags="$R_LLM_BUILD_FLAGS"
    [ -n "$flags" ] || flags="--maxBatchSize 2 --maxInputLen $R_LLM_MAX_INPUT_LEN --maxKVCacheCapacity 4096 --maxKVPoolPages 64"
    stamp=$(llm_stamp "$p" "$flags")
    local log="$BUILD_LOG_DIR/${R_NAME}_${p}.log"; : > "$log"
    if [ -f "$llm_dir/llm.engine" ]; then
      local have=""; [ -f "$llm_dir/build_stamp" ] && have=$(cat "$llm_dir/build_stamp")
      if [ -n "$have" ] && [ "${have#adopted=preexisting }" != "$stamp" ]; then
        say "  [edgellm] $R_NAME $p: build_stamp mismatch in $llm_dir"
        say "     have: $have"; say "     want: $stamp"
        say "     rm $llm_dir/build_stamp (or the engine dir) to rebuild"; return 1
      fi
      say "  [edgellm] $R_NAME $p: reusing $llm_dir"
      [ -f "$llm_dir/build_stamp" ] || echo "adopted=preexisting $stamp" > "$llm_dir/build_stamp"
    else
      say "  [edgellm] $R_NAME $p: quantize -> export -> llm_build"
      if [ "$base" != fp16 ] && [ ! -f "$WS/ckpt_$base/config.json" ]; then
        edgellm_py -m tensorrt_edgellm.scripts.quantize llm --model_dir "$R_CHECKPOINT" \
          --output_dir "$WS/ckpt_$base" --quantization "$base" --device cuda >> "$log" 2>&1 \
          || { tail -8 "$log"; say "  [edgellm] quantize $base FAILED - $log"; return 1; }
      fi
      if [ ! -f "$onnx_dir/llm/model.onnx" ]; then
        local src="$R_CHECKPOINT"; [ "$base" = fp16 ] || src="$WS/ckpt_$base"
        edgellm_py -m tensorrt_edgellm.scripts.export "$src" "$onnx_dir" >> "$log" 2>&1 \
          || { tail -8 "$log"; say "  [edgellm] export $p FAILED - $log"; return 1; }
      fi
      # shellcheck disable=SC2086
      ( cd "$EDGELLM_ROOT" && ./build/examples/llm/llm_build --onnxDir "$onnx_dir/llm" --engineDir "$llm_dir" $flags ) >> "$log" 2>&1 \
        || { tail -8 "$log"; say "  [edgellm] llm_build $p FAILED - $log"; return 1; }
      [ -f "$llm_dir/llm.engine" ] || { say "  [edgellm] llm_build exited 0 but $llm_dir/llm.engine is missing"; return 1; }
      echo "$stamp" > "$llm_dir/build_stamp"
    fi
    python3 "$MB_DIR/validate_engines.py" --serving "$llm_dir" >> "$log" 2>&1 \
      || { tail -8 "$log"; say "  [edgellm] $llm_dir failed validation - $log"; return 1; }
    local vis_dir="" vis_bytes=0
    if [ "$R_LLM_HAS_VISUAL" = 1 ]; then
      vis_dir="$WS/engines/visual"
      if [ ! -f "$vis_dir/visual.engine" ]; then
        say "  [edgellm] $R_NAME: visual export + build (shared by every precision)"
        [ -f "$WS/onnx/visual/model.onnx" ] || edgellm_py -m tensorrt_edgellm.scripts.export --skip-llm "$R_CHECKPOINT" "$WS/onnx" >> "$log" 2>&1 \
          || { tail -8 "$log"; say "  [edgellm] visual export FAILED - $log"; return 1; }
        ( cd "$EDGELLM_ROOT" && ./build/examples/multimodal/visual_build --onnxDir "$WS/onnx/visual" --engineDir "$WS/engines" \
            --minImageTokens 128 --maxImageTokens 4096 --maxImageTokensPerImage 512 ) >> "$log" 2>&1 \
          || { tail -8 "$log"; say "  [edgellm] visual_build FAILED - $log"; return 1; }
      fi
      python3 "$MB_DIR/validate_engines.py" --serving "$vis_dir" >> "$log" 2>&1 \
        || { tail -8 "$log"; say "  [edgellm] $vis_dir failed validation - $log"; return 1; }
      vis_bytes=$(fsize "$vis_dir/visual.engine")
    fi
    jrow row="${R_NAME}_$p" model="$R_NAME" kind=generative runtime=edgellm precision="$p" \
         builder_requested=edgellm builder_used=edgellm llm_dir="$llm_dir" visual_dir="$vis_dir" \
         checkpoint="$R_CHECKPOINT" battery="$R_LLM_BATTERY" image="$R_LLM_IMAGE" \
         chunk="$R_LLM_CHUNK" context_len="$R_LLM_CONTEXT_LEN" reuse_len="$R_LLM_REUSE_LEN" \
         hz="$R_HZ" deadline_ms="$R_DEADLINE_MS" arch_gflops="${R_ARCH_GFLOPS:-0}" \
         engine_bytes="$(fsize "$llm_dir/llm.engine")" visual_bytes="$vis_bytes" \
         stamp="$(cat "$llm_dir/build_stamp")" built_at="$STAMP_DATE" trt="$TRT_VER"
  done
}

#-----------------------------------------------------------------------------
# trtllm: modelopt PTQ to a serving checkpoint dir; fp16 serves the HF
# checkpoint itself. Validated as a checkpoint dir, not an engine.
#-----------------------------------------------------------------------------
build_trtllm(){
  local WS="$R_ENGINE_DIR"; mkdir -p "$WS"
  local Q="$HANDOFF_ROOT/trtllm_quantize.sh"
  [ -x "$Q" ] || Q="$KIT/model_bench/trtllm/trtllm_quantize.sh"
  local p
  for p in $R_PRECISIONS; do
    local dir log="$BUILD_LOG_DIR/${R_NAME}_${p}.log"
    case "$p" in
      fp16|bf16) dir="$R_CHECKPOINT" ;;
      fp8|nvfp4)
        dir="$WS/trtllm_$p"
        if [ ! -f "$dir/config.json" ]; then
          [ -x "$Q" ] || { say "  [trtllm] quantizer missing: $Q"; return 1; }
          say "  [trtllm] $R_NAME $p: modelopt PTQ -> $dir"
          "$Q" "$R_CHECKPOINT" "$p" "$dir" > "$log" 2>&1 || { tail -8 "$log"; say "  [trtllm] quantize $p FAILED - $log"; return 1; }
        else
          say "  [trtllm] $R_NAME $p: reusing $dir"
        fi ;;
      *) say "  [trtllm] $R_NAME: precision '$p' is an Edge-LLM build variant - no TensorRT-LLM row"; continue ;;
    esac
    python3 "$MB_DIR/validate_engines.py" --serving "$dir" >> "$log" 2>&1 \
      || { tail -8 "$log"; say "  [trtllm] $dir failed validation - $log"; return 1; }
    local bytes; bytes=$(du -sb "$dir" 2>/dev/null | cut -f1)
    jrow row="${R_NAME}_$p" model="$R_NAME" kind=generative runtime=trtllm precision="$p" \
         builder_requested=trtllm builder_used=trtllm serving_dir="$dir" \
         checkpoint="$R_CHECKPOINT" battery="$R_LLM_BATTERY" image="$R_LLM_IMAGE" \
         chunk="$R_LLM_CHUNK" context_len="$R_LLM_CONTEXT_LEN" reuse_len="$R_LLM_REUSE_LEN" \
         hz="$R_HZ" deadline_ms="$R_DEADLINE_MS" arch_gflops="${R_ARCH_GFLOPS:-0}" \
         engine_bytes="${bytes:-0}" stamp="ckpt_config_sha256=$(sha "$R_CHECKPOINT/config.json") precision=$p" \
         built_at="$STAMP_DATE"
  done
}

#-----------------------------------------------------------------------------
# per model: every builder, in registration order; fallback per the rules
#-----------------------------------------------------------------------------
n_ok=0; n_fail=0; FAILED=()
for name in $NAMES; do
  eval "$(registry_export "$name")"
  [ -z "$R_ERRORS" ] || die "$name: $R_ERRORS"
  ROWS="$ROWS_DIR/$name.jsonl"
  read -ra BUILDERS <<< "$R_BUILDERS"
  if [ "$LIST" = 1 ]; then
    echo "  $name  kind=$R_KIND builders=$R_BUILDERS precisions=$R_PRECISIONS${R_MEMBERS:+ members=$R_MEMBERS}${R_E2E_SRC:+ e2e=$(basename "$R_E2E_SRC")}  -> $R_ENGINE_DIR"
    continue
  fi
  refuse_if_measuring
  say "=== $name  ($R_KIND; builders: $R_BUILDERS; precisions: $R_PRECISIONS) ==="
  mkdir -p "$R_ENGINE_DIR"; : > "$ROWS"
  status=0; i=0
  while [ $i -lt ${#BUILDERS[@]} ]; do
    b=${BUILDERS[$i]}; i=$((i+1))
    case "$b" in
      trt)     build_trt trt ;;
      adopt)   build_adopt ;;
      edgellm) build_edgellm ;;
      trtllm)  build_trtllm ;;
      *)       die "$name: unknown builder '$b'" ;;
    esac
    rc=$?
    if [ $rc -ne 0 ]; then
      # fallback: only a LATER trt entry with an ONNX to consume
      if [ "$b" != trt ] && in_list trt "${BUILDERS[@]:$i}" && [ -n "$R_ONNX" ]; then
        say "  $b failed - falling back to trt (recorded as builder_requested=$b builder_used=trt)"
        build_trt "$b" || { status=1; break; }
        break   # the trt entry served as the fallback; do not build it twice
      fi
      status=1; break
    fi
    # a trt entry that follows a non-trt builder is a fallback target only
    [ "$b" != trt ] && in_list trt "${BUILDERS[@]:$i}" && { say "  (trt listed after $b is the fallback target - not built separately)"; break; }
  done
  # the e2e driver row rides the primary engine dir; compiled at measure time
  if [ $status -eq 0 ] && [ -n "$R_E2E_SRC" ]; then
    jrow row="${R_NAME}_e2e" model="$R_NAME" kind=e2e engine_dir="$R_ENGINE_DIR" precision="$R_PRIMARY_PRECISION" \
         src="$R_E2E_SRC" inputs="$R_E2E_INPUTS" args="$R_E2E_ARGS" repeats="$R_E2E_REPEATS" \
         hz="$R_E2E_HZ" deadline_ms="$R_E2E_DEADLINE_MS" model_deadline_ms="$R_DEADLINE_MS" arch_gflops="${R_ARCH_GFLOPS:-0}" built_at="$STAMP_DATE"
  fi
  if [ $status -eq 0 ]; then
    n_ok=$((n_ok+1)); say "  rows: $(wc -l < "$ROWS")  -> $ROWS"
  else
    n_fail=$((n_fail+1)); FAILED+=("$name"); rm -f "$ROWS"
    say "  $name FAILED - no rows written (measure_models.sh will skip it)"
  fi
done

[ "$LIST" = 1 ] && exit 0
echo
say "build summary: $n_ok ok, $n_fail failed"
[ $n_fail -eq 0 ] || { say "  failed: ${FAILED[*]}"; exit 1; }
