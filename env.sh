# Source this before running anything in this repository:  . ./env.sh
# Only MODEL_ROOT normally needs editing on a new machine.

# --- the repository itself (auto-detected) -------------------------------------
KIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export KIT_ROOT
export RESULTS_ROOT="${RESULTS_ROOT:-$KIT_ROOT/results}"
export CONFIG_ROOT="${CONFIG_ROOT:-$KIT_ROOT/configs}"
export FIGS_ROOT="${FIGS_ROOT:-$KIT_ROOT/figs}"

# --- platform --------------------------------------------------------------
# Every platform-specific line below keys off this one value, so a Jetson never
# executes a discrete branch and a discrete card never executes a Jetson one.
# BENCH_PLATFORM=jetson|discrete forces it, for exercising the other path.
if [ -n "${BENCH_PLATFORM:-}" ];  then PLATFORM="$BENCH_PLATFORM"
elif [ -f /etc/nv_tegra_release ]; then PLATFORM=jetson
else                                   PLATFORM=discrete; fi
export PLATFORM

# --- what YOU provide on a new machine -------------------------------------
# Model repositories, checkpoints and ONNX files. Everything the pipeline needs
# from outside this repository lives under here: <MODEL_ROOT>/<model_a>,
# <MODEL_ROOT>/<model_b>, ...
export MODEL_ROOT="${MODEL_ROOT:-$HOME/models}"

# Where built TensorRT engines are written and read. Engines are never shipped.
export ENGINE_ROOT="${ENGINE_ROOT:-$KIT_ROOT/engines}"

# Sample/calibration inputs (frames, audio, images) used by the accuracy gates.
export INPUTS_ROOT="${INPUTS_ROOT:-$MODEL_ROOT/inputs}"

# Scratch/working area for engine builds and intermediate artifacts.
export WORK_ROOT="${WORK_ROOT:-$KIT_ROOT/work}"

# --- generative stacks (only needed for the LLM / VLM / ASR rows) ------------
export EDGELLM_ROOT="${EDGELLM_ROOT:-$HOME/tools/TensorRT-Edge-LLM}"
export TRTLLM_ROOT="${TRTLLM_ROOT:-$HOME/tools/TensorRT-LLM}"
export LLM_WORKSPACE="${LLM_WORKSPACE:-$WORK_ROOT/llm-workspace}"      # quantize/export
# An untracked .bench_env.sh beside this file is picked up automatically, so each
# machine can describe its own runtime without editing anything tracked.
export BENCH_ENV_SH="${BENCH_ENV_SH:-}"          # a file to source first (sets LD_LIBRARY_PATH etc.)
[ -z "$BENCH_ENV_SH" ] && [ -r "$KIT_ROOT/.bench_env.sh" ] && export BENCH_ENV_SH="$KIT_ROOT/.bench_env.sh"
export BENCH_PY="${BENCH_PY:-}"                  # the interpreter that has the runtime
if [ "$PLATFORM" = discrete ]; then
  # TensorRT-LLM is normally installed into its own venv rather than against the
  # system interpreter; BENCH_ENV_SH / BENCH_PY are how the stages reach it.
  [ -d "$TRTLLM_ROOT" ] || [ ! -d "${TRTLLM_ROOT}-main" ] || export TRTLLM_ROOT="${TRTLLM_ROOT}-main"
fi

# --- shipped-in results from the other device ------------------------------
export HANDOFF_ROOT="${HANDOFF_ROOT:-$RESULTS_ROOT/handoff}"

# --- names the kit scripts require (aliases of the roots above) --------------
export STAGE_KIT="${STAGE_KIT:-$KIT_ROOT/scripts}"      # the stage runners' kit root
export ONNX_DIR="${ONNX_DIR:-$MODEL_ROOT/onnx}"          # per-family ONNX locations
export VA_ONNX_DIR="${VA_ONNX_DIR:-$ONNX_DIR/va}"     # vision-action policy models
export VFM_ONNX_DIR="${VFM_ONNX_DIR:-$ONNX_DIR/vfm}"   # vision foundation model backbones
# trtexec takes --staticPlugins=<file>, so this must be the .so itself, not a directory.
export EDGELLM_PLUGIN_PATH="${EDGELLM_PLUGIN_PATH:-$EDGELLM_ROOT/build/libNvInfer_edgellm_plugin.so}"

# --- toolchain -------------------------------------------------------------
# JetPack puts trtexec in /usr/src/tensorrt/bin; a discrete box is usually a
# tarball under /opt/tensorrt/<version>. Each platform adds only its own location.
if [ "$PLATFORM" = jetson ]; then
  export PATH="$PATH:/usr/src/tensorrt/bin"
elif ! command -v trtexec >/dev/null 2>&1; then   # never shadow one already on PATH
  for _d in ${TENSORRT_ROOT:+"$TENSORRT_ROOT/bin"} /opt/tensorrt/*/bin; do
    [ -x "$_d/trtexec" ] && { export PATH="$PATH:$_d"; break; }
  done
  unset _d
fi
[ -n "${PYTHONPATH:-}" ] || export PYTHONPATH="$KIT_ROOT/scripts"

mkdir -p "$ENGINE_ROOT" "$WORK_ROOT" 2>/dev/null || true
