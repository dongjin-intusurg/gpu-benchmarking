#!/usr/bin/env bash
#=============================================================================
# Phase 0 — check (and optionally install) everything the benchmark suite needs.
#
#   ./prereqs.sh              report only; prints the exact fix for anything missing
#   ./prereqs.sh --install    run the apt/pip commands for what is missing (needs sudo)
#   ./prereqs.sh --tier figures      only what the figure regeneration needs
#   ./prereqs.sh --tier measure      + everything the hardware measurement needs (default)
#   ./prereqs.sh --tier all          + the generative families and the deck generators
#
# Exit code: 0 = every requirement in the tier is satisfied, 1 = something blocks.
#=============================================================================
set -uo pipefail
G="\033[32m"; Y="\033[33m"; R="\033[31m"; Z="\033[0m"
KITDIR="$(cd "$(dirname "$0")" && pwd)"
PYL="${PYL:-$HOME/tools/bench-pylib}"                 # where --install puts things
# detection searches every plausible prefix, not just the install target
PYL_SEARCH="${PYTHONPATH:-}"
for d in "$PYL" "$HOME/tools/edgellm-pylib" "$HOME/tools/bench-pylib"; do
  [ -d "$d" ] && case ":$PYL_SEARCH:" in *":$d:"*) ;; *) PYL_SEARCH="${PYL_SEARCH:+$PYL_SEARCH:}$d";; esac
done
pyimp(){ python3 -c "import $1;print(getattr($1,'__version__','present'))" 2>/dev/null \
      || PYTHONPATH="$PYL_SEARCH" python3 -c "import $1;print(getattr($1,'__version__','present'))" 2>/dev/null; }
# Some runtimes are deliberately not installed against the system interpreter — a
# discrete box usually keeps TensorRT-LLM in its own venv with an out-of-tree MPI on
# LD_LIBRARY_PATH. Point BENCH_ENV_SH at a file that sets that environment up and/or
# BENCH_PY at the interpreter, and the probe uses them instead of guessing.
export BENCH_ENV_SH="${BENCH_ENV_SH:-}"
[ -z "$BENCH_ENV_SH" ] && [ -r "$KITDIR/.bench_env.sh" ] && export BENCH_ENV_SH="$KITDIR/.bench_env.sh"
export BENCH_PY="${BENCH_PY:-${TRTLLM_PY:-}}"
[ -z "$BENCH_PY" ] && [ -x "$HOME/venv_trtllm/bin/python" ] && export BENCH_PY="$HOME/venv_trtllm/bin/python"
pyimp_alt(){ # module -> version via BENCH_ENV_SH / BENCH_PY; empty if neither resolves it
  BENCH_MOD="$1" bash <<'EOS' 2>/dev/null | tail -1
[ -n "${BENCH_ENV_SH:-}" ] && [ -r "$BENCH_ENV_SH" ] && . "$BENCH_ENV_SH" >/dev/null 2>&1
PY="${BENCH_PY:-${TRTLLM_PY:-python3}}"
[ -x "$PY" ] || PY=$(command -v "$PY" 2>/dev/null) || exit 0
[ -n "$PY" ] || exit 0
"$PY" -c 'import importlib,os
m=importlib.import_module(os.environ["BENCH_MOD"])
print(getattr(m,"__version__","present"))'
EOS
}

TIER=measure; DO_INSTALL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --install) DO_INSTALL=1; shift;;
    --tier) TIER="$2"; shift 2;;
    -h|--help) sed -n '2,12p' "$0"; exit 0;;
    *) echo "unknown option: $1"; exit 2;;
  esac
done
declare -a APT=() PIP=() NOTES=()
MISS=0; OKN=0
ok(){   printf "  ${G}ok${Z}    %-26s %s\n" "$1" "${2:-}"; OKN=$((OKN+1)); }
miss(){ printf "  ${R}MISS${Z}  %-26s %s\n" "$1" "${2:-}"; MISS=$((MISS+1)); }
note(){ printf "  ${Y}note${Z}  %-26s %s\n" "$1" "${2:-}"; }
want(){ case "$TIER:$1" in figures:figures) return 0;; measure:figures|measure:measure) return 0;; all:*) return 0;; *) return 1;; esac; }

echo "=== platform ==="
. /etc/os-release 2>/dev/null || true
printf "  %-26s %s (%s)\n" "os" "${PRETTY_NAME:-unknown}" "$(uname -m)"
if [ -f /etc/nv_tegra_release ]; then
  printf "  %-26s %s\n" "platform" "Jetson — $(sed -n 's/# \(R[0-9]*\) (release), REVISION: \([0-9.]*\).*/\1 rev \2/p' /etc/nv_tegra_release)"
  IS_JETSON=1
else
  printf "  %-26s %s\n" "platform" "discrete"
  IS_JETSON=0
fi
echo

#--------------------------------------------------------------- python modules
echo "=== python modules ==="
pym(){ # module  tier  pip-name  note
  want "$2" || return 0
  local v; v=$(python3 -c "import $1;print(getattr($1,'__version__','present'))" 2>/dev/null)
  if [ -n "$v" ]; then ok "$1" "$v"; return; fi
  v=$(PYTHONPATH="$PYL_SEARCH" python3 -c "import $1;print(getattr($1,'__version__','present'))" 2>/dev/null)
  if [ -n "$v" ]; then note "$1" "$v — only via PYTHONPATH=$PYL_SEARCH"; OKN=$((OKN+1)); return; fi
  miss "$1" "${4:-}"; PIP+=("$3")
}
pym numpy      figures numpy
pym matplotlib figures matplotlib
pym PIL        figures pillow
pym onnx       measure onnx        "needed by compute_arch_gflops.py"
pym tensorrt   measure ""          "ships with the TensorRT apt packages, not pip"
pym torch      measure ""          "Jetson: the JetPack wheel; discrete: a cu13x build"
pym scipy      measure scipy
pym transformers all transformers  "generative accuracy gates only"
pym pptx       all python-pptx     "deck generators only"
echo

#--------------------------------------------------------------- binaries
echo "=== command line tools ==="
bin(){ # name tier apt-package extra-search-path note
  want "$2" || return 0
  local p; p=$(command -v "$1" 2>/dev/null)
  if [ -z "$p" ] && [ -n "${4:-}" ] && [ -x "$4/$1" ]; then
    note "$1" "at $4 — not on PATH; add:  export PATH=\$PATH:$4"; OKN=$((OKN+1)); return
  fi
  if [ -n "$p" ]; then ok "$1" "$p"; else miss "$1" "${5:-}"; [ -n "${3:-}" ] && APT+=("$3"); fi
}
bin trtexec  measure ""            /usr/src/tensorrt/bin  "ships in /usr/src/tensorrt/bin with libnvinfer-bin"
bin g++      measure build-essential
bin cmake    measure cmake
bin nvcc     measure ""            /usr/local/cuda/bin    "CUDA toolkit"
bin envsubst figures gettext-base
bin nsys     measure ""            /usr/local/cuda/bin    "Nsight Systems — timeline stage"
bin ncu      measure ""            /usr/local/cuda/bin    "Nsight Compute — counter stage"
bin nvidia-smi measure ""          /usr/sbin
if [ "$IS_JETSON" = 1 ]; then
  bin nvpmodel      measure "" /usr/sbin
  bin jetson_clocks measure "" /usr/bin
fi
echo

#--------------------------------------------------------------- dev headers
if want measure; then
echo "=== TensorRT / CUDA development files ==="
# Where the headers live depends on how TensorRT got here, and the two platforms
# differ. JetPack installs them as apt packages under the multiarch include dir;
# a discrete box is normally a tarball (or a container) where they sit beside the
# trtexec that is actually on PATH. Deriving the root from that binary matters on
# a machine carrying several /opt/tensorrt/<ver> trees — we must check the one the
# builds will really use, not whichever sorts last.
MULTIARCH=$(dpkg-architecture -qDEB_HOST_MULTIARCH 2>/dev/null || echo "$(uname -m)-linux-gnu")
TRT_ROOT="${TENSORRT_ROOT:-}"
if [ -z "$TRT_ROOT" ]; then
  _tx=$(command -v trtexec 2>/dev/null || true)
  [ -z "$_tx" ] && [ -x /usr/src/tensorrt/bin/trtexec ] && _tx=/usr/src/tensorrt/bin/trtexec
  [ -n "$_tx" ] && TRT_ROOT=$(cd "$(dirname "$_tx")/.." 2>/dev/null && pwd)
fi
TRT_INC_DIRS="/usr/include/$MULTIARCH /usr/include"
[ -n "$TRT_ROOT" ] && [ -d "$TRT_ROOT/include" ] && TRT_INC_DIRS="$TRT_ROOT/include $TRT_INC_DIRS"
# An apt-managed TensorRT can be repaired with apt; a tarball must not be, because
# the apt candidate is frequently a different major version and would shadow it.
case "${TRT_ROOT:-}" in
  ""|/usr|/usr/src/tensorrt) TRT_KIND=apt;;
  *)                         TRT_KIND=tarball;;
esac
[ "$IS_JETSON" = 1 ] && TRT_KIND=apt          # JetPack is always apt-managed
printf "  %-26s %s\n" "header search path" "$TRT_INC_DIRS"

hdr(){ local pkg="$1" file="$2" d found=""
  for d in $TRT_INC_DIRS; do [ -e "$d/$file" ] && { found="$d/$file"; break; }; done
  if [ -n "$found" ]; then ok "$pkg" "$found"
  elif dpkg -l "$pkg" >/dev/null 2>&1; then ok "$pkg" "installed"
  elif [ "$TRT_KIND" = tarball ]; then
    miss "$pkg" "$file not found under $TRT_INC_DIRS — extract the full TensorRT tarball (do NOT apt-install over it)"
  else
    miss "$pkg" "$file absent"; APT+=("$pkg")
  fi
}
hdr libnvinfer-dev              NvInfer.h
hdr libnvinfer-headers-dev      NvInferRuntime.h
hdr libnvinfer-plugin-dev       NvInferPlugin.h
hdr libnvinfer-headers-plugin-dev NvInferPluginBase.h
[ -e /usr/local/cuda/lib64/libcudart.so ] && ok "libcudart" "/usr/local/cuda/lib64" || { miss "libcudart" "the C++ harnesses will not link"; }
V=$(dpkg -l libnvinfer10 2>/dev/null | awk '/^ii/{print $3}')
[ -z "$V" ] && V=$(python3 -c "import tensorrt;print(tensorrt.__version__)" 2>/dev/null)
[ -n "$V" ] && note "TensorRT version" "$V — note the full three-component version; some engine layouts encode it"
[ "$TRT_KIND" = tarball ] && note "TensorRT install" "tarball at ${TRT_ROOT:-?} — --install will not apt-install over it"
echo
fi

#--------------------------------------------------------------- permissions
if want measure; then
echo "=== permissions and profiling ==="
if [ "$(id -u)" = 0 ]; then note "running as root" "fine, but the pipeline expects a normal user with sudo"
elif sudo -n true 2>/dev/null; then ok "sudo" "cached — keep 'sudo -v' alive in a second terminal"
else note "sudo" "will prompt; clock locks and the memory dial need it"; fi
R=$(cat /proc/driver/nvidia/params 2>/dev/null | sed -n 's/^RestrictProfilingToAdminUsers: //p')
if [ "${R:-1}" = "0" ]; then ok "counter profiling" "unrestricted"
else note "counter profiling" "restricted — run ncu under sudo, or set NVreg_RestrictProfilingToAdminUsers=0"; fi
echo
fi

#--------------------------------------------------------------- serving runtimes
if want all; then
echo "=== generative serving runtime ==="
echo "  (only the LLM/VLM and speech models need this; the vision models do not)"
if [ "$IS_JETSON" = 1 ]; then
  ER="${EDGELLM_ROOT:-$HOME/tools/TensorRT-Edge-LLM}"
  if [ -d "$ER" ]; then ok "Edge-LLM tree" "$ER"
  else miss "Edge-LLM tree" "set EDGELLM_ROOT, or clone and build it — see below"; NEED_EDGELLM=1; fi
  for b in llm_bench llm_inference; do
    if [ -x "$ER/build/examples/llm/$b" ]; then ok "$b" "$ER/build/examples/llm/$b"
    else miss "$b" "not built — the LLM rows are driven by this binary"; NEED_EDGELLM=1; fi
  done
  if [ -e "$ER/build/libNvInfer_edgellm_plugin.so" ]; then ok "edgellm plugin" "built"
  else miss "edgellm plugin" "libNvInfer_edgellm_plugin.so absent"; NEED_EDGELLM=1; fi
  V=$(pyimp tensorrt_edgellm)
  [ -n "$V" ] && ok "tensorrt_edgellm" "$V" || { miss "tensorrt_edgellm" "the python side of the same tree"; NEED_EDGELLM=1; }
else
  TR="${TRTLLM_ROOT:-$HOME/tools/TensorRT-LLM}"
  V=$(pyimp tensorrt_llm)
  if [ -n "$V" ]; then ok "tensorrt_llm" "$V"
  else
    V=$(pyimp_alt tensorrt_llm)
    if [ -n "$V" ]; then
      note "tensorrt_llm" "$V — via ${BENCH_ENV_SH:-$BENCH_PY}, not the system interpreter"
      note "" "the stages must run under that same environment"
      OKN=$((OKN+1))
    else miss "tensorrt_llm" "the discrete card's serving runtime"; NEED_TRTLLM=1; fi
  fi
  [ -d "$TR" ] && ok "TensorRT-LLM tree" "$TR" || note "TRTLLM_ROOT" "unset — only needed for the repo's own example scripts"
fi
# the quantizer is pinned, and the pin matters
MV=$(pyimp modelopt)
if [ "$MV" = "0.45.0" ]; then ok "modelopt" "0.45.0 (the pinned version)"
elif [ -n "$MV" ]; then miss "modelopt" "$MV — MUST be 0.45.0; 0.46 breaks the quant configs"; PIP+=("nvidia-modelopt==0.45.0")
else miss "modelopt" "needed to quantize the LLM rungs"; PIP+=("nvidia-modelopt==0.45.0"); fi
echo
fi

#--------------------------------------------------------------- fix list
if [ ${#APT[@]} -gt 0 ] || [ ${#PIP[@]} -gt 0 ]; then
  echo "=== how to fix ==="
  if [ ${#APT[@]} -gt 0 ]; then
    A=$(printf "%s\n" "${APT[@]}" | sort -u | tr '\n' ' ')
    echo "  sudo apt update && sudo apt install -y $A"
  fi
  if [ ${#PIP[@]} -gt 0 ]; then
    P=$(printf "%s\n" "${PIP[@]}" | grep -v '^$' | sort -u | tr '\n' ' ')
    [ -n "$P" ] && {
      echo "  # this platform enforces PEP 668 — install to a directory and use PYTHONPATH:"
      echo "  pip install --break-system-packages --target $PYL $P"
      echo "  export PYTHONPATH=$PYL"
    }
  fi
  if [ "$DO_INSTALL" = 1 ]; then
    echo
    echo "=== installing ==="
    [ ${#APT[@]} -gt 0 ] && { sudo apt update && sudo apt install -y $(printf "%s\n" "${APT[@]}" | sort -u); }
    [ -n "${P:-}" ] && pip install --break-system-packages --target "$PYL" $P
    echo "  re-run ./prereqs.sh to confirm"
  else
    echo
    echo "  re-run with --install to have this script do it"
  fi
  echo
fi

#--------------------------------------------------------------- source builds
install_edgellm(){
  local ER="${EDGELLM_ROOT:-$HOME/tools/TensorRT-Edge-LLM}"
  local REPO="${EDGELLM_REPO:-https://github.com/NVIDIA/TensorRT-Edge-LLM.git}"
  local REF="${EDGELLM_REF:-bb29145}"     # 0.10.0 — a known-good pinned build
  echo
  echo "=== installing TensorRT Edge-LLM ==="
  echo "  repo $REPO  ref $REF  ->  $ER"
  if [ ! -d "$ER/.git" ]; then
    git clone "$REPO" "$ER" || { echo "  clone failed"; return 1; }
    git -C "$ER" checkout -q "$REF" && echo "  checkout      pinned to $REF" \
      || echo "  checkout      could not pin $REF — using $(git -C "$ER" rev-parse --short HEAD)"
  else
    # never rewrite a tree the operator already has — report and let them decide
    local HEAD; HEAD=$(git -C "$ER" rev-parse --short HEAD 2>/dev/null)
    if [ "$HEAD" = "$REF" ] || git -C "$ER" merge-base --is-ancestor "$REF" HEAD 2>/dev/null; then
      echo "  clone         already present at $HEAD"
    else
      echo "  clone         already present at $HEAD — this suite pins $REF."
      echo "                Not touching your tree. To match exactly: git -C $ER checkout $REF"
    fi
  fi

  # apply the large-model quantize patch before anything imports the package
  local PATCH; PATCH=$(find "$KITDIR/patches" -name '*.patch' 2>/dev/null | head -1)
  if [ -n "$PATCH" ]; then
    if git -C "$ER" apply --reverse --check "$PATCH" >/dev/null 2>&1; then echo "  quantize patch already applied"
    elif git -C "$ER" apply --check "$PATCH" >/dev/null 2>&1; then
      git -C "$ER" apply "$PATCH" && echo "  quantize patch applied"
    else echo "  patch does not apply cleanly to $REF — review before a large-model quantize"; fi
  elif [ -d "$KITDIR/patches" ]; then :; fi

  if [ -x "$ER/build/examples/llm/llm_bench" ] && [ -e "$ER/build/libNvInfer_edgellm_plugin.so" ]; then
    echo "  native build  already built"
  else
    local CC ARCH
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
    ARCH="${CUDA_ARCH:-${CC:-110}}"
    echo "  native build  cmake -DCMAKE_CUDA_ARCHITECTURES=$ARCH  (this takes a while)"
    ( cd "$ER" && cmake -B build -DCMAKE_CUDA_ARCHITECTURES="$ARCH" >/dev/null \
        && cmake --build build -j"$(nproc)" ) || { echo "  native build FAILED — see the cmake output above"; return 1; }
    echo "  native build  done"
  fi

  if PYTHONPATH="$PYL_SEARCH" python3 -c "import tensorrt_edgellm" 2>/dev/null; then
    echo "  python pkg    already installed"
  else
    echo "  python pkg    pip install --target $PYL ."
    pip install --break-system-packages --target "$PYL" "$ER" || { echo "  python install FAILED"; return 1; }
  fi
  echo "  add to your shell:  export EDGELLM_ROOT=$ER PYTHONPATH=$PYL"
}

install_trtllm(){
  echo
  echo "=== installing TensorRT-LLM ==="
  if PYTHONPATH="$PYL_SEARCH" python3 -c "import tensorrt_llm" 2>/dev/null; then
    echo "  already installed ($(PYTHONPATH="$PYL_SEARCH" python3 -c 'import tensorrt_llm;print(tensorrt_llm.__version__)'))"
  else
    pip install --break-system-packages --target "$PYL" tensorrt-llm || {
      echo "  pip install failed — on a discrete card NVIDIA's container is the supported path"; return 1; }
  fi
  echo "  record the exact version with the results — the decode numbers move with it"
}

if [ "$DO_INSTALL" = 1 ]; then
  [ "${NEED_EDGELLM:-0}" = 1 ] && install_edgellm
  [ "${NEED_TRTLLM:-0}" = 1 ]  && install_trtllm
fi
if [ "${NEED_EDGELLM:-0}" = 1 ] && [ "$DO_INSTALL" = 0 ]; then
  echo "  Edge-LLM is a source build — re-run with --install and this script will clone,"
  echo "  patch, cmake-build and pip-install it (${EDGELLM_REPO:-https://github.com/NVIDIA/TensorRT-Edge-LLM.git} @ ${EDGELLM_REF:-bb29145})."
  echo
fi
if [ "${NEED_TRTLLM:-0}" = 1 ] && [ "$DO_INSTALL" = 0 ]; then
  echo "  TensorRT-LLM missing — re-run with --install, or use NVIDIA's container."
  echo
fi

echo "=== summary ==="
printf "  tier %s: %d satisfied, %d missing\n" "$TIER" "$OKN" "$MISS"
if [ "$MISS" -gt 0 ]; then printf "  ${R}not ready${Z} — fix the above, then re-run\n"; exit 1; fi
printf "  ${G}ready${Z} — all prerequisites for the requested tier are satisfied\n"
