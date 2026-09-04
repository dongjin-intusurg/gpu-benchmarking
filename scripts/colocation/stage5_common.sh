# Sourced by the stage-5 scripts (validate_mixes.sh, measure_mixes.sh,
# run_colocation.sh, the arms and side loads). Builds on stage4_common.sh
# (path contract, device config, registry, ceilings) and adds what the
# co-location stage needs: the newest stage-4 solo results, the mix
# directory, the row_loop driver build, the clock lock/release pair and the
# stage-5 argument surface. Not executable on its own.
COLOC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "$COLOC_DIR/../model_bench/stage4_common.sh"
MIX_DIR="${MIX_DIR:-$KIT/../configs/colocation}"
[ -d "$MIX_DIR" ] || MIX_DIR="$KIT/configs/colocation"
ROW_LOOP_SRC="$KIT/model_bench/cpp/row_loop.cpp"     # the merged driver (single-row, --multi, --prio-range)
RUN_SECONDS="${RUN_SECONDS:-90}"       # run length of every paced solo and every arm (env; SECONDS itself is a bash builtin)

# The newest stage-4 solo run of this device: the DEMAND side of every composed
# budget and the certified solo p99 every contention factor is measured against.
# SOLO_RESULTS=<results.json> selects another run (e.g. when the newest one
# measured only a subset of the rows a mix needs).
discover_solo_results(){
  if [ -z "${SOLO_RESULTS:-}" ]; then
    SOLO_RESULTS=$(ls -1dt "$RESULTS_ROOT"/solo_${DEVICE_TAG}_*/results.json 2>/dev/null | head -1)
  fi
  [ -n "$SOLO_RESULTS" ] && [ -f "$SOLO_RESULTS" ] \
    || die "no completed stage-4 run for $DEVICE_TAG under $RESULTS_ROOT (solo_${DEVICE_TAG}_*/results.json) - run ./run_model_solo.sh first"
  ROWS_DIR="$ENGINE_ROOT/$DEVICE_TAG/registry_rows"
  [ -d "$ROWS_DIR" ] || die "no registry rows under $ROWS_DIR - run ./run_model_solo.sh (build) first"
  export SOLO_RESULTS ROWS_DIR
}

# Arms this platform runs. MIG on a Jetson needs L4T >= R39.2 (earlier releases
# expose no MIG profiles); it is listed so the matrix has the cell, and the arm
# writes an 'unsupported' marker instead of measuring.
platform_arms(){
  case "$PLATFORM" in
    jetson)   echo "plain mps streams mig" ;;
    discrete) echo "plain mps streams mig" ;;
    *) die "unknown platform $PLATFORM" ;;
  esac
}

# row_loop: built once per results dir from the kit source (same compile line
# as the e2e driver); the binary never lives in the kit tree.
build_row_loop(){
  local bin="$1"
  [ -x "$bin" ] && [ "$bin" -nt "$ROW_LOOP_SRC" ] && return 0
  mkdir -p "$(dirname "$bin")"
  # a discrete box normally has TensorRT as a tarball: take its include/lib from
  # TENSORRT_ROOT or the trtexec on PATH, as setup.sh does
  local troot="${TENSORRT_ROOT:-}" tflags=""
  if [ -z "$troot" ]; then
    local tx; tx=$(command -v trtexec 2>/dev/null || true)
    [ -n "$tx" ] && troot=$(cd "$(dirname "$tx")/.." 2>/dev/null && pwd)
  fi
  [ -n "$troot" ] && [ -d "$troot/include" ] && tflags="-I$troot/include"
  [ -n "$troot" ] && [ -d "$troot/lib" ] && tflags="$tflags -L$troot/lib"
  g++ -O2 -std=c++17 "$ROW_LOOP_SRC" $tflags -I"/usr/include/$(gcc -dumpmachine)" -I/usr/local/cuda/include \
      -L/usr/local/cuda/lib64 -lnvinfer -lnvinfer_plugin -lcudart -ldl -lpthread -o "$bin" > "$bin.compile.log" 2>&1 \
    || { tail -15 "$bin.compile.log"; die "row_loop failed to compile - $bin.compile.log"; }
}

# Clock lock held for a whole stage-5 run (paced solos and every arm), released
# on exit: the same policy as measure_models.sh. Jetson: jetson_clocks with the
# pre-run state stored and restored; discrete: -lgc/-lmc at the device config's
# bins (never the nameplate max). Verified by verify_lock.py right after.
CLOCKS_LOCKED=0; KEEPALIVE_PID=""
start_keepalive(){ ( while true; do sudo -n true 2>/dev/null || exit 0; sleep 60; done ) & KEEPALIVE_PID=$!; }
lock_clocks(){      # lock_clocks <provenance dir> <log>
  local prov="$1" log="$2"
  if [ "$PLATFORM" = jetson ]; then
    sudo -n jetson_clocks --store "$prov/jetson_clocks_saved.conf" 2>/dev/null \
      || die "cannot store pre-run clocks (is sudo primed on this tty? run: sudo -v)"
    sudo -n jetson_clocks 2>/dev/null || die "jetson_clocks failed"
  else
    MAXGC=$(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits | head -1)
    MAXMC=$(nvidia-smi --query-gpu=clocks.max.memory --format=csv,noheader,nounits | head -1)
    local c
    c=$(python3 -c 'import json,sys;v=json.load(open(sys.argv[1])).get("sm_lock_mhz");print(v if v else "")' "$DEVICE_CFG"); [ -n "$c" ] && MAXGC="$c"
    c=$(python3 -c 'import json,sys;v=json.load(open(sys.argv[1])).get("mem_lock_mhz");print(v if v else "")' "$DEVICE_CFG"); [ -n "$c" ] && MAXMC="$c"
    sudo -n nvidia-smi -pm 1 >/dev/null 2>&1; sudo -n nvidia-smi -lgc "$MAXGC" >/dev/null 2>&1 || die "-lgc failed"
    sudo -n nvidia-smi -lmc "$MAXMC" >/dev/null 2>&1 || say "WARN: -lmc unsupported on this part; memory clock floats"
  fi
  CLOCKS_LOCKED=1; start_keepalive; say "clocks pinned"
  python3 "$KIT/common/verify_lock.py" --device "$DEVICE_CFG" --out "$prov/lock_verified.json" \
    ${MAXGC:+--requested "sm=${MAXGC},mem=${MAXMC}"} 2>&1 | tee -a "$log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || die "clock-lock verification FAILED - evidence: $prov/lock_verified.json"
}
release_clocks(){   # release_clocks <provenance dir>
  [ -n "$KEEPALIVE_PID" ] && { kill "$KEEPALIVE_PID" 2>/dev/null || true; KEEPALIVE_PID=""; }
  [ "$CLOCKS_LOCKED" = 1 ] || return 0
  say "releasing clock locks..."
  if [ "$PLATFORM" = jetson ]; then
    sudo -n jetson_clocks --restore "$1/jetson_clocks_saved.conf" 2>/dev/null \
      && say "clocks restored to pre-run state" \
      || say "WARN: clocks still pinned - restore: sudo jetson_clocks --restore $1/jetson_clocks_saved.conf"
  else
    sudo -n nvidia-smi -rgc >/dev/null 2>&1; sudo -n nvidia-smi -rmc >/dev/null 2>&1; say "clock locks released"
  fi
  CLOCKS_LOCKED=0
}

# --mix <m> / --arm <a> (repeatable), --list, --skip-solo: the whole stage-5
# argument surface. Everything else is discovered.
MIXES=(); ARMS=(); LIST=0; SKIP_SOLO=0
parse_coloc_args(){
  while [ $# -gt 0 ]; do
    case "$1" in
      --mix) [ -n "${2:-}" ] || die "--mix needs a mix name"; MIXES+=("$2"); shift 2 ;;
      --mix=*) MIXES+=("${1#--mix=}"); shift ;;
      --arm) [ -n "${2:-}" ] || die "--arm needs an arm name"; ARMS+=("$2"); shift 2 ;;
      --arm=*) ARMS+=("${1#--arm=}"); shift ;;
      --list) LIST=1; shift ;;
      --skip-solo) SKIP_SOLO=1; shift ;;
      -h|--help) sed -n '2,/^#===/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
      *) die "unknown argument '$1' (this script takes only --mix <m>, --arm <a>, --list, --skip-solo)" ;;
    esac
  done
  local a ok
  # the arm set is per platform: the device config must be known before a
  # requested arm can be judged
  [ ${#ARMS[@]} -gt 0 ] && [ -z "${PLATFORM:-}" ] && discover_device
  for a in "${ARMS[@]}"; do
    ok=0; for p in $(platform_arms); do [ "$a" = "$p" ] && ok=1; done
    [ $ok = 1 ] || die "arm '$a' is not one of this platform's arms: $(platform_arms)"
  done
}
mix_args(){ local m; for m in "${MIXES[@]}"; do echo --mix "$m"; done; }
mix_names(){   # registered mixes (or the --mix subset), validated first
  python3 "$COLOC_DIR/mixes.py" list --mix-dir "$MIX_DIR" --rows-dir "$ROWS_DIR" --results "$SOLO_RESULTS" --platform "$PLATFORM" \
    $(mix_args) 2>/dev/null | sed -n 's/^mix \([^ ]*\).*/\1/p'
}
