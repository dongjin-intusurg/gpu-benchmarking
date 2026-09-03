#!/usr/bin/env bash
#=============================================================================
# One-shot machine setup: path contract, engine discovery, symlinks, C++ builds,
# and a resolve check over every mix the suite uses.
#
#   ./setup.sh                      discover under $HOME
#   ./setup.sh --model-root ~/repos --engine-root ~/engine_root
#   ./setup.sh --search ~/repos --search /data/models
#
# Idempotent: safe to re-run after adding a model tree or rebuilding an engine.
# Exits non-zero if anything the suite needs is unresolved.
#=============================================================================
set -uo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
G="\033[32m"; Y="\033[33m"; R="\033[31m"; Z="\033[0m"
ok(){ printf "  ${G}ok${Z}    %s\n" "$1"; }
warn(){ printf "  ${Y}warn${Z}  %s\n" "$1"; WARN=$((WARN+1)); }
bad(){ printf "  ${R}FAIL${Z}  %s\n" "$1"; FAIL=$((FAIL+1)); }
WARN=0; FAIL=0
SEARCH=()

while [ $# -gt 0 ]; do
  case "$1" in
    --model-root)  MODEL_ROOT="$2"; shift 2;;
    --engine-root) ENGINE_ROOT="$2"; shift 2;;
    --search)      SEARCH+=("$2"); shift 2;;
    -h|--help)     sed -n '2,12p' "$0"; exit 0;;
    *) echo "unknown option: $1"; exit 2;;
  esac
done
[ ${#SEARCH[@]} -gt 0 ] || SEARCH=("$HOME")

# One switch drives every platform-specific branch below, so a Jetson never runs
# a discrete branch and a discrete card never runs a Jetson one. Same rule as
# env.sh; BENCH_PLATFORM=jetson|discrete forces it.
if [ -n "${BENCH_PLATFORM:-}" ];  then PLATFORM="$BENCH_PLATFORM"
elif [ -f /etc/nv_tegra_release ]; then PLATFORM=jetson
else                                   PLATFORM=discrete; fi
export PLATFORM

echo "=== 1. machine state ==="
printf "  platform        %s\n" "$PLATFORM"
if [ "$PLATFORM" = jetson ]; then
  MODE=$(nvpmodel -q 2>/dev/null | sed -n 's/.*: //p' | head -1)
  [ -n "$MODE" ] && printf "  power mode      %s\n" "$MODE"
else
  MODE=""
  CAP=$(nvidia-smi --query-gpu=power.limit,power.default_limit --format=csv,noheader 2>/dev/null | head -1)
  case "$CAP" in ""|*N/A*) ;; *) printf "  power cap       %s (enforced / default)\n" "$CAP";; esac
fi
CPUS=$(cat /sys/devices/system/cpu/online 2>/dev/null)
printf "  online cpus     %s\n" "${CPUS:-?}"
DM=$(systemctl is-active display-manager 2>/dev/null || true)
if [ "$DM" = "active" ]; then
  warn "a desktop session is up — certified numbers need 'sudo systemctl isolate multi-user.target' from ssh"
else ok "headless"; fi
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
[ "$BUSY" -eq 0 ] && ok "no other GPU clients" || bad "$BUSY other process(es) hold the GPU"
if [ "$PLATFORM" = discrete ] && [ "$BUSY" -eq 0 ]; then
  # a discrete card can hold memory with no client attached - a stale context
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  # integrated parts report [N/A] here, so only compare when it is really a number
  case "$USED" in ''|*[!0-9]*) ;; *) [ "$USED" -gt 100 ] && warn "$USED MiB of VRAM in use with zero clients - a stale context; reset before measuring";; esac
fi
if [ -n "$MODE" ] && [ "$MODE" != "MAXN" ]; then
  warn "mode is $MODE, not MAXN — recorded targets are per power mode; TPC gating applies at boot, so switching needs a reboot"
fi

echo
echo "=== 2. model root ==="
if [ -z "${MODEL_ROOT:-}" ]; then
  for s in "${SEARCH[@]}"; do
    for m in ${SETUP_MARKERS:-model-assets}; do
      hit=$(find "$s" -maxdepth 3 -type d -name "$m" 2>/dev/null | head -1)
      [ -n "$hit" ] && { MODEL_ROOT=$(dirname "$(dirname "$hit")"); break; }
    done
    [ -n "${MODEL_ROOT:-}" ] && break
  done
fi
if [ -n "${MODEL_ROOT:-}" ] && [ -d "$MODEL_ROOT" ]; then ok "MODEL_ROOT=$MODEL_ROOT"
else bad "MODEL_ROOT not found — pass --model-root <dir containing the model repos>"; MODEL_ROOT="${MODEL_ROOT:-$HOME/models}"; fi
export MODEL_ROOT

echo
echo "=== 3. engine discovery + symlinks ==="
export ENGINE_ROOT="${ENGINE_ROOT:-$KIT/engines}"
mkdir -p "$ENGINE_ROOT"
n=0
while IFS= read -r d; do
  case "$(basename "$d")" in engines_*) ;; *) continue;; esac
  [ -L "$ENGINE_ROOT/$(basename "$d")" ] && [ "$(readlink -f "$ENGINE_ROOT/$(basename "$d")")" = "$(readlink -f "$d")" ] && continue
  case "$(readlink -f "$d")" in "$(readlink -f "$ENGINE_ROOT")"*) continue;; esac
  ln -sfn "$d" "$ENGINE_ROOT/$(basename "$d")"; n=$((n+1))
done < <(for s in "${SEARCH[@]}"; do find "$s" -maxdepth 5 -type d -name 'engines_*' 2>/dev/null; done | sort -u)
ok "ENGINE_ROOT=$ENGINE_ROOT ($n new link(s))"
for e in "$ENGINE_ROOT"/*; do
  [ -e "$e" ] || continue
  printf "        %-34s %s engine(s)\n" "$(basename "$e")" "$(ls "$e"/*.engine 2>/dev/null | wc -l)"
done

echo
echo "=== 4. path contract ==="
# shellcheck disable=SC1091
. "$KIT/env.sh" >/dev/null 2>&1
if command -v envsubst >/dev/null; then
  out=$("$KIT/configure.sh" 2>&1)
  if echo "$out" | grep -q 'none'; then ok "all placeholders resolved"
  else bad "unresolved placeholders:"; echo "$out" | tail -5 | sed 's/^/        /'; fi
else bad "envsubst missing — apt install gettext-base"; fi

echo
echo "=== 5. C++ harnesses ==="
CUDA_INC=/usr/local/cuda/include; CUDA_LIB=/usr/local/cuda/lib64
# JetPack installs the TensorRT headers under the multiarch include dir and the
# libraries on the default linker path. A discrete box is normally a tarball, so
# derive the root from the trtexec that is actually on PATH and add its include
# and lib dirs explicitly - the ones the builds will really use.
TRT_ROOT="${TENSORRT_ROOT:-}"
if [ -z "$TRT_ROOT" ]; then
  _tx=$(command -v trtexec 2>/dev/null || true)
  [ -z "$_tx" ] && [ "$PLATFORM" = jetson ] && [ -x /usr/src/tensorrt/bin/trtexec ] && _tx=/usr/src/tensorrt/bin/trtexec
  [ -n "$_tx" ] && TRT_ROOT=$(cd "$(dirname "$_tx")/.." 2>/dev/null && pwd)
fi
TRT_INC="${TRT_INC:-/usr/include/$(gcc -dumpmachine 2>/dev/null)}"
TRT_FLAGS=""
[ -n "$TRT_ROOT" ] && [ -d "$TRT_ROOT/include" ] && TRT_FLAGS="-I$TRT_ROOT/include"
[ -n "$TRT_ROOT" ] && [ -d "$TRT_ROOT/lib" ]     && TRT_FLAGS="$TRT_FLAGS -L$TRT_ROOT/lib"
build(){ # dir src out extra-libs
  local d="$1" src="$2" out="$3"; shift 3
  [ -f "$d/$src" ] || { warn "$src not present — skipped"; return; }
  if [ -x "$d/$out" ] && [ "$d/$out" -nt "$d/$src" ]; then ok "$out already built"; return; fi
  ( cd "$d" && g++ -O2 -std=c++17 "$src" $TRT_FLAGS -I"$TRT_INC" -I"$CUDA_INC" -L"$CUDA_LIB" \
      -lnvinfer -lnvinfer_plugin -lcudart -ldl -lpthread "$@" -o "$out" ) 2>/tmp/bld.$$ \
    && ok "$out built" || { bad "$out build failed: $(tail -1 /tmp/bld.$$)"; }
  rm -f /tmp/bld.$$
}
build "$KIT/scripts/model_bench/cpp" row_loop.cpp row_loop
# the streaming ASR harness, wherever a stage keeps it
AE=$(find "$KIT/scripts" -name 'asr_e2e.cpp' | head -1)
[ -n "$AE" ] && build "$(dirname "$AE")" asr_e2e.cpp asr_e2e
if command -v nvcc >/dev/null && [ -f "$KIT/scripts/device_ceilings/peak_issue_probe.cu" ]; then
  ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
  [ -n "$ARCH" ] || { [ "$PLATFORM" = jetson ] && ARCH=110 || ARCH=120; }
  ( cd "$KIT/scripts/device_ceilings" && nvcc -O3 -gencode arch=compute_$ARCH,code=sm_$ARCH \
      peak_issue_probe.cu -o peak_issue_probe ) 2>/dev/null \
    && ok "peak_issue_probe built (sm_$ARCH)" || warn "peak_issue_probe build skipped"
fi

echo
echo "=== 6. do the mixes resolve? ==="
# Which mixes belong to THIS device. A mix filename may carry a device tag; a row
# in another device's mix is not this machine's problem. Keying this off the
# platform rather than a hardcoded prefix is what keeps the count meaningful on
# both boxes - the reverse hardcoding silently resolved the wrong device's rows.
MIX_TAGS_ALL="${MIX_TAGS_ALL:-jetson discrete}"
if [ -z "${MIX_TAG:-}" ]; then
  [ "$PLATFORM" = jetson ] && MIX_TAG=jetson || MIX_TAG=discrete
fi
python3 - "$KIT" "$MIX_TAG" "$MIX_TAGS_ALL" <<'PYEOF'
import csv, os, sys, glob
KIT = sys.argv[1]
MIX_TAG = sys.argv[2]
TAGS = [t for t in sys.argv[3].split() if t]
d = os.path.join(KIT, "scripts", "mixes.local")
TEMPLATE = ("/path/to/", "/models/")          # shipped examples, not real rows
res = {"ok": 0, "missing": [], "template": 0, "other_device": 0, "generated": 0}
for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
    base = os.path.basename(f)
    # tagged for a device that is not this one -> not this machine's rows
    tags = [t for t in TAGS if base.startswith("mix_%s_" % t) or base == "mix_%s.csv" % t]
    other = bool(tags) and MIX_TAG not in tags
    try: rows = list(csv.DictReader(open(f)))
    except Exception: continue
    for r in rows:
        paths = []
        e = (r.get("onnx") or r.get("engine") or "").strip()
        if e: paths.append((r.get("name",""), e))
        ex = r.get("extra") or ""
        if "staticPlugins=" in ex:
            paths.append((r.get("name","")+" [plugin]", ex.split("staticPlugins=",1)[1].split()[0]))
        for name, path in paths:
            if any(path.startswith(t) for t in TEMPLATE): res["template"] += 1; continue
            if other: res["other_device"] += 1; continue
            if os.path.exists(path): res["ok"] += 1
            elif "/work/" in path: res["generated"] += 1
            else: res["missing"].append((base, name, path))
tot = res["ok"] + len(res["missing"])
print(f"  {res['ok']}/{tot} paths this device needs resolve")
if tot == 0:
    # a vacuous pass is worse than a failure: say which of the two cases it is
    if res["template"] or res["other_device"]:
        print("        nothing to check yet - only template / other-device rows are present;"
              " add your own mix next to the shipped ones")
    else:
        print("        NO ROWS CHECKED - scripts/mixes.local is empty; run ./configure.sh")
        sys.exit(1)
print(f"        skipped: {res['template']} template rows, {res['other_device']} other-device rows, "
      f"{res['generated']} built later under work/")
seen = set()
for base, name, path in res["missing"]:
    if path in seen: continue
    seen.add(path); print(f"        MISSING  {name or '?':26s} {path}")
sys.exit(1 if res["missing"] else 0)
PYEOF
[ $? -eq 0 ] && ok "every mix row resolves" || warn "some rows unresolved — build or link the engines above, then re-run ./setup.sh"

echo
echo "=== summary ==="
printf "  MODEL_ROOT   %s\n  ENGINE_ROOT  %s\n" "$MODEL_ROOT" "$ENGINE_ROOT"
cat > "$KIT/.setup_env" <<EOF
# written by setup.sh — source this in later shells:  . ./.setup_env
export MODEL_ROOT="$MODEL_ROOT"
export ENGINE_ROOT="$ENGINE_ROOT"
export ROW_LOOP="$KIT/scripts/model_bench/cpp/row_loop"
. "$KIT/env.sh"
EOF
ok "wrote .setup_env — later shells only need:  . ./.setup_env"
if [ $FAIL -gt 0 ]; then printf "\n  ${R}%d blocking problem(s)${Z}, %d warning(s)\n" $FAIL $WARN; exit 1; fi
printf "\n  ${G}ready${Z} — %d warning(s). Next: run the device ceilings\n" $WARN
