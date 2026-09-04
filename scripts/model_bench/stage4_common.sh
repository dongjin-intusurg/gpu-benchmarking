# Sourced by the stage-4 scripts (validate_registry.sh, build_models.sh,
# measure_models.sh, run_model_solo.sh). Discovers everything a stage needs
# from the machine so the scripts themselves take no configuration arguments:
#   env.sh path contract -> KIT, KIT_ROOT, ENGINE_ROOT, RESULTS_ROOT ...
#   device config        -> DEVICE_CFG, PLATFORM (jetson|discrete), DEVICE_TAG
#   registry helpers     -> registry_export <name>  (eval-able R_* assignments)
# Not executable on its own.
set -uo pipefail
MB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIT="$(cd "$MB_DIR/.." && pwd)"
for _e in "$KIT/../env.sh" "$KIT/env.sh"; do
  # shellcheck disable=SC1090
  [ -r "$_e" ] && { . "$_e" >/dev/null 2>&1; break; }
done
unset _e
export PATH="$PATH:/usr/src/tensorrt/bin"
say(){ echo "[$(date +%H:%M:%S)] $*"; }
die(){ echo "FATAL: $*" >&2; exit 1; }

ENGINE_ROOT="${ENGINE_ROOT:-$KIT/../engines}"
RESULTS_ROOT="${RESULTS_ROOT:-$KIT/../results}"
REGISTRY="$MB_DIR/registry.py"
MANIFEST_DIR="${MANIFEST_DIR:-$KIT/manifests.local}"

discover_device(){
  DEVICE_CFG="${DEVICE_CFG:-$(python3 "$KIT/common/pick_device_config.py" 2>/dev/null)}"
  [ -n "${DEVICE_CFG:-}" ] && [ -f "$DEVICE_CFG" ] \
    || die "no device config matches this machine - fill configs/device_configs/<device>.json from the template (see RUNBOOK.md)"
  PLATFORM=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["platform"])' "$DEVICE_CFG")
  DEVICE_TAG=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("device_tag",""))' "$DEVICE_CFG")
  [ -n "$DEVICE_TAG" ] || die "device config $DEVICE_CFG has no device_tag"
  export DEVICE_CFG PLATFORM DEVICE_TAG BENCH_PLATFORM="$PLATFORM"
}

# The newest completed ceilings run for this device: the CAPACITY side of every
# budget. A per-model N is only meaningful against ceilings from the same
# machine and clock regime.
discover_ceilings(){
  if [ -z "${CEILINGS_JSON:-}" ]; then
    CEILINGS_JSON=$(ls -1dt "$KIT"/results_${DEVICE_TAG}_ceilings*/raw/results.json \
                            "$RESULTS_ROOT"/results_${DEVICE_TAG}_ceilings*/raw/results.json 2>/dev/null | head -1)
  fi
  export CEILINGS_JSON
}

registry_args(){ echo --manifest-dir "$MANIFEST_DIR" --platform "$PLATFORM" --engine-root "$ENGINE_ROOT" --device-tag "$DEVICE_TAG"; }
registry_export(){ python3 "$REGISTRY" export "$1" $(registry_args); }
registry_names(){ python3 "$REGISTRY" json $(registry_args) "$@" | python3 -c 'import json,sys; [print(m["name"]) for m in json.load(sys.stdin)["models"]]'; }

# --only <name> (repeatable) is the one filter every stage-4 script accepts.
ONLY=(); LIST=0; SKIP_BUILD=0
parse_only(){
  while [ $# -gt 0 ]; do
    case "$1" in
      --only) [ -n "${2:-}" ] || die "--only needs a model name"; ONLY+=(--only "$2"); shift 2 ;;
      --only=*) ONLY+=(--only "${1#--only=}"); shift ;;
      --list) LIST=1; shift ;;
      --skip-build) SKIP_BUILD=1; shift ;;
      -h|--help) sed -n '2,/^#===/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
      *) die "unknown argument '$1' (this script takes only --only <model>, --list, --skip-build)" ;;
    esac
  done
}

# Builds contaminate a live measurement and vice versa: refuse to start one
# while the other is running on this machine.
refuse_if_measuring(){
  local p
  p=$(pgrep -f "run_model_bench.sh|measure_models.sh|run_device_ceilings.sh|measure_ceilings" | grep -v "^$$\$" | head -3 || true)
  [ -z "$p" ] || die "a measurement is running (pids: $(echo $p)) - never build while anything measures on this GPU"
}
