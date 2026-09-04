# Sourced by run_figures.sh. Builds on stage4_common.sh (say/die, KIT,
# RESULTS_ROOT, device discovery, ceilings discovery) and adds what the figure
# stage needs: the figure root, the newest run of each measuring stage, a
# device lookup by tag for machines without the GPU, and the stage-7 argument
# surface. Not executable on its own.
FIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "$FIG_DIR/../model_bench/stage4_common.sh"
FIGS_ROOT="${FIGS_ROOT:-$KIT/../figs}"

# The newest run by NAME (the run stamp), never by mtime: a re-judged older
# run has a newer mtime and must not shadow the latest measurement.
newest(){ ls -1d "$@" 2>/dev/null | sort | tail -1; }

# DEVICE_TAG=<tag> selects the device config by tag, so figures regenerate on
# a machine that does not carry the GPU; otherwise the config is discovered
# from the machine as every measuring stage does.
device_by_tag(){
  local d f
  if [ -z "${DEVICE_CFG:-}" ] && [ -n "${DEVICE_TAG:-}" ]; then
    for d in "${DEVICE_CONFIG_DIR:-}" "$KIT/device_configs" "$KIT/../configs/device_configs"; do
      [ -n "$d" ] && [ -d "$d" ] || continue
      for f in "$d"/*.json; do
        [ "$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("device_tag",""))' "$f" 2>/dev/null)" = "$DEVICE_TAG" ] \
          && { DEVICE_CFG="$f"; break 2; }
      done
    done
    [ -n "${DEVICE_CFG:-}" ] || die "no device config carries device_tag '$DEVICE_TAG'"
  fi
  discover_device
}

# Each measuring stage is optional: an absent run leaves its variable empty
# and the stage's figures are skipped, the report names what is missing. A
# variable pinned to the empty string means "treat the stage as not run".
discover_solo_run(){  SOLO_RESULTS="${SOLO_RESULTS-$(newest "$RESULTS_ROOT"/solo_"${DEVICE_TAG}"_*/results.json)}"; }
discover_coloc_run(){ COLOC_RUN="${COLOC_RUN-$(newest "$RESULTS_ROOT"/coloc_"${DEVICE_TAG}"_*/verdict.json | xargs -r dirname)}"; }
discover_power_run(){ POWER_RUN="${POWER_RUN-$(newest "$RESULTS_ROOT"/power_"${DEVICE_TAG}"_*/verdict.json | xargs -r dirname)}"; }

# The ceilings of record are the ones the solo run budgeted against; the
# newest ceilings run is the fallback when the solo run is absent or its
# recorded file is gone.
discover_ceilings_of_record(){
  local recorded=""
  [ "${CEILINGS_JSON+set}" = set ] && [ -z "$CEILINGS_JSON" ] && return 0
  if [ -z "${CEILINGS_JSON:-}" ] && [ -n "${SOLO_RESULTS:-}" ] && [ -f "$SOLO_RESULTS" ]; then
    recorded=$(python3 -c 'import json,sys;print(((json.load(open(sys.argv[1])).get("ceilings") or {}).get("ceilings_source") or {}).get("path") or "")' "$SOLO_RESULTS" 2>/dev/null)
    [ -n "$recorded" ] && [ -f "$recorded" ] && CEILINGS_JSON="$recorded"
  fi
  discover_ceilings
}

# The whole stage-7 argument surface: --list, --check. Inputs and the output
# directory are pinned by environment (SOLO_RESULTS, COLOC_RUN, POWER_RUN,
# CEILINGS_JSON, OUT_DIR, DEVICE_TAG).
CHECK=0
parse_figures_args(){
  while [ $# -gt 0 ]; do
    case "$1" in
      --list) LIST=1; shift ;;
      --check) CHECK=1; shift ;;
      -h|--help) sed -n '2,/^#===/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
      *) die "unknown argument '$1' (this script takes only --list, --check)" ;;
    esac
  done
}

# What the run will read: one line per input with its row / cell / point count.
describe_inputs(){
  python3 - "${CEILINGS_JSON:-}" "${SOLO_RESULTS:-}" "${COLOC_RUN:-}" "${POWER_RUN:-}" <<'PY'
import json, os, sys
ceil, solo, coloc, power = sys.argv[1:5]
def load(p):
    try: return json.load(open(p))
    except Exception: return None
def line(stage, path, what):
    print(f"  {stage:9} {path or '(none)'}" + (f"   {what}" if what else ''))
c = load(ceil) if ceil else None
line('ceilings', ceil, c and f"{len(c.get('gemm_sweep') or {})} GEMM sizes, {len(c.get('bandwidth') or {})} bandwidth buffers, sustained {'yes' if c.get('sustained') else 'no'}")
s = load(solo) if solo else None
line('solo', solo, s and f"{len(s.get('rows') or [])} rows, run_valid {(s.get('clock_integrity') or {}).get('run_valid')}")
v = load(os.path.join(coloc, 'verdict.json')) if coloc else None
line('coloc', coloc, v and f"{len(v.get('mixes') or [])} mixes x {len(v.get('arms') or [])} arms, {len(v.get('cells') or [])} cells")
p = load(os.path.join(power, 'verdict.json')) if power else None
line('power', power, p and f"{len(p.get('points') or {})} points ({', '.join(p.get('points_order') or [])}), baseline {p.get('baseline')}, verdict {p.get('verdict')}")
PY
}
