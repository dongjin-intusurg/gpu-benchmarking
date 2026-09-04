# Sourced by the stage-6 scripts (run_power.sh, measure_points.sh). Builds on
# stage5_common.sh (device config, solo results, mixes, clock lock/release) and
# adds what the power stage needs: the point table of this platform, the knob
# dispatcher, the gating marker and the stage-6 argument surface. Not
# executable on its own.
POWER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "$POWER_DIR/../colocation/stage5_common.sh"
POINTS_DIR="${POINTS_DIR:-$KIT/../configs/power}"
[ -d "$POINTS_DIR" ] || POINTS_DIR="$KIT/configs/power"

# The point table of this platform: configs/power/points_<platform>.csv
# (point,knob,value,default,gated,note). Comment lines and blanks are skipped.
points_file(){ echo "$POINTS_DIR/points_$PLATFORM.csv"; }
# point_rows [name...]: the selected rows as "point|knob|value|default|gated|note",
# ungated points FIRST so a gated one never precedes an ungated one in a run
point_rows(){
  python3 - "$(points_file)" "$@" <<'PY'
import csv, sys
path, want = sys.argv[1], sys.argv[2:]
rows = [r for r in csv.DictReader(l for l in open(path) if l.strip() and not l.lstrip().startswith('#'))]
need = {'point', 'knob', 'value', 'default', 'gated'}
missing = need - set(rows[0].keys() if rows else [])
if missing: sys.exit(f"FATAL: {path}: missing columns {sorted(missing)}")
names = [r['point'] for r in rows]
if len(set(names)) != len(names): sys.exit(f"FATAL: {path}: duplicate point names")
if want:
    bad = [w for w in want if w not in names]
    if bad: sys.exit(f"FATAL: unknown point(s) {bad}; this platform's points: {names}")
    sel = [r for r in rows if r['point'] in want]
else:
    sel = [r for r in rows if r['default'].strip() == '1']
sel.sort(key=lambda r: int(r['gated'].strip() or 0))
for r in sel:
    print('|'.join([r['point'], r['knob'], r['value'], r['default'], r['gated'], (r.get('note') or '').strip()]))
PY
}
baseline_point(){   # the first default row of the table: the point every other one is judged against
  python3 - "$(points_file)" <<'PY'
import csv, sys
rows = [r for r in csv.DictReader(l for l in open(sys.argv[1]) if l.strip() and not l.lstrip().startswith('#'))]
d = [r for r in rows if r['default'].strip() == '1']
print(d[0]['point'] if d else '')
PY
}

# knob <apply|readback|restore> <knob> <value> [out.json]: the platform's knob script
knob(){ "$POWER_DIR/knob_$PLATFORM.sh" "$@"; }

# A gated point (Jetson TPC power-gating mask) is applied only at boot: once one
# has run, every ungated point of this boot is invalid. The marker is keyed by
# boot id so a reboot clears it by construction.
GATE_MARK="$RESULTS_ROOT/.power_gated_$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || echo unknown)"
gate_check(){   # gate_check <point> <gated>: die when an ungated point follows a gated one in this boot
  local point="$1" gated="$2"
  [ "$gated" = 1 ] && return 0
  [ -f "$GATE_MARK" ] || return 0
  die "point $point is ungated but a gated point ($(cat "$GATE_MARK")) ran in this boot - its power-gating mask stays applied until reboot; reboot, then rerun"
}
gate_mark(){ [ "$2" = 1 ] && echo "$1 $(date -Is)" > "$GATE_MARK"; return 0; }

# --point <p> / --mix <m> (repeatable), --list, --skip-tops, --skip-mixes: the
# whole stage-6 argument surface. Everything else is discovered.
POINTS=(); SKIP_TOPS=0; SKIP_MIXES=0
parse_power_args(){
  while [ $# -gt 0 ]; do
    case "$1" in
      --point) [ -n "${2:-}" ] || die "--point needs a point name"; POINTS+=("$2"); shift 2 ;;
      --point=*) POINTS+=("${1#--point=}"); shift ;;
      --mix) [ -n "${2:-}" ] || die "--mix needs a mix name"; MIXES+=("$2"); shift 2 ;;
      --mix=*) MIXES+=("${1#--mix=}"); shift ;;
      --list) LIST=1; shift ;;
      --skip-tops) SKIP_TOPS=1; shift ;;
      --skip-mixes) SKIP_MIXES=1; shift ;;
      -h|--help) sed -n '2,/^#===/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
      *) die "unknown argument '$1' (this script takes only --point <p>, --mix <m>, --list, --skip-tops, --skip-mixes)" ;;
    esac
  done
}
point_args(){ local p; for p in "${POINTS[@]}"; do echo --point "$p"; done; }
