# Sourced by the arm scripts. An arm is one way of co-scheduling the mix's
# frame rows on the device; every arm takes the same two arguments,
#   arm_<name>.sh <resolved mix.json> <out_dir>
# reads the environment measure_mixes.sh sets (ROW_LOOP binary, RUN_SECONDS,
# PACED_SOLO dir), and leaves <out_dir>/concurrent_mix.json in the one shape
# run_concurrent_mix.py writes (the arm is a field inside it, never the path).
ARM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$ARM_DIR/../run_concurrent_mix.py"
RESOLVED="${1:?resolved mix json}"; ARM_OUT="${2:?out dir}"; mkdir -p "$ARM_OUT"
: "${ROW_LOOP:?ROW_LOOP (the compiled row_loop binary) must be set}"
RUN_SECONDS="${RUN_SECONDS:-90}"
run_arm(){   # run_arm <mode> [extra runner args]
  local mode="$1"; shift
  python3 "$RUNNER" --resolved "$RESOLVED" --mode "$mode" --out "$ARM_OUT" --row-loop "$ROW_LOOP" \
      --seconds "$RUN_SECONDS" ${PACED_SOLO:+--paced-solo "$PACED_SOLO"} ${ROW_GRID_S:+--grid-s "$ROW_GRID_S"} "$@"
}
unsupported_marker(){   # unsupported_marker <mode> <why>
  python3 - "$RESOLVED" "$ARM_OUT" "$1" "$2" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
json.dump({'mix': m['mix'], 'mode': sys.argv[3], 'arm': {'name': sys.argv[3]}, 'resolved_mix': sys.argv[1], 'rows': [],
           'status': 'unsupported', 'why': sys.argv[4]}, open(f'{sys.argv[2]}/concurrent_mix.json', 'w'), indent=1)
print(f"{m['mix']}/{sys.argv[3]}: unsupported - {sys.argv[4]}")
PY
}
