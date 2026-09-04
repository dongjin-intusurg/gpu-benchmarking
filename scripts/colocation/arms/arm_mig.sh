#!/usr/bin/env bash
# Arm: MIG - hardware partitions, one frame row per instance (round-robin over
# the instances the device config's mig setup creates), --row-device per row.
# Discrete only in practice: a Jetson exposes MIG profiles from L4T R39.2 on,
# so an earlier release writes an 'unsupported' marker and the matrix keeps
off.|off.
set -uo pipefail
. "$(dirname "$0")/arm_common.sh"
if [ -f /etc/nv_tegra_release ]; then
  REL=$(sed -n 's/^# R\([0-9]*\).*REVISION: \([0-9.]*\).*/\1.\2/p' /etc/nv_tegra_release | head -1)
  python3 -c 'import sys; a=[int(x) for x in sys.argv[1].split(".")[:2]]; sys.exit(0 if a>=[39,2] else 1)' "${REL:-0.0}" \
    || { unsupported_marker mig "MIG needs L4T >= R39.2 (this board: R${REL:-?})"; exit 0; }
fi
nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader 2>/dev/null | grep -qi 'enabled\|disabled' \
  || { unsupported_marker mig "this GPU reports no MIG mode"; exit 0; }
MIG_SETUP="${MIG_SETUP:-$ARM_DIR/../mig_setup.sh}"
[ -x "$MIG_SETUP" ] || { unsupported_marker mig "MIG setup script not found: $MIG_SETUP"; exit 0; }
"$MIG_SETUP" on > "$ARM_OUT/mig_setup.log" 2>&1 || { tail -5 "$ARM_OUT/mig_setup.log"; unsupported_marker mig "MIG enable failed (see mig_setup.log)"; exit 0; }
trap '"$MIG_SETUP" off >> "$ARM_OUT/mig_setup.log" 2>&1' EXIT
mapfile -t UUIDS < <(nvidia-smi -L | sed -n 's/.*MIG.*(UUID: \(MIG-[^)]*\)).*/\1/p')
[ "${#UUIDS[@]}" -ge 1 ] || { unsupported_marker mig "no MIG instances after setup"; exit 0; }
mapfile -t FRAMES < <(python3 -c 'import json,sys; [print(r["name"]) for r in json.load(open(sys.argv[1]))["rows"] if r["role"]=="frame"]' "$RESOLVED")
DEV=(); i=0
for r in "${FRAMES[@]}"; do DEV+=(--row-device "$r=${UUIDS[$((i % ${#UUIDS[@]}))]}"); i=$((i + 1)); done
run_arm mig "${DEV[@]}"
