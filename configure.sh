#!/usr/bin/env bash
# Expand the ${VAR} placeholders in the shipped mix CSVs and manifests into
# machine-local copies. Run once after editing env.sh.
#
#   . ./env.sh && ./configure.sh
#
# Writes:  scripts/mixes.local/*.csv   and  scripts/manifests.local/*
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -n "${KIT_ROOT:-}" ] || { echo "source env.sh first: . ./env.sh"; exit 1; }
command -v envsubst >/dev/null || { echo "envsubst not found (apt install gettext-base)"; exit 1; }

VARS='${KIT_ROOT} ${RESULTS_ROOT} ${CONFIG_ROOT} ${FIGS_ROOT} ${MODEL_ROOT} ${ENGINE_ROOT} ${INPUTS_ROOT} ${WORK_ROOT} ${EDGELLM_ROOT} ${TRTLLM_ROOT} ${LLM_WORKSPACE} ${HANDOFF_ROOT} ${HOME}'

for d in mixes manifests; do
  src="$HERE/configs/$d"; dst="$HERE/scripts/$d.local"
  [ -d "$src" ] || continue
  mkdir -p "$dst"; n=0
  for f in "$src"/*; do
    [ -f "$f" ] || continue
    envsubst "$VARS" < "$f" > "$dst/$(basename "$f")"; n=$((n+1))
  done
  echo "  $d -> scripts/$d.local  ($n files)"
done

echo
echo "Unresolved placeholders remaining (each means an env var is empty):"
grep -rhoI '\${[A-Z_]*}' "$HERE/scripts"/*.local/* 2>/dev/null | sort -u | sed 's/^/  /' || echo "  none"
