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

VARS='${KIT_ROOT} ${STAGE_KIT} ${ONNX_DIR} ${VA_ONNX_DIR} ${VFM_ONNX_DIR} ${EDGELLM_PYLIB} ${EDGELLM_PLUGIN_PATH} ${RESULTS_ROOT} ${CONFIG_ROOT} ${FIGS_ROOT} ${MODEL_ROOT} ${ENGINE_ROOT} ${INPUTS_ROOT} ${WORK_ROOT} ${EDGELLM_ROOT} ${TRTLLM_ROOT} ${LLM_WORKSPACE} ${HANDOFF_ROOT} ${HOME}'

for d in mixes manifests; do
  src="$HERE/configs/$d"; dst="$HERE/scripts/$d.local"
  [ -d "$src" ] || continue
  mkdir -p "$dst"; n=0
  for f in "$src"/*; do
    [ -f "$f" ] || continue
    envsubst "$VARS" < "$f" > "$dst/$(basename "$f")"; n=$((n+1))
  done
  # A file whose source was renamed or deleted must not linger: the model
  # registry reads every model_manifest_*.env in manifests.local, so a stale
  # copy would register a model twice (or one that no longer exists).
  stale=0
  for f in "$dst"/*; do
    [ -f "$f" ] && [ ! -f "$src/$(basename "$f")" ] && { rm -f "$f"; stale=$((stale+1)); }
  done
  echo "  $d -> scripts/$d.local  ($n files$( [ $stale = 0 ] || echo ", $stale stale removed"))"
done

# A mix is read BOTH by shell (IFS=, read ... EXTRA) and by python csv readers.
# CRLF endings put a stray \r on the last field, and CSV quotes leak literal
# quote characters into it - either one silently corrupts a trtexec flag string
# (a plugin path becomes unopenable). Catch it here rather than mid-measurement.
bad=0
for f in "$HERE"/scripts/mixes.local/*.csv; do
  [ -f "$f" ] || continue
  if grep -qU $'\r' "$f"; then echo "  ERROR: $(basename "$f") has CRLF endings - flags will carry a stray CR"; bad=1; fi
  if grep -q '"' "$f"; then echo "  ERROR: $(basename "$f") contains quotes - the shell reader takes them literally"; bad=1; fi
done
[ "$bad" = 0 ] || { echo "  fix the mix files above (LF endings, no quotes) and re-run"; exit 1; }

echo
echo "Unresolved placeholders remaining (each means an env var is empty):"
grep -rhoI '\${[A-Z_]*}' "$HERE/scripts"/*.local/* 2>/dev/null | sort -u | sed 's/^/  /' || echo "  none"
