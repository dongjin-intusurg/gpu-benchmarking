#!/usr/bin/env bash
#=============================================================================
# Stage 4 - per-model solo measurement: register -> build -> measure -> N.
#
#   ./run_model_solo.sh                every registered model
#   ./run_model_solo.sh --list         what is registered and what it would build
#   ./run_model_solo.sh --only <m>     one model (repeatable)
#   ./run_model_solo.sh --skip-build   artifacts already built: measure only
#
# No other arguments. Device config, ceilings run, registry and workspaces are
# discovered from the machine (env.sh path contract).
#
#   0. validate   model_bench/validate_registry.sh   no GPU; every error at once
#   1. build      model_bench/build_models.sh        UNLOCKED (builds are untimed):
#                 trt / adopt / edgellm (jetson) / trtllm (discrete) per the
#                 registration; validates every artifact; writes the row set
#   2. measure    model_bench/measure_models.sh      LOCKED + verified, released
#                 on exit: engine rows (trtexec p99 / nsys / NCU / VRAM), e2e
#                 driver rows, generative rows (TTFT, decode tok/s, control
#                 step) -> budgets -> U_max, C, L, N = min(L,C), cause, Score
#=============================================================================
set -uo pipefail
. "$(dirname "$0")/model_bench/stage4_common.sh"
parse_only "$@"
MB="$KIT/model_bench"

if [ "$LIST" = 1 ]; then
  "$MB/validate_registry.sh" "${ONLY[@]}" || exit 1
  echo; "$MB/build_models.sh" --list "${ONLY[@]}"
  exit $?
fi

say "=========== stage 4.0: validate the registry ==========="
"$MB/validate_registry.sh" "${ONLY[@]}" || die "registry invalid - fix the manifests above, then rerun"

if [ "$SKIP_BUILD" = 1 ]; then
  say "=========== stage 4.1: build skipped (--skip-build) ==========="
else
  say "=========== stage 4.1: build + validate artifacts (unlocked) ==========="
  "$MB/build_models.sh" "${ONLY[@]}" || die "build failed - see the build logs named above; measurement not started"
fi

say "=========== stage 4.2: measure (locked) -> N ==========="
"$MB/measure_models.sh" "${ONLY[@]}"
