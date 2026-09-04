#!/usr/bin/env bash
#=============================================================================
# Stage 4, step 0 - validate the model registry for THIS device.
#
#   ./validate_registry.sh [--only <model>]...
#
# No GPU, no clocks. Reads every configured model_manifest_*.env, resolves the
# defaults, and reports EVERY error at once (missing sources, an unknown
# precision, a builder that is invalid on this platform - trtllm on a Jetson,
# edgellm on a discrete card - a fallback that could never fire). Exit 1 on
# any error; the build and measure steps refuse to start until it passes.
#=============================================================================
. "$(dirname "$0")/stage4_common.sh"
parse_only "$@"
discover_device
[ -d "$MANIFEST_DIR" ] || die "no configured manifests at $MANIFEST_DIR - run:  . ./env.sh && ./configure.sh"
say "device: $(basename "$DEVICE_CFG")  platform=$PLATFORM  tag=$DEVICE_TAG"
python3 "$REGISTRY" validate $(registry_args) "${ONLY[@]}"
