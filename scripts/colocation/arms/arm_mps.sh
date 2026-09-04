#!/usr/bin/env bash
# Arm: MPS - the same one-process-per-row layout under an MPS control daemon
# the runner starts for this arm (per-arm pipe dir, quit on exit, existence
# verified and recorded). A row's mps_pct becomes its
# CUDA_MPS_ACTIVE_THREAD_PERCENTAGE. Refuses to run over a daemon left by
# someone else (mps_ctl.sh stop).
set -uo pipefail
. "$(dirname "$0")/arm_common.sh"
command -v nvidia-cuda-mps-control >/dev/null || { unsupported_marker mps "nvidia-cuda-mps-control not installed"; exit 0; }
run_arm mps
