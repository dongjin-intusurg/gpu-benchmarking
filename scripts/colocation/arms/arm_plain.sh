#!/usr/bin/env bash
# Arm: plain time-sharing - one process (one CUDA context) per frame row,
# default scheduler, side loads in their own processes.
set -uo pipefail
. "$(dirname "$0")/arm_common.sh"
run_arm plain
