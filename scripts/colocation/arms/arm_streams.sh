#!/usr/bin/env bash
# Arm: streams - one process, one context, one CUDA stream per frame row at
# the row's prio (row_loop --multi); the device's stream priority range is
# recorded in the cell. Side loads stay separate processes.
set -uo pipefail
. "$(dirname "$0")/arm_common.sh"
run_arm streams
