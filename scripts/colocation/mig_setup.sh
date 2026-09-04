#!/usr/bin/env bash
# MIG on a discrete card that supports it: two equal instances. Enabling MIG
# needs an idle GPU (no clients, desktop down) and may need a GPU reset.
# Usage: ./mig_setup.sh on | off | list    (arm_mig.sh calls on/off around a cell)
set -uo pipefail
case "${1:-list}" in
  on)  sudo nvidia-smi -mig 1 || { echo "MIG enable refused - kill GPU clients / reboot headless and retry"; exit 1; }
       sudo nvidia-smi mig -lgip
       PROF="${MIG_PROFILE:-$(sudo nvidia-smi mig -lgip | awk '/MIG/ && /gb/{print $2; exit}')}"
       echo "creating two instances with profile $PROF (MIG_PROFILE=<id> to pick the half-card profile explicitly)"
       sudo nvidia-smi mig -cgi "$PROF,$PROF" -C || { echo "instance creation failed - check the profiles above"; exit 1; }
       nvidia-smi -L ;;
  off) sudo nvidia-smi mig -dci; sudo nvidia-smi mig -dgi; sudo nvidia-smi -mig 0; nvidia-smi -L ;;
  *)   nvidia-smi -L; nvidia-smi --query-gpu=mig.mode.current --format=csv ;;
esac
