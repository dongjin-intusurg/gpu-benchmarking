#!/usr/bin/env bash
# MPS control daemon helper. The mps arm starts and quits its own daemon; this
# is for the preflight check and for cleaning up after an interrupted arm.
#   mps_ctl.sh status | stop
set -uo pipefail
case "${1:-status}" in
  status)
    if pgrep -fa '^nvidia-cuda-mps-control' ; then echo "MPS control daemon running (pipe: $(tr '\0' '\n' < /proc/$(pgrep -f '^nvidia-cuda-mps-control' | head -1)/environ 2>/dev/null | sed -n 's/^CUDA_MPS_PIPE_DIRECTORY=//p'))"; exit 1
    else echo "no MPS control daemon"; fi ;;
  stop)
    for p in $(pgrep -f '^nvidia-cuda-mps-control'); do
      d=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | sed -n 's/^CUDA_MPS_PIPE_DIRECTORY=//p')
      CUDA_MPS_PIPE_DIRECTORY="${d:-/tmp/nvidia-mps}" timeout 5 bash -c 'echo quit | nvidia-cuda-mps-control' 2>/dev/null
    done
    sleep 1; pkill -f '^nvidia-cuda-mps-control' 2>/dev/null; pkill -f '^nvidia-cuda-mps-server' 2>/dev/null
    rm -rf /tmp/nvmps_* 2>/dev/null; echo "MPS stopped" ;;
  *) echo "usage: $0 status|stop"; exit 2 ;;
esac
