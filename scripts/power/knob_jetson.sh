#!/usr/bin/env bash
# Jetson operating-point knob: nvpmodel modes.
#   knob_jetson.sh apply    nvpmodel <id>            apply mode <id>; prints the mode name nvpmodel reports
#   knob_jetson.sh readback nvpmodel <id> [out.json] the applied state: mode name, GPU/EMC caps, online CPUs
#   knob_jetson.sh restore  nvpmodel <id>            same as apply (the baseline id)
# nvpmodel -m exits non-zero for some modes on JetPack 7 (it complains about a
# power-gating sysfs node the driver does not expose) yet applies the mode: the
# result is judged by the mode NAME it reports afterwards, never by the exit code.
set -uo pipefail
op="${1:?apply|readback|restore}"; knob="${2:?knob}"; val="${3:?value}"; out="${4:-}"
[ "$knob" = nvpmodel ] || { echo "FATAL: jetson knob must be nvpmodel (got $knob)" >&2; exit 1; }
mode_name(){ nvpmodel -q 2>/dev/null | sed -n 's/^NV Power Mode: *//p' | head -1; }
conf_name(){ sed -n "s/^< POWER_MODEL ID=$val NAME=\([^ >]*\) >.*/\1/p" /etc/nvpmodel.conf | head -1; }
case "$op" in
  apply|restore)
    want=$(conf_name); [ -n "$want" ] || { echo "FATAL: no POWER_MODEL ID=$val in /etc/nvpmodel.conf" >&2; exit 1; }
    sudo -n nvpmodel -m "$val" </dev/null 2>&1 | grep -v '^$' | sed 's/^/  nvpmodel: /' >&2; sleep 3
    got=$(mode_name)
    [ "$got" = "$want" ] || { echo "FATAL: nvpmodel -m $val did not take: wanted $want, device reports '$got'" >&2; exit 1; }
    echo "$got" ;;
  readback)
    python3 - "$val" "$(mode_name)" "$out" <<'PY'
import json, sys
def rd(p):
    try: return open(p).read().strip()
    except Exception: return None
mid, name, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
gpu = {k: rd(f'/sys/class/devfreq/gpu-gpc-0/{k}') for k in ('cur_freq', 'min_freq', 'max_freq', 'governor')}
emc = {k: rd(f'/sys/class/devfreq/bwmgr/{k}') for k in ('cur_freq', 'min_freq', 'max_freq')}
d = dict(knob='nvpmodel', value=mid, mode=name, gpu_devfreq=gpu, emc_devfreq=emc, cpu_online=rd('/sys/devices/system/cpu/online'),
         # the caps the mode leaves on the clocks: what a lock at this point can reach, hence the lock targets of the derived config
         cap_mhz=dict(gpu=int(gpu['max_freq']) / 1e6 if gpu.get('max_freq') else None, emc=int(emc['max_freq']) / 1e6 if emc.get('max_freq') else None))
if out: json.dump(d, open(out, 'w'), indent=1)
print(json.dumps(d))
PY
    ;;
  *) echo "FATAL: unknown op $op" >&2; exit 1 ;;
esac
