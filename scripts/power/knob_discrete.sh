#!/usr/bin/env bash
# Discrete-card operating-point knob: an enforced power limit, or an SM clock-cap
# recipe at the default limit.
#   knob_discrete.sh apply    pl  <W>                nvidia-smi -pl; prints the limit read back
#   knob_discrete.sh apply    lgc <MHz>              records the recipe (the lock step realizes it: -lgc <MHz>)
#   knob_discrete.sh readback <knob> <value> [out.json]   power limit, default/min/max limits, current clocks
#   knob_discrete.sh restore  pl  <W>                back to the baseline limit and -rgc
# A -pl value below power.min_limit cannot be enforced: apply exits 3 (the point
# is skipped and recorded), never rounds.
set -uo pipefail
op="${1:?apply|readback|restore}"; knob="${2:?knob}"; val="${3:?value}"; out="${4:-}"
q(){ nvidia-smi --query-gpu="$1" --format=csv,noheader,nounits | head -1 | cut -d. -f1; }
case "$op:$knob" in
  apply:pl|restore:pl)
    w="$val"; [ "$w" = default ] && w=$(q power.default_limit)   # 'default' still accepted: the card's own default limit
    mn=$(q power.min_limit); mx=$(q power.max_limit)
    [ "$w" -ge "$mn" ] && [ "$w" -le "$mx" ] || { echo "SKIP: -pl $w W outside the card's range $mn-$mx W" >&2; exit 3; }
    sudo -n nvidia-smi -pm 1 >/dev/null 2>&1; sudo -n nvidia-smi -pl "$w" >/dev/null 2>&1 || { echo "FATAL: nvidia-smi -pl $w failed (sudo primed?)" >&2; exit 1; }
    [ "$op" = restore ] && sudo -n nvidia-smi -rgc >/dev/null 2>&1
    sleep 2; got=$(q power.limit)
    [ "$got" = "$w" ] || { echo "FATAL: power limit read back $got W after -pl $w" >&2; exit 1; }
    echo "${got}W" ;;
  apply:lgc)
    # the recipe itself is applied by lock_clocks (-lgc from the derived config's sm_lock_mhz); here only sanity
    mxc=$(q clocks.max.graphics); [ "$val" -le "$mxc" ] || { echo "SKIP: -lgc $val above clocks.max.graphics $mxc" >&2; exit 3; }
    echo "lgc$val" ;;
  restore:lgc) sudo -n nvidia-smi -rgc >/dev/null 2>&1; echo "unlocked" ;;
  readback:*)
    python3 - "$knob" "$val" "$out" "$(q power.limit)" "$(q power.default_limit)" "$(q power.min_limit)" "$(q power.max_limit)" "$(q clocks.sm)" "$(q clocks.mem)" "$(q clocks.max.graphics)" "$(q clocks.max.memory)" <<'PY'
import json, sys
knob, val, out = sys.argv[1], sys.argv[2], sys.argv[3]
pl, pld, plmin, plmax, sm, mem, smmax, memmax = [int(x) if x.strip().lstrip('-').isdigit() else None for x in sys.argv[4:12]]
d = dict(knob=knob, value=val, mode=(f'{pl}W' if knob == 'pl' else f'lgc{val}'), power_limit_w=pl, power_default_limit_w=pld, power_min_limit_w=plmin,
         power_max_limit_w=plmax, clocks_mhz=dict(sm=sm, mem=mem), clocks_max_mhz=dict(sm=smmax, mem=memmax),
         cap_mhz=dict(sm=(int(val) if knob == 'lgc' else None), mem=None))
if out: json.dump(d, open(out, 'w'), indent=1)
print(json.dumps(d))
PY
    ;;
  *) echo "FATAL: unknown op/knob $op $knob" >&2; exit 1 ;;
esac
