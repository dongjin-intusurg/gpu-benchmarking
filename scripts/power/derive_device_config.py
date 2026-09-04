#!/usr/bin/env python3
"""Derive the device config of one operating point from the base config and the
knob read-back, so every downstream tool (preflight, verify_lock, the sampler,
drift_report) judges the point against ITS OWN caps instead of the baseline's.

  derive_device_config.py --base <cfg.json> --point <name> --readback <readback.json> --out <derived.json>

jetson:   required_power_mode := the mode name read back; lock_targets_mhz.gpu/emc
          := the devfreq max_freq caps the mode leaves (a jetson_clocks lock
          pins the clocks there, not at the baseline's 1575 / 4266)
discrete: pl  -> power_envelope_w := the enforced limit (lock targets unchanged)
          lgc -> sm_lock_mhz and lock_targets_mhz.sm := the recipe's cap
The derived file carries a `power_point` block naming what changed; stdout is one
line `derived config for <point>: <key> <base> -> <point>, ...` (grepped by the stage).
"""
import argparse
import json


def derive_jetson(config, point, readback):
    """Mode name and the devfreq caps the mode leaves; returns {key: (base, point)}."""
    if readback.get('mode') != point:
        raise SystemExit(f"point {point} but the device reports mode {readback.get('mode')!r} "
                         f"- the point name must be the mode name")
    changed = {'required_power_mode': (config.get('required_power_mode'), point)}
    config['required_power_mode'] = point
    lock_targets = dict(config.get('lock_targets_mhz') or {})
    caps = readback.get('cap_mhz') or {}
    for clock in ('gpu', 'emc'):
        if not caps.get(clock):
            continue
        cap_mhz = int(round(caps[clock]))
        changed[f'lock_targets_mhz.{clock}'] = (lock_targets.get(clock), cap_mhz)
        lock_targets[clock] = cap_mhz
    config['lock_targets_mhz'] = lock_targets
    return changed


def derive_discrete(config, point, readback):
    """Power limit (pl) or SM clock cap (lgc); returns {key: (base, point)}."""
    if readback.get('mode') and readback.get('mode') != point:
        raise SystemExit(f"point {point} but the device reads back {readback.get('mode')!r} "
                         f"- name the point after the limit / recipe it applies")
    changed = {}
    caps = readback.get('cap_mhz') or {}
    if readback.get('knob') == 'pl' and readback.get('power_limit_w'):
        changed['power_envelope_w'] = (config.get('power_envelope_w'), readback['power_limit_w'])
        config['power_envelope_w'] = readback['power_limit_w']
    elif readback.get('knob') == 'lgc' and caps.get('sm'):
        sm_mhz = int(caps['sm'])
        lock_targets = dict(config.get('lock_targets_mhz') or {})
        changed['sm_lock_mhz'] = (config.get('sm_lock_mhz'), sm_mhz)
        config['sm_lock_mhz'] = sm_mhz
        changed['lock_targets_mhz.sm'] = (lock_targets.get('sm'), sm_mhz)
        lock_targets['sm'] = sm_mhz
        config['lock_targets_mhz'] = lock_targets
    return changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--point', required=True)
    parser.add_argument('--readback', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    config = json.load(open(args.base))
    readback = json.load(open(args.readback))
    derive = derive_jetson if config.get('platform') == 'jetson' else derive_discrete
    changed = derive(config, args.point, readback)
    config['power_point'] = dict(
        point=args.point, base_config=args.base, readback=readback,
        changed={key: dict(base=base, point=point) for key, (base, point) in changed.items()})
    json.dump(config, open(args.out, 'w'), indent=1)
    summary = ', '.join(f'{key} {base} -> {point}' for key, (base, point) in changed.items())
    print(f"derived config for {args.point}: " + (summary or 'nothing changed (baseline)'))


if __name__ == '__main__':
    main()
