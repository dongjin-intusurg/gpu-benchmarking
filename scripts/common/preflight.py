#!/usr/bin/env python3
"""Headless + exclusivity gate: refuse to measure on a box that is not solo.

Runs before every lock/measure stage and writes preflight/v1 evidence (--out) in every case.
Exit 0 = proceed (verdict PASS or WARN), 2 = REFUSED (refusals and remediation on stderr), 1 = bad
config. Prints "preflight verdict: <verdict> -> <path>" last. --allow-desktop downgrades every
refusal to WARN and stamps smoke_only:true - rehearsal only, downstream numbers are then invalid.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# clock_sampler lives beside this file; its make_sampler() gives the idle baseline the SAME
# by-name source resolution the under-load drift record uses, so the two are comparable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import clock_sampler  # noqa: E402

# Pre-declared gates (refusal) and warnings (recorded, never fatal):
#   R1 a display manager unit or compositor process: a compositor holds a GPU context and
#      schedules work at vsync, contaminating latency tails and idle bandwidth.
#   R2 any compute/graphics client nvidia-smi reports: solo-exclusive means ours is the only context.
#   R3 (discrete) memory.used > ZOMBIE_VRAM_MIB with zero clients: a leaked context skews the VRAM
#      budget and can hold clocks up; 100 MiB clears normal driver/ECC reserved overhead.
#   R4 (jetson, --require-maxn) nvpmodel not at the config's required_power_mode, or unverifiable.
#   R5 (discrete, --require-maxn) enforced power limit off the default: a stale `nvidia-smi -pl`
#      from a power sweep silently lowers every ceiling.
#   W1 R4/R5 without --require-maxn: ceilings define capacity, model timing runs merely record.
#   W2 stale DISPLAY with no compositor: harmless to the GPU, but GUI-launching tools may hang.
#   W3 idle-power baseline mean > IDLE_POWER_WARN_FACTOR x config idle_power_w_expected. Power is
#      the corroborating witness on Tegra, where nvidia-smi lists no per-process memory and an
#      empty client list proves nothing.
COMPOSITOR_NAMES = ['Xorg', 'Xwayland', 'gnome-shell', 'kwin_wayland', 'weston']
ZOMBIE_VRAM_MIB = 100
IDLE_POWER_WARN_FACTOR = 1.5
BASELINE_SECONDS_DEFAULT = 10
POWER_LIMIT_TOLERANCE_W = 1.0

REMEDIATION_HEADLESS = """Remediation (go headless, then re-run this preflight):
  sudo systemctl isolate multi-user.target    # stops the display manager and every compositor
  # ... run the stages ...
  sudo systemctl isolate graphical.target     # restores the desktop when the stages are done"""

REMEDIATION_CLIENTS = """Remediation (clear foreign GPU clients):
  kill or wait out the listed processes, then re-run this preflight
  (compositor-owned clients disappear with: sudo systemctl isolate multi-user.target)"""

REMEDIATION_ZOMBIE = """Remediation (leaked GPU context with no live owner):
  sudo fuser -v /dev/nvidia*     # identify any hidden holder
  reboot if nothing owns it      # a leaked context survives until driver reset"""

REMEDIATION_MAXN = """Remediation (power mode):
  sudo nvpmodel -p --verbose     # list modes, find the MAXN index
  sudo nvpmodel -m <MAXN index>  # ceilings define capacity - run them at the required mode"""

TEGRA_CLIENT_QUIRK = ('tegra nvidia-smi reports no per-process memory - an empty '
                      'client list is NOT evidence of idleness; corroborated by the '
                      'idle-power baseline')

# row shape: |  0  N/A  N/A   1234   G   /usr/lib/xorg/Xorg   392MiB |
PROCESS_TABLE_ROW = re.compile(r'^\|\s+\d+\s+\S+\s+\S+\s+(\d+)\s+([A-Z+]+)\s+(.+?)\s+(\d+)MiB\s*\|')


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_cmd(argv, timeout=15):
    """(returncode, stdout) - rc None if the binary is missing or timed out."""
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout
    except (OSError, subprocess.TimeoutExpired):
        return None, ''


def parse_float(text):
    try:
        return float(text)
    except Exception:
        return None


class Findings:
    """Everything the gates accumulate; becomes the refusals/warnings/notes/checks of the JSON."""

    def __init__(self):
        self.refusals = []
        self.warnings = []
        self.notes = []
        self.remediations = []
        self.checks = {}


def check_display_manager():
    rc, out = run_cmd(['systemctl', 'is-active', 'display-manager'])
    if rc is None:
        return {'unit': 'display-manager', 'state': 'unknown',
                'active': False, 'note': 'SKIP: systemctl unavailable'}
    state = out.strip() or 'unknown'
    return {'unit': 'display-manager', 'state': state, 'active': state == 'active'}


def check_compositors():
    found = []
    for name in COMPOSITOR_NAMES:
        # -x = exact comm match: an editor whose command line mentions a compositor is not one
        rc, out = run_cmd(['pgrep', '-x', name])
        if rc == 0 and out.strip():
            found.append({'name': name, 'pids': [int(pid) for pid in out.split()]})
    return found


def query_compute_apps():
    """List of compute clients, or None if the query is unsupported here."""
    rc, out = run_cmd(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                       '--format=csv,noheader,nounits'])
    if rc != 0:
        return None
    apps = []
    for line in out.strip().splitlines():
        parts = [part.strip() for part in line.split(',')]
        if len(parts) >= 3 and parts[0].isdigit():
            mib = int(parts[2]) if parts[2].isdigit() else None
            apps.append({'pid': int(parts[0]), 'name': parts[1], 'mib': mib, 'type': 'C'})
    return apps


def query_process_table():
    """All rows of the plain nvidia-smi process table (covers G-type, which has no csv query)."""
    rc, out = run_cmd(['nvidia-smi'])
    if rc != 0:
        return None
    processes = []
    for line in out.splitlines():
        match = PROCESS_TABLE_ROW.match(line)
        if match:
            processes.append({'pid': int(match.group(1)), 'type': match.group(2),
                              'name': match.group(3).strip(), 'mib': int(match.group(4))})
    return processes


def query_memory_used_mib():
    rc, out = run_cmd(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'])
    if rc != 0 or not out.strip():
        return None
    first = out.strip().splitlines()[0].strip()
    return int(first) if first.isdigit() else None


def query_power_limit_w():
    rc, out = run_cmd(['nvidia-smi',
                       '--query-gpu=power.limit,power.default_limit,power.max_limit',
                       '--format=csv,noheader,nounits'])
    if rc != 0 or not out.strip():
        return None
    parts = [part.strip() for part in out.strip().splitlines()[0].split(',')]
    return {'limit_w': parse_float(parts[0]),
            'default_w': parse_float(parts[1]) if len(parts) > 1 else None,
            'max_w': parse_float(parts[2]) if len(parts) > 2 else None}


def check_nvpmodel():
    for argv in (['nvpmodel', '-q'], ['sudo', '-n', 'nvpmodel', '-q']):
        rc, out = run_cmd(argv)
        if rc is None and argv[0] == 'nvpmodel':
            return {'available': False, 'mode': None, 'is_maxn': None,
                    'note': 'SKIP: nvpmodel binary not found'}
        if rc == 0 and out.strip():
            mode = None
            for line in out.splitlines():
                if 'NV Power Mode' in line:
                    mode = line.split(':', 1)[-1].strip()
                    break
            if mode is None:
                mode = out.strip().splitlines()[0]
            return {'available': True, 'mode': mode, 'is_maxn': 'MAXN' in mode.upper()}
    return {'available': True, 'mode': None, 'is_maxn': None,
            'note': 'SKIP: nvpmodel query failed (may need interactive sudo)'}


def measure_idle_baseline(config, seconds, notes):
    """Sample power/clocks for the window via clock_sampler's sources."""
    sampler = clock_sampler.make_sampler(config)
    notes.extend(sampler.notes)
    header = sampler.header  # includes t_s at index 0; rows exclude it
    power_col = 'module_w' if 'module_w' in header else 'power_w'
    clock_col = 'gpu_mhz' if 'gpu_mhz' in header else 'sm_mhz'
    power_idx = header.index(power_col) - 1
    clock_idx = header.index(clock_col) - 1

    powers, clocks_mhz, samples = [], [], 0
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        row = sampler.row()
        samples += 1
        for idx, values in ((power_idx, powers), (clock_idx, clocks_mhz)):
            try:
                if row[idx]:
                    values.append(float(row[idx]))
            except (ValueError, IndexError):
                pass
        delay = (t0 + samples * sampler.period_s) - time.monotonic()
        if delay > 0:
            time.sleep(min(delay, sampler.period_s))
    sampler.close()

    baseline = {
        'seconds': seconds,
        'samples': samples,
        'backend': sampler.backend,
        'power_field': power_col,
        'power_w': None,
        'clock_mhz_mean': round(sum(clocks_mhz) / len(clocks_mhz), 1) if clocks_mhz else None,
    }
    if powers:
        baseline['power_w'] = {
            'mean': round(sum(powers) / len(powers), 2),
            'min': round(min(powers), 2),
            'max': round(max(powers), 2),
        }
    else:
        notes.append(f'SKIP: no readable {power_col} samples - idle power baseline unavailable')
    return baseline


def gate_display_stack(findings):
    """R1 + W2; returns the compositor list."""
    display_manager = check_display_manager()
    compositors = check_compositors()
    findings.checks['display_manager'] = display_manager
    findings.checks['compositors'] = compositors
    if 'note' in display_manager:
        findings.notes.append(display_manager['note'])
    if display_manager['active']:
        findings.refusals.append('display-manager systemd unit is active - the box is not headless')
    if compositors:
        listing = ', '.join(f"{c['name']}(pid {','.join(map(str, c['pids']))})" for c in compositors)
        findings.refusals.append(f'compositor process(es) running: {listing}')
    if display_manager['active'] or compositors:
        findings.remediations.append(REMEDIATION_HEADLESS)

    display_env = os.environ.get('DISPLAY', '')
    findings.checks['display_env'] = {'DISPLAY': display_env or None,
                                      'stale': bool(display_env) and not compositors}
    if display_env and not compositors:
        findings.warnings.append(f'DISPLAY={display_env} set but no compositor running - '
                                 'stale desktop-session environment')
    return compositors


def gate_foreign_clients(findings, is_jetson):
    """R2; returns the foreign client list."""
    compute_apps = query_compute_apps()
    table = query_process_table()
    graphics = [proc for proc in (table or []) if 'G' in proc['type']]
    if compute_apps is None:
        compute = [proc for proc in (table or []) if 'C' in proc['type']]
        findings.notes.append('SKIP: --query-compute-apps unsupported here - compute clients taken '
                              'from the plain nvidia-smi process table')
    else:
        compute = compute_apps
    clients = {'compute': compute, 'graphics': graphics}
    if is_jetson:
        clients['quirk'] = TEGRA_CLIENT_QUIRK
        findings.notes.append(TEGRA_CLIENT_QUIRK)
    findings.checks['gpu_clients'] = clients
    foreign = compute + graphics
    if foreign:
        for proc in foreign:
            mib = proc.get('mib')
            say(f"  foreign client: pid {proc['pid']} {proc.get('type', '?')} "
                f"{proc.get('name', '?')} {mib if mib is not None else '?'} MiB")
        findings.refusals.append(f'{len(foreign)} foreign GPU client(s) hold a context '
                                 '(listed above and in checks.gpu_clients)')
        findings.remediations.append(REMEDIATION_CLIENTS)
    return foreign


def gate_zombie_vram(findings, foreign):
    """R3 (discrete only)."""
    used = query_memory_used_mib()
    findings.checks['zombie_vram'] = {'memory_used_mib': used, 'threshold_mib': ZOMBIE_VRAM_MIB,
                                      'clients_listed': len(foreign)}
    if used is None:
        findings.notes.append('SKIP: memory.used query failed - zombie-VRAM check unavailable')
    elif used > ZOMBIE_VRAM_MIB and not foreign:
        findings.refusals.append(f'{used} MiB VRAM resident with zero listed clients - '
                                 'leaked (zombie) GPU context')
        findings.remediations.append(REMEDIATION_ZOMBIE)


def gate_power_mode(findings, config, require):
    """R4 / W1 (jetson only)."""
    power_mode = check_nvpmodel()
    findings.checks['nvpmodel'] = power_mode
    if 'note' in power_mode:
        findings.notes.append(power_mode['note'])
    want = (config.get('required_power_mode') or 'MAXN').strip()
    power_mode['required_mode'] = want
    have = power_mode['mode'] or ''
    power_mode['at_required_mode'] = (want.upper() in have.upper()) if power_mode['mode'] else None
    if power_mode['at_required_mode'] is True:
        return
    if require:
        reason = (f"nvpmodel mode is '{power_mode['mode']}', not the required '{want}'"
                  if power_mode['mode'] else 'nvpmodel mode is unverifiable')
        findings.refusals.append(reason + ' (ceilings define capacity, so they must run at the '
                                          'mode the run measures at)')
        findings.remediations.append(REMEDIATION_MAXN.replace('MAXN index', f'{want} index')
                                                     .replace('find the MAXN', f'find the {want}'))
        return
    findings.warnings.append(f"nvpmodel mode is '{power_mode['mode']}' (required mode is '{want}') - "
                             'recorded; enforced only for ceilings runs')


def gate_power_limit(findings, require):
    """R5 / W1 (discrete only): the enforced limit must equal the default (= maximum on these cards)."""
    power_limit = query_power_limit_w()
    findings.checks['power_limit'] = power_limit
    if not power_limit or power_limit.get('limit_w') is None or power_limit.get('default_w') is None:
        findings.notes.append('SKIP: power-limit query failed - the discrete power-cap check is unavailable')
        return
    limit_w, default_w = power_limit['limit_w'], power_limit['default_w']
    at_default = abs(limit_w - default_w) <= POWER_LIMIT_TOLERANCE_W
    power_limit['at_default'] = at_default
    if at_default:
        return
    if require:
        findings.refusals.append(f"power limit is {limit_w:.0f} W, not the default "
                                 f"{default_w:.0f} W (ceilings define capacity, so they must "
                                 'run at the default power limit - a lower cap is a deliberate sweep)')
        findings.remediations.append('Remediation (power limit):\n'
                                     f"  sudo nvidia-smi -pl {default_w:.0f}   "
                                     '# restore the default (= maximum) power limit')
        return
    findings.warnings.append(f"power limit is {limit_w:.0f} W (default {default_w:.0f} W) - "
                             'recorded; enforced only for ceilings runs')


def idle_baseline_with_check(findings, config, seconds):
    """W3; returns (baseline block, idle power mean)."""
    say(f'sampling {seconds:g} s idle baseline...')
    baseline = measure_idle_baseline(config, seconds, findings.notes)
    expected = config.get('idle_power_w_expected')
    baseline['expected_power_w'] = expected
    if baseline['power_w'] is None:
        return baseline, None
    power_mean = baseline['power_w']['mean']
    if expected is None:
        findings.notes.append('SKIP: idle_power_w_expected is null in the device config - '
                              'power-vs-expected check not applicable')
    elif power_mean > IDLE_POWER_WARN_FACTOR * expected:
        findings.warnings.append(
            f'idle power mean {power_mean} W exceeds '
            f'{IDLE_POWER_WARN_FACTOR} x expected {expected} W - '
            'something is exercising the GPU or the platform is not settled')
    return baseline, power_mean


def print_refusal(findings, out_path):
    print('FATAL: preflight REFUSED - the box is not in the measurement regime:',
          file=sys.stderr, flush=True)
    for refusal in findings.refusals:
        print(f'  - {refusal}', file=sys.stderr, flush=True)
    seen = set()
    for remediation in findings.remediations:
        if remediation not in seen:
            seen.add(remediation)
            print(remediation, file=sys.stderr, flush=True)
    say(f'preflight verdict: REFUSED -> {out_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Headless + exclusivity preflight gate')
    parser.add_argument('--device', required=True, help='device config JSON')
    parser.add_argument('--out', required=True, help='output preflight.json')
    parser.add_argument('--baseline-seconds', type=float, default=BASELINE_SECONDS_DEFAULT,
                        help='idle baseline sampling window (default %(default)s)')
    parser.add_argument('--require-maxn', action='store_true',
                        help='refuse (not just warn) when nvpmodel is not the required mode - set for '
                             'ceilings runs. The required mode is the device config\'s '
                             'required_power_mode (default MAXN): a run that measures at a lower '
                             'mode declares it there, and the ceilings it produces are that mode\'s '
                             'capacity')
    parser.add_argument('--allow-desktop', action='store_true',
                        help='smoke tests only: downgrade refusals to warnings, stamp smoke_only')
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        config = clock_sampler.load_device_config(args.device)
    except (OSError, json.JSONDecodeError) as error:
        print(f'FATAL: cannot read device config {args.device}: {error}', file=sys.stderr)
        sys.exit(1)

    platform = config.get('platform', 'discrete')
    is_jetson = platform == 'jetson'
    findings = Findings()

    say(f"preflight: {config.get('device_id', '?')} ({platform})"
        + (' [allow-desktop: smoke only]' if args.allow_desktop else ''))

    gate_display_stack(findings)
    foreign = gate_foreign_clients(findings, is_jetson)
    if is_jetson:
        gate_power_mode(findings, config, args.require_maxn)
    else:
        gate_zombie_vram(findings, foreign)
        gate_power_limit(findings, args.require_maxn)

    smoke_only = False
    if findings.refusals and args.allow_desktop:
        smoke_only = True
        findings.warnings.extend(f'(smoke-only downgrade) {refusal}' for refusal in findings.refusals)
        findings.refusals = []
        say('WARN: refusal conditions present but --allow-desktop set - '
            'proceeding as SMOKE RUN; downstream numbers are invalid')
    refused = bool(findings.refusals)

    # a non-idle box has no idle baseline
    idle_baseline, idle_power_mean = None, None
    if refused:
        findings.notes.append('SKIP: idle baseline not measured - preflight refused before baseline')
    else:
        idle_baseline, idle_power_mean = idle_baseline_with_check(findings, config, args.baseline_seconds)

    verdict = 'REFUSED' if refused else ('WARN' if findings.warnings else 'PASS')
    result = {
        'schema': 'preflight/v1',
        'run_date': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'device_tag': config.get('device_tag'),
        'device_id': config.get('device_id'),
        'platform': platform,
        'device_config': os.path.abspath(args.device),
        'require_maxn': args.require_maxn,
        'allow_desktop': args.allow_desktop,
        'verdict': verdict,
        'smoke_only': smoke_only,
        'idle_power_w_mean': idle_power_mean,
        'refusals': findings.refusals,
        'warnings': findings.warnings,
        'notes': findings.notes,
        'checks': findings.checks,
        'idle_baseline': idle_baseline,
    }
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as handle:
        json.dump(result, handle, indent=1)

    for warning in findings.warnings:
        say(f'WARN: {warning}')

    if refused:
        print_refusal(findings, out_path)
        sys.exit(2)

    say(f'preflight verdict: {verdict}'
        + (' (smoke_only - numbers invalid downstream)' if smoke_only else '')
        + f' -> {out_path}')
    sys.exit(0)


if __name__ == '__main__':
    main()
