#!/usr/bin/env python3
"""Headless + exclusivity gate: refuse to measure on a box that is not solo.

Every run number carries the regime "headless, solo-exclusive". an earlier unlocked run's
numbers were taken with the desktop up (~392 MiB / 6 GPU clients on Jetson);
this gate exists so that cannot happen silently again. It runs before every
lock/measure stage and writes machine-readable evidence (preflight/v1) into
provenance.

Pre-declared gates (refusal = exit 2, driver scripts hard-stop):
  R1 display stack up: systemd display-manager unit active, OR any compositor
     process present (exact comm match against Xorg, Xwayland, gnome-shell,
     kwin_wayland, weston). A compositor holds a GPU context and schedules
     work at vsync - it contaminates both latency tails and idle bandwidth.
  R2 foreign GPU clients: any compute (C) or graphics (G) process reported by
     nvidia-smi. Solo-exclusive means OUR measurement process is the only
     context; each offender is listed pid/name/MiB.
  R3 zombie VRAM (discrete only): memory.used > 100 MiB with ZERO listed
     clients means a crashed process leaked a context; its resident footprint
     skews the VRAM budget and can hold clocks up. 100 MiB clears normal
     driver/ECC reserved overhead.
  R4 power mode (jetson only, --require-maxn): nvpmodel mode not MAXN, or
     unverifiable while the flag is set. Without the flag this is WARN-only:
     ceilings runs require MAXN (they define capacity), model timing runs
     merely record the mode.

Pre-declared warnings (recorded, never fatal):
  W1 nvpmodel not MAXN / unverifiable (without --require-maxn).
  W2 stale DISPLAY env var with no compositor running - harmless to the GPU
     but a sign the shell came from a desktop session; GUI-launching tools
     may hang.
  W3 idle power: 10 s baseline mean > 1.5 x config idle_power_w_expected
     (IDLE_POWER_WARN_FACTOR). Power is the corroborating witness on Tegra,
     where an empty client list is NOT evidence of idleness (nvidia-smi on
     Tegra reports no per-process memory - recorded as a quirk in the JSON).
     Expected value null in config => SKIP, noted.

Escape hatch: --allow-desktop downgrades every refusal to WARN and stamps
smoke_only:true in the JSON - downstream stages must then mark their numbers
invalid. For rehearsing the pipeline only, never for run numbers.

Idle baseline: 10 s (--baseline-seconds) sampled through clock_sampler's
make_sampler(), i.e. the SAME by-name source resolution the drift record
uses - the baseline and the under-load samples are comparable by
construction. Skipped (with a SKIP note) when refusing: a non-idle box has
no idle baseline.

Exit codes: 0 = proceed (PASS or WARN), 2 = refused. The JSON is written in
every case. stdlib only.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# clock_sampler lives beside this file; import it for the shared source
# resolution helpers so baseline and drift samples come from identical sources
sys.path.insert(0, str(Path(__file__).resolve().parent))
import clock_sampler  # noqa: E402

COMPOSITOR_NAMES = ['Xorg', 'Xwayland', 'gnome-shell', 'kwin_wayland', 'weston']
ZOMBIE_VRAM_MIB = 100
IDLE_POWER_WARN_FACTOR = 1.5
BASELINE_SECONDS_DEFAULT = 10

REMEDIATION_HEADLESS = """Remediation (go headless, then re-run this preflight):
  sudo systemctl isolate multi-user.target    # stops the display manager and every compositor
  # ... run the run ...
  sudo systemctl isolate graphical.target     # restores the desktop when the run is done"""

REMEDIATION_CLIENTS = """Remediation (clear foreign GPU clients):
  kill or wait out the listed processes, then re-run this preflight
  (compositor-owned clients disappear with: sudo systemctl isolate multi-user.target)"""

REMEDIATION_ZOMBIE = """Remediation (leaked GPU context with no live owner):
  sudo fuser -v /dev/nvidia*     # identify any hidden holder
  reboot if nothing owns it      # a leaked context survives until driver reset"""

REMEDIATION_MAXN = """Remediation (power mode):
  sudo nvpmodel -p --verbose     # list modes, find the MAXN index
  sudo nvpmodel -m <MAXN index>  # ceilings define capacity - run them at the required mode"""


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_cmd(argv, timeout=15):
    """(returncode, stdout) - rc None if the binary is missing or timed out."""
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout
    except (OSError, subprocess.TimeoutExpired):
        return None, ''


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

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
        # -x = exact comm match: avoids false hits on e.g. an editor whose
        # command line happens to contain a compositor name
        rc, out = run_cmd(['pgrep', '-x', name])
        if rc == 0 and out.strip():
            found.append({'name': name, 'pids': [int(p) for p in out.split()]})
    return found


def query_compute_apps():
    """List of compute clients, or None if the query is unsupported here."""
    rc, out = run_cmd(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                       '--format=csv,noheader,nounits'])
    if rc != 0:
        return None
    apps = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 3 and parts[0].isdigit():
            mib = int(parts[2]) if parts[2].isdigit() else None
            apps.append({'pid': int(parts[0]), 'name': parts[1], 'mib': mib, 'type': 'C'})
    return apps


def query_process_table():
    """All rows of the plain nvidia-smi process table (covers G-type, which
    has no csv query)."""
    rc, out = run_cmd(['nvidia-smi'])
    if rc != 0:
        return None
    procs = []
    # row shape: |  0  N/A  N/A   1234   G   /usr/lib/xorg/Xorg   392MiB |
    row_re = re.compile(r'^\|\s+\d+\s+\S+\s+\S+\s+(\d+)\s+([A-Z+]+)\s+(.+?)\s+(\d+)MiB\s*\|')
    for line in out.splitlines():
        m = row_re.match(line)
        if m:
            procs.append({'pid': int(m.group(1)), 'type': m.group(2),
                          'name': m.group(3).strip(), 'mib': int(m.group(4))})
    return procs


def query_memory_used_mib():
    rc, out = run_cmd(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'])
    if rc != 0 or not out.strip():
        return None
    first = out.strip().splitlines()[0].strip()
    return int(first) if first.isdigit() else None


def query_power_limit_w():
    # enforced limit vs the card's default. A stale `nvidia-smi -pl <lower>` left
    # over from a power sweep silently lowers every ceiling, so the discrete
    # analogue of the jetson power-mode gate is: the enforced limit must equal
    # the default (which on these cards is also the maximum).
    rc, out = run_cmd(['nvidia-smi',
                       '--query-gpu=power.limit,power.default_limit,power.max_limit',
                       '--format=csv,noheader,nounits'])
    if rc != 0 or not out.strip():
        return None
    parts = [x.strip() for x in out.strip().splitlines()[0].split(',')]
    def f(x):
        try: return float(x)
        except Exception: return None
    return {'limit_w': f(parts[0]), 'default_w': f(parts[1]) if len(parts) > 1 else None,
            'max_w': f(parts[2]) if len(parts) > 2 else None}


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


def measure_idle_baseline(cfg, seconds, notes):
    """Sample power/clocks for the window via clock_sampler's sources."""
    sampler = clock_sampler.make_sampler(cfg)
    notes.extend(sampler.notes)
    header = sampler.header  # includes t_s at index 0; rows exclude it
    power_col = 'module_w' if 'module_w' in header else 'power_w'
    clock_col = 'gpu_mhz' if 'gpu_mhz' in header else 'sm_mhz'
    p_idx = header.index(power_col) - 1
    c_idx = header.index(clock_col) - 1

    powers, clocks_mhz, n = [], [], 0
    t0 = time.monotonic()
    k = 0
    while time.monotonic() - t0 < seconds:
        row = sampler.row()
        n += 1
        try:
            if row[p_idx]:
                powers.append(float(row[p_idx]))
        except (ValueError, IndexError):
            pass
        try:
            if row[c_idx]:
                clocks_mhz.append(float(row[c_idx]))
        except (ValueError, IndexError):
            pass
        k += 1
        delay = (t0 + k * sampler.period_s) - time.monotonic()
        if delay > 0:
            time.sleep(min(delay, sampler.period_s))
    sampler.close()

    baseline = {
        'seconds': seconds,
        'samples': n,
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Headless + exclusivity preflight gate')
    ap.add_argument('--device', required=True, help='device config JSON')
    ap.add_argument('--out', required=True, help='output preflight.json')
    ap.add_argument('--baseline-seconds', type=float, default=BASELINE_SECONDS_DEFAULT,
                    help='idle baseline sampling window (default %(default)s)')
    ap.add_argument('--require-maxn', action='store_true',
                    help='refuse (not just warn) when nvpmodel is not the required mode - set for '
                         'ceilings runs. The required mode is the device config\'s '
                         'required_power_mode (default MAXN): a run that measures at a lower '
                         'mode declares it there, and the ceilings it produces are that mode\'s '
                         'capacity')
    ap.add_argument('--allow-desktop', action='store_true',
                    help='smoke tests only: downgrade refusals to warnings, stamp smoke_only')
    args = ap.parse_args()

    try:
        cfg = clock_sampler.load_device_config(args.device)
    except (OSError, json.JSONDecodeError) as e:
        print(f'FATAL: cannot read device config {args.device}: {e}', file=sys.stderr)
        sys.exit(1)

    platform = cfg.get('platform', 'discrete')
    is_jetson = platform == 'jetson'
    refusals, warnings, notes, remediations = [], [], [], []
    checks = {}

    say(f"preflight: {cfg.get('device_id', '?')} ({platform})"
        + (' [allow-desktop: smoke only]' if args.allow_desktop else ''))

    # --- R1: display stack ---------------------------------------------------
    dm = check_display_manager()
    compositors = check_compositors()
    checks['display_manager'] = dm
    checks['compositors'] = compositors
    if 'note' in dm:
        notes.append(dm['note'])
    if dm['active']:
        refusals.append('display-manager systemd unit is active - the box is not headless')
    if compositors:
        listing = ', '.join(f"{c['name']}(pid {','.join(map(str, c['pids']))})" for c in compositors)
        refusals.append(f'compositor process(es) running: {listing}')
    if (dm['active'] or compositors):
        remediations.append(REMEDIATION_HEADLESS)

    # --- W2: stale DISPLAY ---------------------------------------------------
    display_env = os.environ.get('DISPLAY', '')
    checks['display_env'] = {'DISPLAY': display_env or None,
                             'stale': bool(display_env) and not compositors}
    if display_env and not compositors:
        warnings.append(f'DISPLAY={display_env} set but no compositor running - '
                        'stale desktop-session environment')

    # --- R2: foreign GPU clients --------------------------------------------
    compute_apps = query_compute_apps()
    table = query_process_table()
    graphics = [p for p in (table or []) if 'G' in p['type']]
    if compute_apps is None:
        compute = [p for p in (table or []) if 'C' in p['type']]
        notes.append('SKIP: --query-compute-apps unsupported here - compute clients taken '
                     'from the plain nvidia-smi process table')
    else:
        compute = compute_apps
    clients = {'compute': compute, 'graphics': graphics}
    if is_jetson:
        # Tegra quirk, recorded: nvidia-smi reports no per-process memory
        # there, so an empty list proves nothing - the idle-power baseline
        # below is the corroborating witness
        clients['quirk'] = ('tegra nvidia-smi reports no per-process memory - an empty '
                            'client list is NOT evidence of idleness; corroborated by the '
                            'idle-power baseline')
        notes.append(clients['quirk'])
    checks['gpu_clients'] = clients
    foreign = compute + graphics
    if foreign:
        for p in foreign:
            mib = p.get('mib')
            say(f"  foreign client: pid {p['pid']} {p.get('type', '?')} "
                f"{p.get('name', '?')} {mib if mib is not None else '?'} MiB")
        refusals.append(f'{len(foreign)} foreign GPU client(s) hold a context '
                        '(listed above and in checks.gpu_clients)')
        remediations.append(REMEDIATION_CLIENTS)

    # --- R3: zombie VRAM (discrete only) ------------------------------------
    if not is_jetson:
        used = query_memory_used_mib()
        checks['zombie_vram'] = {'memory_used_mib': used, 'threshold_mib': ZOMBIE_VRAM_MIB,
                                 'clients_listed': len(foreign)}
        if used is None:
            notes.append('SKIP: memory.used query failed - zombie-VRAM check unavailable')
        elif used > ZOMBIE_VRAM_MIB and not foreign:
            refusals.append(f'{used} MiB VRAM resident with zero listed clients - '
                            'leaked (zombie) GPU context')
            remediations.append(REMEDIATION_ZOMBIE)

    # --- R4/W1: power mode (jetson only) ------------------------------------
    if is_jetson:
        pm = check_nvpmodel()
        checks['nvpmodel'] = pm
        if 'note' in pm:
            notes.append(pm['note'])
        want = (cfg.get('required_power_mode') or 'MAXN').strip()
        pm['required_mode'] = want
        have = (pm['mode'] or '')
        pm['at_required_mode'] = (want.upper() in have.upper()) if pm['mode'] else None
        if pm['at_required_mode'] is True:
            pass
        elif args.require_maxn:
            reason = (f"nvpmodel mode is '{pm['mode']}', not the required '{want}'"
                      if pm['mode'] else 'nvpmodel mode is unverifiable')
            refusals.append(reason + ' (ceilings define capacity, so they must run at the '
                                     'mode the run measures at)')
            remediations.append(REMEDIATION_MAXN.replace('MAXN index', f'{want} index')
                                                .replace('find the MAXN', f'find the {want}'))
        else:
            warnings.append(f"nvpmodel mode is '{pm['mode']}' (required mode is '{want}') - "
                            'recorded; enforced only for ceilings runs')

    # --- R5/W2: power limit (discrete only) ---------------------------------
    if not is_jetson:
        pl = query_power_limit_w()
        checks['power_limit'] = pl
        if not pl or pl.get('limit_w') is None or pl.get('default_w') is None:
            notes.append('SKIP: power-limit query failed - the discrete power-cap check is unavailable')
        else:
            at_default = abs(pl['limit_w'] - pl['default_w']) <= 1.0
            pl['at_default'] = at_default
            if at_default:
                pass
            elif args.require_maxn:
                refusals.append(f"power limit is {pl['limit_w']:.0f} W, not the default "
                                f"{pl['default_w']:.0f} W (ceilings define capacity, so they must "
                                'run at the default power limit - a lower cap is a deliberate sweep)')
                remediations.append('Remediation (power limit):\n'
                                    f"  sudo nvidia-smi -pl {pl['default_w']:.0f}   "
                                    '# restore the default (= maximum) power limit')
            else:
                warnings.append(f"power limit is {pl['limit_w']:.0f} W (default {pl['default_w']:.0f} W) - "
                                'recorded; enforced only for ceilings runs')

    # --- allow-desktop downgrade ---------------------------------------------
    smoke_only = False
    if refusals and args.allow_desktop:
        smoke_only = True
        warnings.extend(f'(smoke-only downgrade) {r}' for r in refusals)
        refusals = []
        say('WARN: refusal conditions present but --allow-desktop set - '
            'proceeding as SMOKE RUN; downstream numbers are invalid')

    refused = bool(refusals)

    # --- idle baseline (skipped on refusal: a non-idle box has no baseline) --
    idle_baseline = None
    idle_power_mean = None
    if refused:
        notes.append('SKIP: idle baseline not measured - preflight refused before baseline')
    else:
        say(f'sampling {args.baseline_seconds:g} s idle baseline...')
        idle_baseline = measure_idle_baseline(cfg, args.baseline_seconds, notes)
        expected = cfg.get('idle_power_w_expected')
        idle_baseline['expected_power_w'] = expected
        if idle_baseline['power_w'] is not None:
            idle_power_mean = idle_baseline['power_w']['mean']
            if expected is None:
                notes.append('SKIP: idle_power_w_expected is null in the device config - '
                             'power-vs-expected check not applicable')
            elif idle_power_mean > IDLE_POWER_WARN_FACTOR * expected:
                warnings.append(
                    f'idle power mean {idle_power_mean} W exceeds '
                    f'{IDLE_POWER_WARN_FACTOR} x expected {expected} W - '
                    'something is exercising the GPU or the platform is not settled')

    verdict = 'REFUSED' if refused else ('WARN' if warnings else 'PASS')

    result = {
        'schema': 'preflight/v1',
        'run_date': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'device_tag': cfg.get('device_tag'),
        'device_id': cfg.get('device_id'),
        'platform': platform,
        'device_config': os.path.abspath(args.device),
        'require_maxn': args.require_maxn,
        'allow_desktop': args.allow_desktop,
        'verdict': verdict,
        'smoke_only': smoke_only,
        'idle_power_w_mean': idle_power_mean,
        'refusals': refusals,
        'warnings': warnings,
        'notes': notes,
        'checks': checks,
        'idle_baseline': idle_baseline,
    }
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=1)

    for w in warnings:
        say(f'WARN: {w}')

    if refused:
        print('FATAL: preflight REFUSED - the box is not in the measurement regime:',
              file=sys.stderr, flush=True)
        for r in refusals:
            print(f'  - {r}', file=sys.stderr, flush=True)
        seen = set()
        for rem in remediations:
            if rem not in seen:
                seen.add(rem)
                print(rem, file=sys.stderr, flush=True)
        say(f'preflight verdict: REFUSED -> {out_path}')
        sys.exit(2)

    say(f'preflight verdict: {verdict}'
        + (' (smoke_only - numbers invalid downstream)' if smoke_only else '')
        + f' -> {out_path}')
    sys.exit(0)


if __name__ == '__main__':
    main()
