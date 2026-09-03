#!/usr/bin/env python3
"""Per-PID clock/power/thermal sampler, device-aware via the run device config.

Runs alongside a measurement process (--pid) or for a fixed window (--duration)
and writes one CSV row per tick. The drift verdict (drift_report.py) and the
per-model clock_integrity blocks are computed from these samples, so the header
is a contract:

  jetson:   t_s,gpu_mhz,emc_mhz,module_w,vdd_gpu_w,tj_c,oc_event_count
  discrete: t_s,sm_mhz,mem_mhz,power_w,temp_c,throttle_reasons_hex

Source resolution (pre-declared policy):
  * hwmon chips are resolved BY NAME: glob /sys/class/hwmon/hwmon*/name and
    match the config's hwmon_name. Indices (hwmon5, hwmon6) are boot-order
    artifacts and must never be hardcoded.
  * INA3221 rail channels are resolved by their in*_label content (e.g.
    VDD_GPU); the config's curr/volt attrs are the fallback when no label
    file matches.
  * thermal zones are resolved by /sys/.../thermal_zone*/type content.
  * devfreq dirs come straight from the config (they are stable by-name paths).
  * an unreadable source produces a BLANK field for that tick plus one note in
    the sidecar meta - never a crash, never a fabricated value. Downstream
    readers treat blank as "no data".

Sampling-overhead argument (why this sampler cannot perturb the measurement):
  * jetson: every column is an unprivileged sysfs read - about 50 us of CPU
    per file and ZERO GPU work. At the default 0.1 s period that is well under
    0.1% of one CPU core on a 14-core module; the GPU never sees the sampler.
  * discrete, pynvml backend: NVML calls on a persistent device handle are
    in-process ioctls, comparable cost; default period 0.2 s.
  * discrete, nvidia-smi fallback: each sample spawns a process (~tens of ms
    of CPU). That is why the fallback period is pinned to 1.0 s
    (sampler_period_s_subprocess_fallback) - at 1 Hz the spawn cost stays
    below a few percent of one CPU core and still zero GPU work.

Other pre-declared policies:
  * default period comes from the device config (sampler_period_s); an
    explicit --period always wins, with a WARN if it drives the subprocess
    fallback faster than 0.5 s.
  * ticks run on a fixed schedule (t0 + n*period) so slow reads do not
    accumulate cadence drift.
  * rows are flushed as written and SIGTERM/SIGINT finalize the sidecar meta,
    so a sampler killed by its parent still leaves a complete, parseable CSV.
  * exit code is 0 in all normal paths (the sampler is evidence-gathering,
    not a gate); only unusable arguments or an unreadable config are fatal.

CLI:
  clock_sampler.py --device <cfg.json> --out <samples.csv>
                   (--pid <PID> | --duration <seconds>)
                   [--period <seconds>] [--phase <label>]

A sidecar <samples.csv>.meta.json records phase label, backend, period, row
count and every source-resolution note (Tegra quirks included).

stdlib only; pynvml is an optional import with a subprocess fallback.
"""

import argparse
import csv
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def die(msg):
    print(f"FATAL: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def load_device_config(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Source resolution helpers - shared with preflight.py (which imports this
# module so its idle baseline uses the exact same sources as the drift record).
# ---------------------------------------------------------------------------

def resolve_hwmon_dir(hwmon_name):
    """Directory of the hwmon chip whose name file matches, else None."""
    for name_file in sorted(glob.glob('/sys/class/hwmon/hwmon*/name')):
        try:
            with open(name_file) as f:
                if f.read().strip() == hwmon_name:
                    return os.path.dirname(name_file)
        except OSError:
            continue
    return None


def resolve_thermal_zone_temp(zone_type):
    """temp file of the thermal zone whose type matches, else None."""
    for type_file in sorted(glob.glob('/sys/devices/virtual/thermal/thermal_zone*/type')):
        try:
            with open(type_file) as f:
                if f.read().strip() == zone_type:
                    return os.path.join(os.path.dirname(type_file), 'temp')
        except OSError:
            continue
    return None


def read_num(path):
    """Numeric content of a sysfs file, or None if unreadable/non-numeric."""
    try:
        with open(path) as f:
            return float(f.read().strip())
    except (OSError, ValueError):
        return None


def _resolve_ina3221_channel(hwmon_dir, want_label, cfg_curr_attr, cfg_volt_attr, notes):
    """Rail channel by in*_label content; configured attrs are the fallback."""
    for label_file in sorted(glob.glob(os.path.join(hwmon_dir, 'in*_label'))):
        try:
            with open(label_file) as f:
                content = f.read().strip()
        except OSError:
            continue
        if content == want_label:
            m = re.search(r'in(\d+)_label$', label_file)
            if m:
                n = m.group(1)
                return (os.path.join(hwmon_dir, f'curr{n}_input'),
                        os.path.join(hwmon_dir, f'in{n}_input'))
    notes.append(f'ina3221 label {want_label} not found by scan - '
                 f'using configured attrs {cfg_curr_attr}/{cfg_volt_attr}')
    return (os.path.join(hwmon_dir, cfg_curr_attr),
            os.path.join(hwmon_dir, cfg_volt_attr))


# ---------------------------------------------------------------------------
# Sampler construction
# ---------------------------------------------------------------------------

JETSON_HEADER = ['t_s', 'gpu_mhz', 'emc_mhz', 'module_w', 'vdd_gpu_w', 'tj_c', 'oc_event_count']
DISCRETE_HEADER = ['t_s', 'sm_mhz', 'mem_mhz', 'power_w', 'temp_c', 'throttle_reasons_hex']


class Sampler:
    """header (incl. t_s), a row() producing the non-t_s fields as strings,
    the recommended period, and resolution notes for the sidecar meta."""

    def __init__(self, header, row_fn, period_s, platform, backend, notes, close_fn=None):
        self.header = header
        self._row_fn = row_fn
        self.period_s = period_s
        self.platform = platform
        self.backend = backend
        self.notes = notes
        self._close_fn = close_fn

    def row(self):
        return self._row_fn()

    def close(self):
        if self._close_fn:
            try:
                self._close_fn()
            except Exception:
                pass


def _devfreq_reader(spec, label, notes):
    path = os.path.join(spec['dir'], 'cur_freq')
    scale = float(spec.get('scale_hz_to_mhz', 1e-06))
    if not os.path.exists(path):
        notes.append(f'SKIP: {label}: {path} missing - column will be blank')

    def rd():
        v = read_num(path)
        return '' if v is None else f'{v * scale:.0f}'
    return rd


def _blank_reader():
    return lambda: ''


def _make_jetson_sampler(cfg):
    notes = []
    clocks = cfg.get('clock_sources', {})
    power = cfg.get('power_sources', {})
    throttle = cfg.get('throttle_sources', {})
    thermal = cfg.get('thermal', {})

    gpu_rd = _devfreq_reader(clocks['gpu_mhz'], 'gpu_mhz', notes) if 'gpu_mhz' in clocks else _blank_reader()
    emc_rd = _devfreq_reader(clocks['emc_mhz'], 'emc_mhz', notes) if 'emc_mhz' in clocks else _blank_reader()

    # module_w: single power_input attr on the named chip, scaled to watts
    module_rd = _blank_reader()
    mspec = power.get('module_w')
    if mspec:
        mdir = resolve_hwmon_dir(mspec['hwmon_name'])
        if mdir is None:
            notes.append(f"SKIP: module_w: no hwmon named {mspec['hwmon_name']} - column will be blank")
        else:
            mpath = os.path.join(mdir, mspec.get('attr', 'power1_input'))
            mscale = float(mspec.get('scale', 1e-06))
            notes.append(f'module_w: {mpath} x {mscale}')

            def module_rd(p=mpath, s=mscale):
                v = read_num(p)
                return '' if v is None else f'{v * s:.3f}'

    # vdd_gpu_w: INA3221 rail, watts = mV * mA / 1e6
    vdd_rd = _blank_reader()
    vspec = power.get('vdd_gpu')
    if vspec:
        vdir = resolve_hwmon_dir(vspec['hwmon_name'])
        if vdir is None:
            notes.append(f"SKIP: vdd_gpu_w: no hwmon named {vspec['hwmon_name']} - column will be blank")
        else:
            curr_path, volt_path = _resolve_ina3221_channel(
                vdir, vspec.get('label', 'VDD_GPU'),
                vspec.get('curr_attr', 'curr1_input'), vspec.get('volt_attr', 'in1_input'), notes)
            notes.append(f'vdd_gpu_w: {curr_path} * {volt_path} / 1e6')

            def vdd_rd(cp=curr_path, vp=volt_path):
                ma, mv = read_num(cp), read_num(vp)
                return '' if (ma is None or mv is None) else f'{ma * mv / 1e6:.3f}'

    # tj_c: hottest-junction zone by type name
    tj_rd = _blank_reader()
    tspec = thermal.get('tj_c')
    if tspec:
        tpath = resolve_thermal_zone_temp(tspec.get('zone_type', 'tj-thermal'))
        if tpath is None:
            notes.append(f"SKIP: tj_c: no thermal zone of type {tspec.get('zone_type')} - column will be blank")
        else:
            tscale = float(tspec.get('scale', 0.001))
            notes.append(f'tj_c: {tpath} x {tscale}')

            def tj_rd(p=tpath, s=tscale):
                v = read_num(p)
                return '' if v is None else f'{v * s:.1f}'

    # oc_event_count: attribute layout on the soctherm_oc chip is undocumented,
    # so discover counter-looking files once and sum them per tick; a blank
    # field beats a guessed layout.
    oc_rd = _blank_reader()
    ospec = None
    for entry in throttle.values():
        if isinstance(entry, dict) and 'hwmon_name' in entry:
            ospec = entry
            break
    if ospec:
        odir = resolve_hwmon_dir(ospec['hwmon_name'])
        if odir is None:
            notes.append(f"SKIP: oc_event_count: no hwmon named {ospec['hwmon_name']} - column will be blank")
        else:
            candidates = [p for p in sorted(glob.glob(os.path.join(odir, '*')))
                          if os.path.isfile(p)
                          and re.search(r'(count|event|cnt)', os.path.basename(p))
                          and read_num(p) is not None]
            if not candidates:
                notes.append(f'SKIP: oc_event_count: no readable counter-like attrs under {odir} - column will be blank')
            else:
                notes.append(f'oc_event_count: sum of {[os.path.basename(p) for p in candidates]} under {odir}')

                def oc_rd(paths=candidates):
                    vals = [read_num(p) for p in paths]
                    vals = [v for v in vals if v is not None]
                    return '' if not vals else f'{int(sum(vals))}'

    readers = [gpu_rd, emc_rd, module_rd, vdd_rd, tj_rd, oc_rd]

    def row():
        return [r() for r in readers]

    period = float(cfg.get('sampler_period_s', 0.1))
    return Sampler(JETSON_HEADER, row, period, 'jetson', 'sysfs', notes)


def _make_pynvml_sampler(cfg, notes):
    import pynvml  # optional dependency; caller catches failure
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    # NVML renamed throttle reasons to event reasons; accept either binding
    reasons_fn = (getattr(pynvml, 'nvmlDeviceGetCurrentClocksEventReasons', None)
                  or getattr(pynvml, 'nvmlDeviceGetCurrentClocksThrottleReasons', None))
    if reasons_fn is None:
        notes.append('SKIP: pynvml has no clocks event/throttle reasons call - column will be blank')

    def row():
        out = []
        for clock in (pynvml.NVML_CLOCK_SM, pynvml.NVML_CLOCK_MEM):
            try:
                out.append(str(pynvml.nvmlDeviceGetClockInfo(handle, clock)))
            except Exception:
                out.append('')
        try:
            out.append(f'{pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0:.2f}')
        except Exception:
            out.append('')
        try:
            out.append(str(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)))
        except Exception:
            out.append('')
        if reasons_fn:
            try:
                out.append(f'0x{reasons_fn(handle):016x}')
            except Exception:
                out.append('')
        else:
            out.append('')
        return out

    period = float(cfg.get('sampler_period_s', 0.2))
    notes.append('backend pynvml: persistent device handle 0')
    return Sampler(DISCRETE_HEADER, row, period, 'discrete', 'pynvml', notes,
                   close_fn=pynvml.nvmlShutdown)


def _make_smi_sampler(cfg, notes):
    clocks = cfg.get('clock_sources', {})
    base_fields = [f.strip() for f in
                   clocks.get('smi_query', 'clocks.sm,clocks.mem,power.draw,temperature.gpu').split(',')]
    # driver renamed clocks_throttle_reasons -> clocks_event_reasons; probe
    # the configured name first, fall back to the legacy one, then to none
    throttle_candidates = [clocks.get('throttle_query', 'clocks_event_reasons.active'),
                           'clocks_throttle_reasons.active']
    state = {'query': None, 'fields': None}

    def try_query(fields):
        try:
            r = subprocess.run(
                ['nvidia-smi', '--query-gpu=' + ','.join(fields), '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if r.returncode != 0 or not r.stdout.strip():
            return None
        return r.stdout.strip().splitlines()[0]

    for tc in throttle_candidates:
        fields = base_fields + [tc]
        if try_query(fields) is not None:
            state['fields'] = fields
            break
    if state['fields'] is None:
        state['fields'] = base_fields
        notes.append('SKIP: no throttle-reasons query accepted by this driver - column will be blank')

    fields = state['fields']

    def pick(parts, field_name):
        try:
            v = parts[fields.index(field_name)].strip()
        except (ValueError, IndexError):
            return ''
        return '' if (not v or 'N/A' in v) else v

    def row():
        line = try_query(fields)
        if line is None:
            return [''] * 5
        parts = line.split(',')
        sm = pick(parts, 'clocks.sm')
        mem = pick(parts, 'clocks.mem')
        power = pick(parts, 'power.draw')
        temp = pick(parts, 'temperature.gpu')
        throttle = ''
        if len(fields) > len(base_fields):
            throttle = pick(parts, fields[-1])
            if throttle and not throttle.startswith('0x'):
                throttle = ''
        return [sm, mem, power, temp, throttle]

    period = float(cfg.get('sampler_period_s_subprocess_fallback', 1.0))
    notes.append('backend nvidia-smi subprocess: period pinned to '
                 f'{period} s (process-spawn cost, see module docstring)')
    return Sampler(DISCRETE_HEADER, row, period, 'discrete', 'nvidia-smi', notes)


def make_sampler(cfg):
    """Build the platform-appropriate sampler from a loaded device config."""
    if cfg.get('platform') == 'jetson':
        return _make_jetson_sampler(cfg)
    notes = []
    try:
        return _make_pynvml_sampler(cfg, notes)
    except Exception as e:
        notes.append(f'pynvml unavailable ({e.__class__.__name__}: {e}) - nvidia-smi subprocess fallback')
        return _make_smi_sampler(cfg, notes)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

_STOP = {'flag': False}


def _handle_signal(signum, frame):
    # finalize gracefully so a parent-killed sampler still leaves valid output
    _STOP['flag'] = True


def pid_alive(pid):
    return os.path.exists(f'/proc/{pid}')


def main():
    ap = argparse.ArgumentParser(description='Device-aware clock/power/thermal sampler')
    ap.add_argument('--device', required=True, help='device config JSON')
    ap.add_argument('--out', required=True, help='output samples CSV')
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument('--pid', type=int, help='sample until this PID exits')
    grp.add_argument('--duration', type=float, help='sample for this many seconds')
    ap.add_argument('--period', type=float, default=None,
                    help='seconds between samples (default: device config)')
    ap.add_argument('--phase', default='', help='label recorded in the sidecar meta')
    args = ap.parse_args()

    try:
        cfg = load_device_config(args.device)
    except (OSError, json.JSONDecodeError) as e:
        die(f'cannot read device config {args.device}: {e}')

    sampler = make_sampler(cfg)
    period = args.period if args.period is not None else sampler.period_s
    if sampler.backend == 'nvidia-smi' and period < 0.5:
        say(f'WARN: period {period}s with subprocess backend - each sample spawns nvidia-smi')
    for n in sampler.notes:
        say(n)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    start_iso = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    rows = 0
    t0 = time.monotonic()
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(sampler.header)
        f.flush()
        n = 0
        while not _STOP['flag']:
            if args.pid is not None and not pid_alive(args.pid):
                break
            if args.duration is not None and time.monotonic() - t0 >= args.duration:
                break
            w.writerow([f'{time.monotonic() - t0:.3f}'] + sampler.row())
            f.flush()
            rows += 1
            n += 1
            # fixed schedule so slow reads do not accumulate cadence drift
            delay = (t0 + n * period) - time.monotonic()
            if delay > 0:
                time.sleep(min(delay, period))

    sampler.close()
    meta = {
        'schema': 'clocksampler/v1',
        'device_tag': cfg.get('device_tag'),
        'platform': sampler.platform,
        'backend': sampler.backend,
        'period_s': period,
        'phase': args.phase,
        'pid': args.pid,
        'duration_s': args.duration,
        'start': start_iso,
        'end': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'rows': rows,
        'header': sampler.header,
        'source_notes': sampler.notes,
    }
    with open(args.out + '.meta.json', 'w') as f:
        json.dump(meta, f, indent=1)
    say(f'sampler done: {rows} rows -> {args.out} (phase: {args.phase or "unlabeled"})')
    sys.exit(0)


if __name__ == '__main__':
    main()
