#!/usr/bin/env python3
"""Per-PID clock/power/thermal sampler, device-aware via the device config.

Runs beside a measurement process (--pid) or for a window (--duration) and writes one CSV row per
tick plus a <out>.meta.json sidecar (schema clocksampler/v1). The header is a contract consumed by
drift_report.py and the per-model clock_integrity blocks:
  jetson:   t_s,gpu_mhz,emc_mhz,module_w,vdd_gpu_w,tj_c,oc_event_count
  discrete: t_s,sm_mhz,mem_mhz,power_w,temp_c,throttle_reasons_hex
An unreadable source yields a BLANK field plus one sidecar note, never a crash or a fabricated value.
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

JETSON_HEADER = ['t_s', 'gpu_mhz', 'emc_mhz', 'module_w', 'vdd_gpu_w', 'tj_c', 'oc_event_count']
DISCRETE_HEADER = ['t_s', 'sm_mhz', 'mem_mhz', 'power_w', 'temp_c', 'throttle_reasons_hex']

# Sampling overhead is the reason the sampler cannot perturb the measurement: jetson columns are
# unprivileged sysfs reads (~50 us each, zero GPU work) and pynvml calls on a persistent handle are
# in-process ioctls, so 0.1-0.2 s periods cost well under 0.1% of one core. The nvidia-smi fallback
# spawns a process per tick, which is why its period is pinned to 1.0 s
# (config sampler_period_s_subprocess_fallback) and an explicit --period below 0.5 s draws a WARN.
JETSON_DEFAULT_PERIOD_S = 0.1
PYNVML_DEFAULT_PERIOD_S = 0.2
SMI_FALLBACK_DEFAULT_PERIOD_S = 1.0
SMI_FAST_PERIOD_WARN_S = 0.5


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def die(msg):
    print(f"FATAL: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def load_device_config(path):
    with open(path) as handle:
        return json.load(handle)


# Source resolution is by NAME (hwmon name file, in*_label content, thermal zone type): sysfs
# indices such as hwmon5 are boot-order artifacts and must never be hardcoded. preflight.py
# imports these helpers so its idle baseline reads the exact sources the drift record does.

def resolve_hwmon_dir(hwmon_name):
    """Directory of the hwmon chip whose name file matches, else None."""
    for name_file in sorted(glob.glob('/sys/class/hwmon/hwmon*/name')):
        try:
            with open(name_file) as handle:
                if handle.read().strip() == hwmon_name:
                    return os.path.dirname(name_file)
        except OSError:
            continue
    return None


def resolve_thermal_zone_temp(zone_type):
    """temp file of the thermal zone whose type matches, else None."""
    for type_file in sorted(glob.glob('/sys/devices/virtual/thermal/thermal_zone*/type')):
        try:
            with open(type_file) as handle:
                if handle.read().strip() == zone_type:
                    return os.path.join(os.path.dirname(type_file), 'temp')
        except OSError:
            continue
    return None


def read_num(path):
    """Numeric content of a sysfs file, or None if unreadable/non-numeric."""
    try:
        with open(path) as handle:
            return float(handle.read().strip())
    except (OSError, ValueError):
        return None


def resolve_ina3221_channel(hwmon_dir, want_label, config_curr_attr, config_volt_attr, notes):
    """(current, voltage) input paths of the rail whose in*_label matches; configured attrs are
    the fallback when no label file does."""
    for label_file in sorted(glob.glob(os.path.join(hwmon_dir, 'in*_label'))):
        try:
            with open(label_file) as handle:
                content = handle.read().strip()
        except OSError:
            continue
        if content != want_label:
            continue
        match = re.search(r'in(\d+)_label$', label_file)
        if match:
            channel = match.group(1)
            return (os.path.join(hwmon_dir, f'curr{channel}_input'),
                    os.path.join(hwmon_dir, f'in{channel}_input'))
    notes.append(f'ina3221 label {want_label} not found by scan - '
                 f'using configured attrs {config_curr_attr}/{config_volt_attr}')
    return (os.path.join(hwmon_dir, config_curr_attr),
            os.path.join(hwmon_dir, config_volt_attr))


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


def blank_reader():
    return lambda: ''


def scaled_reader(path, scale, digits):
    def read():
        value = read_num(path)
        return '' if value is None else f'{value * scale:.{digits}f}'
    return read


def devfreq_reader(spec, label, notes):
    path = os.path.join(spec['dir'], 'cur_freq')
    if not os.path.exists(path):
        notes.append(f'SKIP: {label}: {path} missing - column will be blank')
    return scaled_reader(path, float(spec.get('scale_hz_to_mhz', 1e-06)), 0)


def module_power_reader(spec, notes):
    """module_w: single power_input attr on the named chip, scaled to watts."""
    hwmon_dir = resolve_hwmon_dir(spec['hwmon_name'])
    if hwmon_dir is None:
        notes.append(f"SKIP: module_w: no hwmon named {spec['hwmon_name']} - column will be blank")
        return blank_reader()
    path = os.path.join(hwmon_dir, spec.get('attr', 'power1_input'))
    scale = float(spec.get('scale', 1e-06))
    notes.append(f'module_w: {path} x {scale}')
    return scaled_reader(path, scale, 3)


def vdd_gpu_reader(spec, notes):
    """vdd_gpu_w: INA3221 rail, watts = mV * mA / 1e6."""
    hwmon_dir = resolve_hwmon_dir(spec['hwmon_name'])
    if hwmon_dir is None:
        notes.append(f"SKIP: vdd_gpu_w: no hwmon named {spec['hwmon_name']} - column will be blank")
        return blank_reader()
    curr_path, volt_path = resolve_ina3221_channel(
        hwmon_dir, spec.get('label', 'VDD_GPU'),
        spec.get('curr_attr', 'curr1_input'), spec.get('volt_attr', 'in1_input'), notes)
    notes.append(f'vdd_gpu_w: {curr_path} * {volt_path} / 1e6')

    def read():
        milliamps, millivolts = read_num(curr_path), read_num(volt_path)
        return '' if (milliamps is None or millivolts is None) else f'{milliamps * millivolts / 1e6:.3f}'
    return read


def junction_temp_reader(spec, notes):
    """tj_c: hottest-junction zone by type name."""
    path = resolve_thermal_zone_temp(spec.get('zone_type', 'tj-thermal'))
    if path is None:
        notes.append(f"SKIP: tj_c: no thermal zone of type {spec.get('zone_type')} - column will be blank")
        return blank_reader()
    scale = float(spec.get('scale', 0.001))
    notes.append(f'tj_c: {path} x {scale}')
    return scaled_reader(path, scale, 1)


def oc_event_reader(throttle_sources, notes):
    """oc_event_count: the soctherm_oc chip's attribute layout is undocumented, so counter-looking
    files are discovered once and summed per tick; a blank field beats a guessed layout."""
    spec = next((entry for entry in throttle_sources.values()
                 if isinstance(entry, dict) and 'hwmon_name' in entry), None)
    if not spec:
        return blank_reader()
    hwmon_dir = resolve_hwmon_dir(spec['hwmon_name'])
    if hwmon_dir is None:
        notes.append(f"SKIP: oc_event_count: no hwmon named {spec['hwmon_name']} - column will be blank")
        return blank_reader()
    counters = [path for path in sorted(glob.glob(os.path.join(hwmon_dir, '*')))
                if os.path.isfile(path)
                and re.search(r'(count|event|cnt)', os.path.basename(path))
                and read_num(path) is not None]
    if not counters:
        notes.append(f'SKIP: oc_event_count: no readable counter-like attrs under {hwmon_dir} '
                     '- column will be blank')
        return blank_reader()
    notes.append(f'oc_event_count: sum of {[os.path.basename(path) for path in counters]} under {hwmon_dir}')

    def read():
        values = [value for value in (read_num(path) for path in counters) if value is not None]
        return '' if not values else f'{int(sum(values))}'
    return read


def make_jetson_sampler(config):
    notes = []
    clocks = config.get('clock_sources', {})
    power = config.get('power_sources', {})
    throttle = config.get('throttle_sources', {})
    thermal = config.get('thermal', {})

    readers = [
        devfreq_reader(clocks['gpu_mhz'], 'gpu_mhz', notes) if 'gpu_mhz' in clocks else blank_reader(),
        devfreq_reader(clocks['emc_mhz'], 'emc_mhz', notes) if 'emc_mhz' in clocks else blank_reader(),
        module_power_reader(power['module_w'], notes) if power.get('module_w') else blank_reader(),
        vdd_gpu_reader(power['vdd_gpu'], notes) if power.get('vdd_gpu') else blank_reader(),
        junction_temp_reader(thermal['tj_c'], notes) if thermal.get('tj_c') else blank_reader(),
        oc_event_reader(throttle, notes),
    ]

    def row():
        return [read() for read in readers]

    period = float(config.get('sampler_period_s', JETSON_DEFAULT_PERIOD_S))
    return Sampler(JETSON_HEADER, row, period, 'jetson', 'sysfs', notes)


def make_pynvml_sampler(config, notes):
    import pynvml  # optional dependency; caller catches failure
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    # NVML renamed throttle reasons to event reasons; accept either binding
    reasons_fn = (getattr(pynvml, 'nvmlDeviceGetCurrentClocksEventReasons', None)
                  or getattr(pynvml, 'nvmlDeviceGetCurrentClocksThrottleReasons', None))
    if reasons_fn is None:
        notes.append('SKIP: pynvml has no clocks event/throttle reasons call - column will be blank')

    def field(query):
        try:
            return query()
        except Exception:
            return ''

    def row():
        fields = [field(lambda clock=clock: str(pynvml.nvmlDeviceGetClockInfo(handle, clock)))
                  for clock in (pynvml.NVML_CLOCK_SM, pynvml.NVML_CLOCK_MEM)]
        fields.append(field(lambda: f'{pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0:.2f}'))
        fields.append(field(lambda: str(pynvml.nvmlDeviceGetTemperature(handle,
                                                                        pynvml.NVML_TEMPERATURE_GPU))))
        fields.append(field(lambda: f'0x{reasons_fn(handle):016x}') if reasons_fn else '')
        return fields

    period = float(config.get('sampler_period_s', PYNVML_DEFAULT_PERIOD_S))
    notes.append('backend pynvml: persistent device handle 0')
    return Sampler(DISCRETE_HEADER, row, period, 'discrete', 'pynvml', notes,
                   close_fn=pynvml.nvmlShutdown)


def smi_query_line(fields):
    """First line of one nvidia-smi CSV query, None on any failure."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=' + ','.join(fields), '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return result.stdout.strip().splitlines()[0]


def make_smi_sampler(config, notes):
    clocks = config.get('clock_sources', {})
    base_fields = [name.strip() for name in
                   clocks.get('smi_query', 'clocks.sm,clocks.mem,power.draw,temperature.gpu').split(',')]
    # the driver renamed clocks_throttle_reasons -> clocks_event_reasons: probe the configured
    # name first, then the legacy one, then go without the column
    throttle_candidates = [clocks.get('throttle_query', 'clocks_event_reasons.active'),
                           'clocks_throttle_reasons.active']
    fields = None
    for candidate in throttle_candidates:
        if smi_query_line(base_fields + [candidate]) is not None:
            fields = base_fields + [candidate]
            break
    if fields is None:
        fields = base_fields
        notes.append('SKIP: no throttle-reasons query accepted by this driver - column will be blank')
    has_throttle = len(fields) > len(base_fields)

    def pick(parts, field_name):
        try:
            value = parts[fields.index(field_name)].strip()
        except (ValueError, IndexError):
            return ''
        return '' if (not value or 'N/A' in value) else value

    def row():
        line = smi_query_line(fields)
        if line is None:
            return [''] * 5
        parts = line.split(',')
        throttle = ''
        if has_throttle:
            throttle = pick(parts, fields[-1])
            if throttle and not throttle.startswith('0x'):
                throttle = ''
        return [pick(parts, 'clocks.sm'), pick(parts, 'clocks.mem'), pick(parts, 'power.draw'),
                pick(parts, 'temperature.gpu'), throttle]

    period = float(config.get('sampler_period_s_subprocess_fallback', SMI_FALLBACK_DEFAULT_PERIOD_S))
    notes.append('backend nvidia-smi subprocess: period pinned to '
                 f'{period} s (process-spawn cost, see module docstring)')
    return Sampler(DISCRETE_HEADER, row, period, 'discrete', 'nvidia-smi', notes)


def make_sampler(config):
    """Build the platform-appropriate sampler from a loaded device config."""
    if config.get('platform') == 'jetson':
        return make_jetson_sampler(config)
    notes = []
    try:
        return make_pynvml_sampler(config, notes)
    except Exception as error:
        notes.append(f'pynvml unavailable ({error.__class__.__name__}: {error}) '
                     '- nvidia-smi subprocess fallback')
        return make_smi_sampler(config, notes)


STOP_REQUESTED = {'flag': False}


def request_stop(signum, frame):
    # finalize gracefully so a parent-killed sampler still leaves a complete sidecar
    STOP_REQUESTED['flag'] = True


def pid_alive(pid):
    return os.path.exists(f'/proc/{pid}')


def parse_args():
    parser = argparse.ArgumentParser(description='Device-aware clock/power/thermal sampler')
    parser.add_argument('--device', required=True, help='device config JSON')
    parser.add_argument('--out', required=True, help='output samples CSV')
    stop = parser.add_mutually_exclusive_group(required=True)
    stop.add_argument('--pid', type=int, help='sample until this PID exits')
    stop.add_argument('--duration', type=float, help='sample for this many seconds')
    parser.add_argument('--period', type=float, default=None,
                        help='seconds between samples (default: device config)')
    parser.add_argument('--phase', default='', help='label recorded in the sidecar meta')
    return parser.parse_args()


def sample_until_done(sampler, period, out_path, pid, duration):
    """Write rows on a fixed schedule (t0 + n*period) so slow reads never accumulate cadence
    drift; every row is flushed so a killed sampler still leaves a parseable CSV."""
    rows = 0
    t0 = time.monotonic()
    with open(out_path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(sampler.header)
        handle.flush()
        while not STOP_REQUESTED['flag']:
            if pid is not None and not pid_alive(pid):
                break
            if duration is not None and time.monotonic() - t0 >= duration:
                break
            writer.writerow([f'{time.monotonic() - t0:.3f}'] + sampler.row())
            handle.flush()
            rows += 1
            delay = (t0 + rows * period) - time.monotonic()
            if delay > 0:
                time.sleep(min(delay, period))
    return rows


def main():
    args = parse_args()
    try:
        config = load_device_config(args.device)
    except (OSError, json.JSONDecodeError) as error:
        die(f'cannot read device config {args.device}: {error}')

    sampler = make_sampler(config)
    period = args.period if args.period is not None else sampler.period_s
    if sampler.backend == 'nvidia-smi' and period < SMI_FAST_PERIOD_WARN_S:
        say(f'WARN: period {period}s with subprocess backend - each sample spawns nvidia-smi')
    for note in sampler.notes:
        say(note)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    start_iso = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    rows = sample_until_done(sampler, period, args.out, args.pid, args.duration)
    sampler.close()

    meta = {
        'schema': 'clocksampler/v1',
        'device_tag': config.get('device_tag'),
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
    with open(args.out + '.meta.json', 'w') as handle:
        json.dump(meta, handle, indent=1)
    say(f'sampler done: {rows} rows -> {args.out} (phase: {args.phase or "unlabeled"})')
    sys.exit(0)


if __name__ == '__main__':
    main()
