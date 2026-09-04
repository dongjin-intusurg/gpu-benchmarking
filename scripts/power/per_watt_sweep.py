#!/usr/bin/env python3
"""Per-watt sweep of one operating point: compute per watt (TOPS/W per precision) and
bandwidth per watt (GB/s per W) against utilization, on the device config's rails.

A square GEMM of known work (2 N^3 ops; or a 256 MB DRAM copy for `copy`) is
duty-cycled inside a fixed period so the GPU is busy a target fraction of wall
time (10 .. 100 %). Per (precision, target) point: delivered throughput (ops /
wall), kernel throughput (ops / busy), module and GPU-rail power, clock,
temperature, throttle / overcurrent events; then watts vs delivered throughput
is fitted over the >= 45 % points (and over the unsaturated points when a power
cap pins the board). Clocks are deliberately NOT locked: DVFS under partial
load is part of the power behaviour being measured - the point's knob (mode /
power limit / clock-cap recipe) is what is held.

  per_watt_sweep.py --device <cfg.json> --out <dir> [--n 4096] [--precisions fp16,int8,fp8,copy]
                    [--targets 10,25,40,50,60,70,80,90,100] [--seconds 15] [--period-ms 40] [--idle-seconds 8]

jetson: rails from the config's power_sources (module_w on the named hwmon chip,
        vdd_gpu by ina3221 label), clock from clock_sources.gpu_mhz.dir, tj from
        thermal.tj_c.zone_type, overcurrent events from throttle_sources.
discrete: NVML board power / SM clock / temperature / throttle-reason bits, plus
        the total-energy counter (exact energy over each point).
Outputs: power_tops_sweep.json (points + fits + provenance), power_samples.csv, power_tops_points.csv.
Stdout lines `idle ...`, `== <precision> ...`, `  fit ...`, `  power-cap knee ...`, `wrote ...` are grepped
by the stage.
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import threading
import time
from datetime import datetime

import torch

NAN = float('nan')
COPY_ELEMENTS = 128 * 1024 * 1024          # 256 MB fp16 source for the DRAM copy workload
COPY_BYTES_PER_CALL = 2.0 * COPY_ELEMENTS * 2   # read + write
FIT_MIN_BUSY_PCT = 45.0
SATURATION_FRACTION = 0.98                 # of the highest module W seen: the board is pinned at its cap
SAMPLE_COLUMNS = ['t', 'module_w', 'vdd_gpu_w', 'gpu_mhz', 'tj_c', 'oc']


def is_nan(x):
    return x != x


def hwmon_dir(chip_name):
    for name_file in glob.glob('/sys/class/hwmon/hwmon*/name'):
        try:
            if open(name_file).read().strip() == chip_name:
                return os.path.dirname(name_file)
        except OSError:
            pass
    return None


def read_float(path, default=NAN):
    try:
        return float(open(path).read().strip())
    except Exception:
        return default


def thermal_zone(zone_type):
    for type_file in glob.glob('/sys/devices/virtual/thermal/thermal_zone*/type'):
        try:
            if open(type_file).read().strip() == zone_type:
                return os.path.join(os.path.dirname(type_file), 'temp')
        except OSError:
            pass
    return None


class Rails:
    """Jetson sysfs rails, resolved from the device config (hwmon by chip name, never hwmonN)."""

    def __init__(self, config):
        power_sources = config.get('power_sources') or {}
        clock_source = (config.get('clock_sources') or {}).get('gpu_mhz') or {}
        thermal = (config.get('thermal') or {}).get('tj_c') or {}
        throttle = (config.get('throttle_sources') or {}).get('soctherm_oc') or {}

        module = power_sources.get('module_w') or {}
        module_dir = hwmon_dir(module.get('hwmon_name', 'ina238'))
        self.module = os.path.join(module_dir, module.get('attr', 'power1_input')) if module_dir else None
        self.module_scale = float(module.get('scale', 1e-6))

        self.gpu_current = self.gpu_voltage = None
        gpu_rail = power_sources.get('vdd_gpu') or {}
        rail_dir = hwmon_dir(gpu_rail.get('hwmon_name', 'ina3221'))
        if rail_dir:
            for label_file in glob.glob(os.path.join(rail_dir, 'in*_label')):
                if open(label_file).read().strip() == gpu_rail.get('label', 'VDD_GPU'):
                    channel = os.path.basename(label_file)[2:-6]
                    self.gpu_current = os.path.join(rail_dir, f'curr{channel}_input')
                    self.gpu_voltage = os.path.join(rail_dir, f'in{channel}_input')

        self.devfreq = clock_source.get('dir', '/sys/class/devfreq/gpu-gpc-0')
        self.clock_scale = float(clock_source.get('scale_hz_to_mhz', 1e-6))
        self.tj = thermal_zone(thermal.get('zone_type', 'tj-thermal'))
        self.tj_scale = float(thermal.get('scale', 1e-3))
        oc_dir = hwmon_dir(throttle.get('hwmon_name', 'soctherm_oc'))
        self.oc_counters = sorted(glob.glob(os.path.join(oc_dir, '*event_cnt'))) if oc_dir else []
        self.notes = [f"module_w: {self.module}", f"vdd_gpu: {self.gpu_current}",
                      f"gpu clock: {self.devfreq}", f"tj: {self.tj}", f"oc counters: {len(self.oc_counters)}"]

    def sample(self):
        return dict(
            module_w=read_float(self.module) * self.module_scale if self.module else NAN,
            vdd_gpu_w=(read_float(self.gpu_current) * read_float(self.gpu_voltage) / 1e6)
            if self.gpu_current else NAN,
            gpu_mhz=read_float(os.path.join(self.devfreq, 'cur_freq')) * self.clock_scale,
            tj_c=read_float(self.tj) * self.tj_scale if self.tj else NAN,
            oc=sum(read_float(path, 0.0) for path in self.oc_counters))

    def provenance(self):
        info = {}
        try:
            query = subprocess.run(['nvpmodel', '-q'], capture_output=True, text=True, timeout=5)
            info['nvpmodel'] = query.stdout.strip().splitlines()[0]
        except Exception as error:
            info['nvpmodel'] = f'unavailable: {error}'
        info['devfreq'] = {key: read_float(os.path.join(self.devfreq, key)) * self.clock_scale
                           for key in ('min_freq', 'max_freq')}
        try:
            info['devfreq']['governor'] = open(os.path.join(self.devfreq, 'governor')).read().strip()
        except Exception:
            pass
        return info


class NvmlRails:
    """Discrete-GPU rails via NVML: board power (nvmlDeviceGetPowerUsage, ~1 s hardware average on most
    boards), SM clock, GPU temperature; 'oc' counts samples with an active SW/HW power-cap or thermal
    throttle reason (the analogue of the Jetson overcurrent event counter)."""

    REASON_BITS = ((0x4, 'sw_power_cap'), (0x8, 'hw_slowdown'), (0x20, 'sw_thermal'),
                   (0x40, 'hw_thermal'), (0x80, 'hw_power_brake'))
    THROTTLE_MASK = 0x4 | 0x8 | 0x20 | 0x40 | 0x80

    def __init__(self, config=None):
        import pynvml
        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.reasons = (getattr(pynvml, 'nvmlDeviceGetCurrentClocksEventReasons', None)
                        or getattr(pynvml, 'nvmlDeviceGetCurrentClocksThrottleReasons', None))
        self.throttled = 0
        self.notes = ['nvml: power usage, SM clock, temperature, clock-event reasons, total-energy counter']
        try:
            self.limit_w = pynvml.nvmlDeviceGetPowerManagementLimit(self.handle) / 1000.0
        except Exception:
            self.limit_w = NAN

    def energy_j(self):
        """NVML total-energy accumulator (mJ since driver load) -> J. Exact where sampled power lags:
        power.draw is a ~1 s hardware average, so short windows under-read by ~10 %; dE/dt does not."""
        try:
            return self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1000.0
        except Exception:
            return NAN

    def sample(self):
        nvml = self.nvml
        try:
            reasons = self.reasons(self.handle) if self.reasons else 0
        except Exception:
            reasons = 0
        if reasons & self.THROTTLE_MASK:
            self.throttled += 1
        row = dict(module_w=nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0, vdd_gpu_w=NAN,
                   gpu_mhz=float(nvml.nvmlDeviceGetClockInfo(self.handle, nvml.NVML_CLOCK_SM)),
                   tj_c=float(nvml.nvmlDeviceGetTemperature(self.handle, nvml.NVML_TEMPERATURE_GPU)),
                   oc=float(self.throttled))
        for bit, name in self.REASON_BITS:
            row[name] = 1.0 if (reasons & bit) else 0.0
        return row

    def provenance(self):
        info = {}
        try:
            fields = 'power.limit,power.default_limit,clocks.max.sm,driver_version,pstate'
            query = subprocess.run(['nvidia-smi', f'--query-gpu={fields}', '--format=csv,noheader'],
                                   capture_output=True, text=True, timeout=5).stdout.strip()
            info['nvidia_smi'] = query
            info['power_limit'] = query.split(',')[0].strip()
        except Exception as error:
            info['nvidia_smi'] = f'unavailable: {error}'
        return info


class Sampler(threading.Thread):
    # the default period is deliberately not commensurate with the 40 ms duty period (ina238 is unaveraged)
    def __init__(self, rails, period=0.0731):
        super().__init__(daemon=True)
        self.rails = rails
        self.period = period
        self.rows = []
        self.stop = threading.Event()
        self.t0 = time.monotonic()

    def run(self):
        while not self.stop.is_set():
            row = self.rails.sample()
            row['t'] = time.monotonic() - self.t0
            self.rows.append(row)
            time.sleep(self.period)

    def window(self, t_from, t_to):
        return [row for row in self.rows if t_from <= row['t'] <= t_to]


def make_workload(n, precision):
    """One callable launching a single GEMM (or DRAM copy) of the precision on the GPU."""
    device = 'cuda'
    if precision == 'fp16':
        a = torch.randn(n, n, device=device, dtype=torch.float16)
        b = torch.randn(n, n, device=device, dtype=torch.float16)
        return lambda: torch.matmul(a, b)
    if precision == 'int8':
        a = torch.randint(-127, 127, (n, n), device=device, dtype=torch.int8)
        # column-major B: without it torch._int_mm falls off the fast int8 tensor path
        # (measured 141 vs ~430 TOPS on sm_120)
        b = torch.randint(-127, 127, (n, n), device=device, dtype=torch.int8).t().contiguous().t()
        return lambda: torch._int_mm(a, b)
    if precision == 'copy':
        source = torch.randn(COPY_ELEMENTS, device=device, dtype=torch.float16)
        destination = torch.empty_like(source)
        return lambda: destination.copy_(source)
    if precision == 'fp8':
        a = torch.randn(n, n, device=device).to(torch.float8_e4m3fn)
        b = torch.randn(n, n, device=device).t().contiguous().t().to(torch.float8_e4m3fn)
        scale_a = torch.tensor(1.0, device=device)
        scale_b = torch.tensor(1.0, device=device)
        return lambda: torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float16)
    raise ValueError(precision)


def kernel_ms(workload, reps=20):
    for _ in range(5):
        workload()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        workload()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps


def read_energy_j(rails):
    return rails.energy_j() if rails is not None and hasattr(rails, 'energy_j') else NAN


def run_point(workload, ops_per_call, k_ms, target_pct, seconds, period_ms, sampler, scale=1e12, rails=None):
    """Duty-cycle, closed loop: per period launch enough GEMMs to fill target % of it,
    re-estimating the per-call time from the running busy tally (the kernel speeds up
    as DVFS raises the clock under load), then sleep the remainder of the period."""
    period_s = period_ms / 1000.0
    full = target_pct >= 100
    energy_start = read_energy_j(rails)
    t_start = time.monotonic() - sampler.t0
    wall_start = time.monotonic()
    busy_ms = 0.0
    n_calls = 0
    call_ms_estimate = k_ms
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    while time.monotonic() - wall_start < seconds:
        period_start = time.monotonic()
        if full:
            calls = 32
        else:
            calls = max(1, int(round(target_pct / 100.0 * period_s / (call_ms_estimate / 1000.0))))
        start.record()
        for _ in range(calls):
            workload()
        end.record()
        torch.cuda.synchronize()
        burst_ms = start.elapsed_time(end)
        busy_ms += burst_ms
        n_calls += calls
        call_ms_estimate = 0.5 * call_ms_estimate + 0.5 * burst_ms / calls
        if not full:
            rest = period_s - (time.monotonic() - period_start)
            if rest > 0:
                time.sleep(rest)
    wall_s = time.monotonic() - wall_start
    t_end = time.monotonic() - sampler.t0
    energy_end = read_energy_j(rails)
    settle_s = min(3.0, seconds / 3.0)
    window = sampler.window(t_start + settle_s, t_end)

    def mean(key):
        values = [row[key] for row in window if not is_nan(row[key])]
        return sum(values) / len(values) if values else NAN

    ops = ops_per_call * n_calls
    energy_j = energy_end - energy_start if not (is_nan(energy_start) or is_nan(energy_end)) else NAN
    throttle_pct = {name: (100.0 * sum(row.get(name, 0.0) for row in window) / len(window) if window else NAN)
                    for _, name in getattr(rails, 'REASON_BITS', ())}
    have_energy = not is_nan(energy_j)
    return dict(target_pct=target_pct, busy_pct=100.0 * busy_ms / 1000.0 / wall_s, wall_s=wall_s,
                calls=n_calls, energy_j=energy_j,
                mean_w_energy=(energy_j / wall_s if have_energy else NAN),
                j_per_unit=(energy_j / (ops_per_call * n_calls / scale) if have_energy and n_calls else NAN),
                throttle_pct=throttle_pct,
                delivered_tops=ops / wall_s / scale, kernel_tops=ops / (busy_ms / 1000.0) / scale,
                module_w=mean('module_w'), vdd_gpu_w=mean('vdd_gpu_w'), gpu_mhz=mean('gpu_mhz'),
                tj_c=mean('tj_c'),
                gpu_mhz_min=min(row['gpu_mhz'] for row in window) if window else NAN,
                oc_events=(window[-1]['oc'] - window[0]['oc']) if len(window) > 1 else 0,
                n_samples=len(window))


def linear_fit(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    if sxx == 0:
        return None
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    return dict(slope=slope, intercept=intercept, r2=(1 - ss_res / ss_tot) if ss_tot else 1.0, n=n)


def fit_watts_vs_throughput(points):
    return linear_fit([point['delivered_tops'] for point in points], [point['module_w'] for point in points])


def provenance(config, rails, config_path):
    info = dict(date=datetime.now().isoformat(timespec='seconds'), torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0), device_config=os.path.abspath(config_path),
                device_tag=config.get('device_tag'), platform=config.get('platform'),
                power_point=(config.get('power_point') or {}).get('point'), rails=rails.notes,
                clocks='NOT locked (DVFS is part of the measurement)')
    info.update(rails.provenance())
    try:
        query = subprocess.run(['systemctl', 'is-active', 'graphical.target'], capture_output=True, text=True)
        info['graphical_target'] = query.stdout.strip()
    except Exception:
        pass
    return info


def measure_idle(sampler, idle_seconds):
    """Module and GPU-rail idle power over the first idle_seconds (the first second discarded)."""
    time.sleep(idle_seconds)
    idle = sampler.window(1.0, idle_seconds)
    idle_w = sum(row['module_w'] for row in idle) / len(idle)
    gpu_rail = [row['vdd_gpu_w'] for row in idle if not is_nan(row['vdd_gpu_w'])]
    idle_gpu_w = sum(gpu_rail) / len(gpu_rail) if gpu_rail else NAN
    return idle_w, idle_gpu_w


def print_point(target_pct, point):
    print(f"  target {target_pct:5.0f}%  busy {point['busy_pct']:5.1f}%  "
          f"delivered {point['delivered_tops']:6.1f} TOPS  kernel {point['kernel_tops']:6.1f}  "
          f"E {point.get('energy_j', NAN):6.0f} J ({point.get('mean_w_energy', NAN):5.1f} W exact, "
          f"{point.get('j_per_unit', NAN):.3f} J/unit)  "
          f"module {point['module_w']:6.1f} W  VDD_GPU {point['vdd_gpu_w']:5.1f} W  "
          f"gpu {point['gpu_mhz']:.0f} MHz (min {point['gpu_mhz_min']:.0f})  Tj {point['tj_c']:.0f}C  "
          f"oc {point['oc_events']:.0f}  TOPS/W {point['tops_per_w']:.3f} "
          f"(above idle {point['tops_per_w_above_idle']:.3f})")


def sweep_precision(args, precision, sampler, rails, idle_w):
    """Every target of one precision: the per-point rows, the fits and the power-cap knee."""
    if precision == 'copy':
        ops_per_call, unit, scale = COPY_BYTES_PER_CALL, 'GB/s', 1e9
    else:
        ops_per_call, unit, scale = 2.0 * args.n ** 3, 'TOPS', 1e12
    workload = make_workload(args.n, precision)
    k_ms = kernel_ms(workload)
    full_rate = ops_per_call / (k_ms / 1e3) / scale
    print(f'== {precision}: kernel {k_ms:.3f} ms -> {full_rate:.1f} {unit} at 100 % busy')
    points = []
    for target_pct in [float(x) for x in args.targets.split(',')]:
        point = run_point(workload, ops_per_call, k_ms, target_pct, args.seconds, args.period_ms, sampler,
                          scale=scale, rails=rails)
        point['tops_per_w'] = point['delivered_tops'] / point['module_w']
        if not is_nan(point.get('mean_w_energy')):
            point['tops_per_w_energy'] = point['delivered_tops'] / point['mean_w_energy']
        point['tops_per_w_above_idle'] = point['delivered_tops'] / max(point['module_w'] - idle_w, 1e-9)
        point['tops_per_gpu_w'] = point['delivered_tops'] / point['vdd_gpu_w']
        points.append(point)
        print_point(target_pct, point)
        time.sleep(2.0)

    fit_ge50 = fit_watts_vs_throughput([point for point in points if point['busy_pct'] >= FIT_MIN_BUSY_PCT])
    fit_all = fit_watts_vs_throughput(points)
    # power-cap saturation: once the board pins at its limit, extra throughput costs clock, not watts,
    # so those points are a horizontal segment in the W-vs-throughput plane and must not be fitted.
    # Fit below the knee instead.
    cap_w = max((point['module_w'] for point in points), default=0.0)
    unsaturated = [point for point in points if point['module_w'] < SATURATION_FRACTION * cap_w]
    fit_unsaturated = fit_watts_vs_throughput(unsaturated) if len(unsaturated) >= 3 else None
    knee = min((point for point in points if point['module_w'] >= SATURATION_FRACTION * cap_w),
               key=lambda point: point['busy_pct'], default=None)
    result = dict(kernel_ms=k_ms, unit=unit, ops_per_call=ops_per_call, points=points,
                  fit_w_vs_tops_ge50=fit_ge50, fit_w_vs_tops_all=fit_all,
                  fit_w_vs_tops_unsaturated=fit_unsaturated,
                  saturation_knee=dict(busy_pct=knee['busy_pct'], delivered=knee['delivered_tops'],
                                       module_w=knee['module_w'], gpu_mhz=knee['gpu_mhz']) if knee else None,
                  n_unsaturated=len(unsaturated), n_points=len(points))
    if fit_ge50:
        print(f"  fit (>=45 % busy): module W = {fit_ge50['intercept']:.1f} + {fit_ge50['slope']:.3f} "
              f"W per {unit} x {unit}, R2 {fit_ge50['r2']:.4f}")
    if fit_unsaturated:
        print(f"  fit (unsaturated, {len(unsaturated)}/{len(points)} pts below the cap): "
              f"module W = {fit_unsaturated['intercept']:.1f} + {fit_unsaturated['slope']:.3f} per {unit}, "
              f"R2 {fit_unsaturated['r2']:.4f}")
    if knee:
        print(f"  power-cap knee: busy {knee['busy_pct']:.0f} % at {knee['module_w']:.0f} W "
              f"({knee['delivered_tops']:.0f} {unit}, {knee['gpu_mhz']:.0f} MHz) "
              "— beyond this the card trades clock, not watts")
    del workload
    torch.cuda.empty_cache()
    time.sleep(5.0)
    return result


def write_points_csv(path, precisions):
    """Flat per-point table; the columns follow the last precision's first point (throttle_pct unpacked)."""
    last_points = list(precisions.values())[-1]['points']
    flat = [key for key in last_points[0].keys() if key != 'throttle_pct']
    columns = ['precision'] + flat + ['throttle_sw_power_cap_pct', 'throttle_hw_thermal_pct']
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for precision, result in precisions.items():
            for point in result['points']:
                row = {key: point[key] for key in flat}
                throttle = point.get('throttle_pct') or {}
                row['throttle_sw_power_cap_pct'] = throttle.get('sw_power_cap')
                row['throttle_hw_thermal_pct'] = throttle.get('hw_thermal')
                writer.writerow(dict(precision=precision, **row))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', required=True,
                        help='device config JSON (the derived config of the point)')
    parser.add_argument('--out', required=True)
    parser.add_argument('--n', type=int, default=4096)
    parser.add_argument('--precisions', default='fp16,int8,fp8,copy',
                        help='GEMM precisions and/or copy (DRAM bandwidth per watt)')
    parser.add_argument('--targets', default='10,25,40,50,60,70,80,90,100')
    parser.add_argument('--seconds', type=float, default=15.0)
    parser.add_argument('--period-ms', type=float, default=40.0)
    parser.add_argument('--idle-seconds', type=float, default=8.0)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    config = json.load(open(args.device))
    rails = Rails(config) if config.get('platform') == 'jetson' else NvmlRails(config)
    sampler = Sampler(rails, period=float(config.get('sampler_period_s', 0.0731)))
    sampler.start()
    info = provenance(config, rails, args.device)
    print('provenance:', json.dumps(info))
    idle_w, idle_gpu_w = measure_idle(sampler, args.idle_seconds)
    print(f'idle baseline: module {idle_w:.1f} W, VDD_GPU {idle_gpu_w:.1f} W')
    out = dict(provenance=info, n=args.n, ops_per_call=2.0 * args.n ** 3, idle_module_w=idle_w,
               idle_vdd_gpu_w=idle_gpu_w, precisions={})
    for precision in args.precisions.split(','):
        out['precisions'][precision] = sweep_precision(args, precision, sampler, rails, idle_w)
    sampler.stop.set()
    sampler.join()
    with open(os.path.join(args.out, 'power_tops_sweep.json'), 'w') as handle:
        json.dump(out, handle, indent=1)
    with open(os.path.join(args.out, 'power_samples.csv'), 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_COLUMNS)
        writer.writeheader()
        writer.writerows(sampler.rows)
    write_points_csv(os.path.join(args.out, 'power_tops_points.csv'), out['precisions'])
    print('wrote', args.out)


if __name__ == '__main__':
    main()
