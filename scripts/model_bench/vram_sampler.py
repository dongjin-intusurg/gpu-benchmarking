#!/usr/bin/env python3
"""Peak memory of one process while it runs -> {peak_mb, source, n_samples}.

  vram_sampler.py --pid <pid> --platform jetson|discrete --out <json> [--interval 0.1]
                  [--baseline-mb <MemAvailable MB taken BEFORE the process started>]

discrete : nvidia-smi --query-compute-apps used_gpu_memory for the pid (device
           memory the process actually holds).
jetson   : the iGPU shares system memory and per-process GPU queries return
           nothing, so this is the /proc/meminfo MemAvailable DROP relative to
           the moment BEFORE the process started - pass --baseline-mb from the
           launcher, since by the time the sampler attaches the process may
           already hold memory (process-delta-meminfo). It includes
           the process's host allocations - an upper bound on the footprint.
Runs until the pid exits. Writes the JSON on exit (also on SIGTERM).
"""
import argparse, json, os, signal, subprocess, sys, time


def mem_available_mb():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable:'):
            return float(line.split()[1]) / 1024.0
    return None


def nvsmi_used_mb(pid):
    try:
        out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return None
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(',')]
        if len(parts) >= 2 and parts[0] == str(pid):
            try:
                return float(parts[1])
            except ValueError:
                return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pid', type=int, required=True)
    ap.add_argument('--platform', required=True, choices=['jetson', 'discrete'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--interval', type=float, default=0.1)
    ap.add_argument('--baseline-mb', type=float)
    a = ap.parse_args()

    peak, n, base = 0.0, 0, None
    src = 'process-peak-sampled' if a.platform == 'discrete' else 'process-delta-meminfo'
    if a.platform == 'jetson':
        base = a.baseline_mb if a.baseline_mb is not None else mem_available_mb()

    def write(*_):
        json.dump({'peak_mb': round(peak, 1), 'source': src, 'n_samples': n,
                   'baseline_available_mb': base}, open(a.out, 'w'), indent=1)
    signal.signal(signal.SIGTERM, lambda *_: (write(), sys.exit(0)))

    while True:
        try:
            os.kill(a.pid, 0)
        except OSError:
            break
        v = nvsmi_used_mb(a.pid) if a.platform == 'discrete' else (
            (base - mem_available_mb()) if base is not None else None)
        if v is not None:
            n += 1
            peak = max(peak, v)
        time.sleep(a.interval)
    write()


if __name__ == '__main__':
    main()
