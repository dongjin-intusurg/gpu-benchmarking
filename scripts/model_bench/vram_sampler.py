#!/usr/bin/env python3
"""Peak memory of one process while it runs -> {peak_mb, source, n_samples, baseline_available_mb}.

  vram_sampler.py --pid <pid> --platform jetson|discrete --out <json> [--interval 0.1] [--baseline-mb <MB>]
discrete: nvidia-smi used_gpu_memory for the pid (source process-peak-sampled).
jetson: the iGPU shares system memory and per-process GPU queries return nothing, so this is the /proc/meminfo
MemAvailable DROP relative to BEFORE the process started (source process-delta-meminfo; pass --baseline-mb
from the launcher, the sampler attaches late). Runs until the pid exits; writes the JSON on exit and SIGTERM.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time


def mem_available_mb():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable:'):
            return float(line.split()[1]) / 1024.0
    return None


def nvsmi_used_mb(pid):
    try:
        out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                              '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return None
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(',')]
        if len(parts) >= 2 and parts[0] == str(pid):
            try:
                return float(parts[1])
            except ValueError:
                return None
    return None


def process_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--platform', required=True, choices=['jetson', 'discrete'])
    parser.add_argument('--out', required=True)
    parser.add_argument('--interval', type=float, default=0.1)
    parser.add_argument('--baseline-mb', type=float)
    args = parser.parse_args()

    discrete = args.platform == 'discrete'
    source = 'process-peak-sampled' if discrete else 'process-delta-meminfo'
    baseline_mb = None
    if not discrete:
        baseline_mb = args.baseline_mb if args.baseline_mb is not None else mem_available_mb()
    peak_mb, n_samples = 0.0, 0

    def write(*_signal_args):
        json.dump({'peak_mb': round(peak_mb, 1), 'source': source, 'n_samples': n_samples,
                   'baseline_available_mb': baseline_mb}, open(args.out, 'w'), indent=1)

    signal.signal(signal.SIGTERM, lambda *_: (write(), sys.exit(0)))

    while process_alive(args.pid):
        if discrete:
            sample_mb = nvsmi_used_mb(args.pid)
        else:
            sample_mb = (baseline_mb - mem_available_mb()) if baseline_mb is not None else None
        if sample_mb is not None:
            n_samples += 1
            peak_mb = max(peak_mb, sample_mb)
        time.sleep(args.interval)
    write()


if __name__ == '__main__':
    main()
