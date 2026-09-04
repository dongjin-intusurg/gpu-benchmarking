#!/usr/bin/env python3
"""Calibrate ops-per-tensor-instruction against the known-truth GEMM engine.

  calibrate_ops_per_tinst.py --engine <gemm.engine> --precision int8|fp16|fp8 [--arch-flops N] [--extra "..."]
                             [--out calibration.json]
The roofline classifier's per-arch, per-precision constant does not transfer across architectures, so it is
measured on the box: one NCU counter pass (counts only, no clock lock needed) over an engine whose true op
count is known exactly, truth / sum(tensor insts), snapped to a power of two. Never edits the device
config - prints the line to paste. Exit 0 calibrated; 1 environment/run/parse failure or snap-tolerance FAIL.
"""
import argparse
import collections
import csv
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Pre-declared policy (not tuned post-hoc): MMA shapes make the true constant an exact power of two,
# so the raw quotient must land within this relative distance of one. A miss means the engine is not
# a clean single-precision tactic (TensorRT mixed precisions, or it is not the known-truth GEMM) and
# the run fails loudly rather than shipping a blended constant.
SNAP_TOL = 0.10

# The selftest 4096^3 MatMul: TOTAL ops with FMA=2 already inside, so ops_per_tinst =
# arch_flops / sum(tensor insts) with no extra factor of 2 anywhere.
KNOWN_TRUTH_FLOPS = 2 * 4096 ** 3

TENSOR_INST_METRIC = 'sm__inst_executed_pipe_tensor.sum'
DURATION_METRIC = 'gpu__time_duration.sum'
NCU_METRICS = f'{TENSOR_INST_METRIC},{DURATION_METRIC}'


def say(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}")


def die(message, code=1):
    print(f"FATAL: {message}", file=sys.stderr)
    sys.exit(code)


def resolve_tools():
    """Absolute ncu and trtexec paths (sudo's secure_path would not find them otherwise)."""
    # trtexec lives off-PATH on Jetson - the standard idiom appends the TRT bin dir
    os.environ['PATH'] = os.environ.get('PATH', '') + ':/usr/src/tensorrt/bin'
    ncu_bin = shutil.which('ncu')
    trtexec_bin = shutil.which('trtexec')
    if not ncu_bin:
        die("ncu not on PATH - install Nsight Compute (counter access also needs "
            "root or NVreg_RestrictProfilingToAdminUsers=0)")
    if not trtexec_bin:
        die("trtexec not found on PATH or /usr/src/tensorrt/bin")
    return ncu_bin, trtexec_bin


def run_ncu_pass(engine, precision, extra, ncu_bin, trtexec_bin, csv_path, log_path):
    # sudo resets the environment - LD_LIBRARY_PATH must ride along or the profiled trtexec
    # can't load libnvinfer* from non-ldconfig TRT installs
    command = ['sudo', f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}",
               ncu_bin, '--csv', '--target-processes', 'all', '-f',
               '--metrics', NCU_METRICS, '--log-file', str(csv_path),
               trtexec_bin, f'--loadEngine={engine}',
               # exactly ONE inference: without --duration=0, trtexec's default 3-second duration
               # floor keeps enqueueing under ncu (~a dozen inferences fit even at replay speed),
               # every launch lands in the instruction sum, and the quotient shrinks by that
               # factor - observed 80.31 on sm_120 = 1024 / ~12.75 inferences
               '--iterations=1', '--duration=0', '--warmUp=0'] + shlex.split(extra)
    say(f"ncu counter pass over {engine.name} ({precision})...")
    with open(log_path, 'w') as log:
        returncode = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    if returncode == 0 and csv_path.is_file():
        return
    tail = ''
    if log_path.is_file():
        tail = ''.join(log_path.read_text(errors='ignore').splitlines(True)[-15:])
    die(f"ncu pass failed (rc={returncode}); log tail from {log_path}:\n{tail}\n"
        "hints: counter access needs root (sudo) or the "
        "NVreg_RestrictProfilingToAdminUsers=0 modprobe option; check that "
        "LD_LIBRARY_PATH covers the TRT libs")


def metrics_per_kernel(csv_path):
    """kernel ID -> {metric name: summed value}; the ==PROF== preamble is skipped and 'n/a' rows drop
    out (never zero-filled)."""
    lines = csv_path.read_text(errors='ignore').splitlines(True)
    header_index = next((i for i, line in enumerate(lines)
                         if line.startswith('"ID"') or line.startswith('ID,')), None)
    if header_index is None:
        die(f"no NCU header row in {csv_path} - not an ncu --csv log?")
    per_kernel = collections.defaultdict(lambda: collections.defaultdict(float))
    for record in csv.DictReader(lines[header_index:]):
        try:
            value = float(str(record.get('Metric Value', '0')).replace(',', ''))
        except ValueError:
            continue
        per_kernel[record.get('ID', '')][record.get('Metric Name', '')] += value
    return per_kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--engine', required=True, help='known-truth GEMM engine plan')
    parser.add_argument('--precision', required=True, choices=('int8', 'fp16', 'fp8'))
    parser.add_argument('--arch-flops', type=float, default=float(KNOWN_TRUTH_FLOPS),
                        help='true total ops of the engine, FMA=2 included '
                             '(default: the 4096^3 selftest GEMM, 137.44e9)')
    parser.add_argument('--extra', default='',
                        help='extra trtexec args (plugins, shapes) - shell-quoted string')
    parser.add_argument('--out', help='write calibration JSON here')
    args = parser.parse_args()

    engine = Path(args.engine).resolve()
    if not engine.is_file():
        die(f"engine not found: {engine}")
    ncu_bin, trtexec_bin = resolve_tools()

    # evidence files live next to --out when given, else in a kept temp dir
    workdir = Path(args.out).resolve().parent if args.out else Path(tempfile.mkdtemp(prefix='ops_per_tinst_'))
    workdir.mkdir(parents=True, exist_ok=True)
    csv_path = workdir / 'calibrate_ncu.csv'
    log_path = workdir / 'calibrate_trtexec.log'
    run_ncu_pass(engine, args.precision, args.extra, ncu_bin, trtexec_bin, csv_path, log_path)

    per_kernel = metrics_per_kernel(csv_path)
    kernels_total = len(per_kernel)
    # zero and 'n/a' tensor counts are reformat/elementwise helpers, not MMA work
    tensor_kernels = {kernel: metrics for kernel, metrics in per_kernel.items()
                      if metrics.get(TENSOR_INST_METRIC, 0) > 0}
    inst_total = sum(metrics[TENSOR_INST_METRIC] for metrics in tensor_kernels.values())
    time_us_total = sum(metrics.get(DURATION_METRIC, 0) for metrics in per_kernel.values()) / 1e3
    if inst_total <= 0:
        die(f"no tensor-pipe instructions recorded ({kernels_total} kernels in "
            f"{csv_path}) - the engine's chosen tactic does not run on tensor "
            f"cores at precision {args.precision}; rebuild the known-truth GEMM "
            "engine with the precision pinned and retry")

    raw = args.arch_flops / inst_total
    snapped = 2 ** round(math.log2(raw))
    snap_error = abs(raw - snapped) / snapped

    say(f"kernels: {kernels_total} total, {len(tensor_kernels)} with tensor insts")
    say(f"tensor insts: {inst_total:.0f}; true ops: {args.arch_flops:.6g}")
    say(f"ops_per_tinst raw = {raw:.2f} -> nearest power of two {snapped} ({100 * snap_error:.2f}% off)")

    result = {
        'schema': 'ops-per-tinst-cal/v1',
        'engine': str(engine),
        'precision': args.precision,
        'arch_flops': args.arch_flops,
        'tensor_inst_total': inst_total,
        'kernels_with_tensor_inst': len(tensor_kernels),
        'kernels_total': kernels_total,
        'gpu_time_us_total': round(time_us_total, 1),
        'ops_per_tinst_raw': round(raw, 3),
        'ops_per_tinst': snapped,
        'snap_error_pct': round(100 * snap_error, 2),
        'snap_tol_pct': 100 * SNAP_TOL,
        'ncu_csv': str(csv_path),
        'date': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    }
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1) + '\n')
        say(f"wrote {args.out}")

    if snap_error > SNAP_TOL:
        die(f"raw ops_per_tinst {raw:.2f} is {100 * snap_error:.1f}% from the nearest "
            f"power of two ({snapped}) - beyond the {100 * SNAP_TOL:.0f}% snap "
            "tolerance. The engine is mixing tensor tactics of more than one "
            "precision (TensorRT tactic choice) or is not the known-truth "
            "4096^3 GEMM; rebuild the selftest engine with the precision pinned "
            f"(--{args.precision} only) and recalibrate. Evidence: {csv_path}")

    say(f"CALIBRATED: ops_per_tinst[{args.precision}] = {snapped}")
    say("paste into the device config (device_configs/<device>.json), merging with any existing entries:")
    say(f'  "ops_per_tinst": {{ "{args.precision}": {snapped} }}')
    say("this tool never edits the config - the paste is a deliberate operator step")


if __name__ == '__main__':
    main()
