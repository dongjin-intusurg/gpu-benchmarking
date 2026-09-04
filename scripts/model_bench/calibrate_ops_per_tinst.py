#!/usr/bin/env python3
"""Calibrate ops-per-tensor-instruction against the known-truth GEMM engine.

Usage:
  calibrate_ops_per_tinst.py --engine <known_truth_gemm.engine>
                             --precision int8|fp16|fp8
                             [--arch-flops 137.44e9] [--extra "..."]
                             [--out <calibration.json>]

Why: the roofline classifier converts sm__inst_executed_pipe_tensor counts to
ops via a per-arch, per-precision constant (a compute-capability-11.0 Jetson, sm_110: 2048 int8 / 1024 fp16
per warp-instruction). The constant does NOT transfer across architectures -
sm_120 must be measured on the box. Method: run one NCU counter pass over an
engine whose true op count is known exactly (the selftest 4096^3 MatMul,
2 x 4096^3 = 137.44 GFLOPs with FMA=2 already counted), divide truth by the
measured instruction total, and demand the result land on a power of two.

Pre-declared policy (declared here, not tuned post-hoc):
  - arch-flops default = 2 * 4096^3 = 137,438,953,472 ops. This is TOTAL ops
    with FMA=2 already inside, so ops_per_tinst = arch_flops / sum(tensor insts)
    - no extra factor of 2 anywhere.
  - only kernels whose tensor-instruction count is > 0 enter the sum: zero and
    'n/a' rows are reformat/elementwise helpers, not MMA work.
  - snap tolerance 10%: the raw quotient must sit within 10% of the nearest
    power of two (MMA shapes make the true constant an exact power of two). A
    miss means the engine is NOT a clean single-precision tactic - TensorRT
    quietly picked mixed-precision tactics, or the engine is not the known-truth
    GEMM - and the run FAILS loudly rather than shipping a blended constant.
  - NCU manages its own clocks during the pass; only COUNTS are read here
    (instruction totals), never times - no clock lock is required.
  - this tool NEVER edits the device config: it prints the exact line to paste,
    and the paste stays a deliberate, reviewed operator action.

Exit codes: 0 calibrated; 1 environment/run/parse failure or snap-tolerance FAIL.
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

SNAP_TOL = 0.10                       # max relative distance to a power of two
KNOWN_TRUTH_FLOPS = 2 * 4096 ** 3     # selftest GEMM, FMA=2 included

NCU_METRICS = 'sm__inst_executed_pipe_tensor.sum,gpu__time_duration.sum'


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def die(msg, code=1):
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(code)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--engine', required=True, help='known-truth GEMM engine plan')
    ap.add_argument('--precision', required=True, choices=('int8', 'fp16', 'fp8'))
    ap.add_argument('--arch-flops', type=float, default=float(KNOWN_TRUTH_FLOPS),
                    help='true total ops of the engine, FMA=2 included '
                         '(default: the 4096^3 selftest GEMM, 137.44e9)')
    ap.add_argument('--extra', default='',
                    help='extra trtexec args (plugins, shapes) - shell-quoted string')
    ap.add_argument('--out', help='write calibration JSON here')
    args = ap.parse_args()

    engine = Path(args.engine).resolve()
    if not engine.is_file():
        die(f"engine not found: {engine}")

    # trtexec lives off-PATH on Jetson - the standard idiom appends the TRT bin dir
    os.environ['PATH'] = os.environ.get('PATH', '') + ':/usr/src/tensorrt/bin'
    ncu_bin = shutil.which('ncu')
    trtexec_bin = shutil.which('trtexec')
    if not ncu_bin:
        die("ncu not on PATH - install Nsight Compute (counter access also needs "
            "root or NVreg_RestrictProfilingToAdminUsers=0)")
    if not trtexec_bin:
        die("trtexec not found on PATH or /usr/src/tensorrt/bin")

    # evidence files live next to --out when given, else in a kept temp dir
    workdir = Path(args.out).resolve().parent if args.out \
        else Path(tempfile.mkdtemp(prefix='ops_per_tinst_'))
    workdir.mkdir(parents=True, exist_ok=True)
    csv_path = workdir / 'calibrate_ncu.csv'
    log_path = workdir / 'calibrate_trtexec.log'

    # sudo resets PATH (secure_path) - both binaries are pre-resolved absolute.
    # sudo also resets the environment - LD_LIBRARY_PATH must ride along or the
    # profiled trtexec can't load libnvinfer* from non-ldconfig TRT installs.
    cmd = ['sudo', f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}",
           ncu_bin, '--csv', '--target-processes', 'all', '-f',
           '--metrics', NCU_METRICS, '--log-file', str(csv_path),
           trtexec_bin, f'--loadEngine={engine}',
           # exactly ONE inference: without --duration=0, trtexec's default
           # 3-second duration floor keeps enqueueing under ncu (~a dozen
           # inferences fit even at replay speed), every launch lands in the
           # instruction sum, and the quotient shrinks by that factor —
           # observed 80.31 on sm_120 = 1024 / ~12.75 inferences
           '--iterations=1', '--duration=0', '--warmUp=0'] + shlex.split(args.extra)

    say(f"ncu counter pass over {engine.name} ({args.precision})...")
    with open(log_path, 'w') as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
    if rc != 0 or not csv_path.is_file():
        tail = ''
        if log_path.is_file():
            tail = ''.join(log_path.read_text(errors='ignore').splitlines(True)[-15:])
        die(f"ncu pass failed (rc={rc}); log tail from {log_path}:\n{tail}\n"
            "hints: counter access needs root (sudo) or the "
            "NVreg_RestrictProfilingToAdminUsers=0 modprobe option; check that "
            "LD_LIBRARY_PATH covers the TRT libs")

    # -- parse: skip ==PROF== preamble, sum tensor insts over count>0 kernels --
    lines = csv_path.read_text(errors='ignore').splitlines(True)
    start = next((i for i, l in enumerate(lines)
                  if l.startswith('"ID"') or l.startswith('ID,')), None)
    if start is None:
        die(f"no NCU header row in {csv_path} - not an ncu --csv log?")
    per_kernel = collections.defaultdict(lambda: collections.defaultdict(float))
    for r in csv.DictReader(lines[start:]):
        try:
            v = float(str(r.get('Metric Value', '0')).replace(',', ''))
        except ValueError:
            continue  # 'n/a' rows drop out, never zero-fill
        per_kernel[r.get('ID', '')][r.get('Metric Name', '')] += v

    kernels_total = len(per_kernel)
    tensor_kernels = {k: m for k, m in per_kernel.items()
                      if m.get('sm__inst_executed_pipe_tensor.sum', 0) > 0}
    inst_total = sum(m['sm__inst_executed_pipe_tensor.sum']
                     for m in tensor_kernels.values())
    time_us_total = sum(m.get('gpu__time_duration.sum', 0)
                        for m in per_kernel.values()) / 1e3
    if inst_total <= 0:
        die(f"no tensor-pipe instructions recorded ({kernels_total} kernels in "
            f"{csv_path}) - the engine's chosen tactic does not run on tensor "
            f"cores at precision {args.precision}; rebuild the known-truth GEMM "
            "engine with the precision pinned and retry")

    raw = args.arch_flops / inst_total
    snapped = 2 ** round(math.log2(raw))
    err = abs(raw - snapped) / snapped

    say(f"kernels: {kernels_total} total, {len(tensor_kernels)} with tensor insts")
    say(f"tensor insts: {inst_total:.0f}; true ops: {args.arch_flops:.6g}")
    say(f"ops_per_tinst raw = {raw:.2f} -> nearest power of two {snapped} "
        f"({100 * err:.2f}% off)")

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
        'snap_error_pct': round(100 * err, 2),
        'snap_tol_pct': 100 * SNAP_TOL,
        'ncu_csv': str(csv_path),
        'date': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    }
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1) + '\n')
        say(f"wrote {args.out}")

    if err > SNAP_TOL:
        die(f"raw ops_per_tinst {raw:.2f} is {100 * err:.1f}% from the nearest "
            f"power of two ({snapped}) - beyond the {100 * SNAP_TOL:.0f}% snap "
            "tolerance. The engine is mixing tensor tactics of more than one "
            "precision (TensorRT tactic choice) or is not the known-truth "
            "4096^3 GEMM; rebuild the selftest engine with the precision pinned "
            f"(--{args.precision} only) and recalibrate. Evidence: {csv_path}")

    say(f"CALIBRATED: ops_per_tinst[{args.precision}] = {snapped}")
    say("paste into the device config (device_configs/<device>.json), merging "
        "with any existing entries:")
    say(f'  "ops_per_tinst": {{ "{args.precision}": {snapped} }}')
    say("this tool never edits the config - the paste is a deliberate operator step")


if __name__ == '__main__':
    main()
