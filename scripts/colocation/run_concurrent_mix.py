#!/usr/bin/env python3
"""Co-run the rows of one resolved mix under one arm and leave one concurrent_mix.json.

  run_concurrent_mix.py --resolved <mix.json> --mode plain|mps|streams|mig --out <dir> --row-loop <bin>
        [--seconds 90] [--paced-solo <dir>] [--row-device row=UUID]... [--side-wait S] [--grid-s S]

  plain    one row_loop process per frame row (time-sliced contexts)
  mps      the same processes as MPS clients: the daemon is started here into a per-arm pipe dir,
           asserted alive, its server log kept (mps_verified); CUDA_MPS_ACTIVE_THREAD_PERCENTAGE per
           row from the mix's mps_pct column (side rows too)
  streams  one process, one context, one CUDA stream per frame row at the mix's prio (row_loop --multi)
  mig      one row_loop process per frame row, each pinned to its instance (--row-device row=UUID);
           the instances are created/destroyed by the arm script around this runner

Frame rows launch on one shared grid: every row writes ready_<row> after its warm-up and the start time
goes into row_start_ns.txt only when ALL rows are warm (+1 s). Side rows (role side) are launched
first, in their own process groups, through side_<runtime>.sh; the grid is armed only after every
side load reports ready (readiness regex per runtime). After the frame rows finish the harness waits
(bounded) for the finite side loads so they dump their own summaries, then tears the process groups
down. Every row ends with a status/rc/why; a row that cannot run is FAILED, never silently skipped.
Traces (trace_<row>.csv) feed period_makespan.py; the paced-solo p99 (--paced-solo) gives the
contention factor.
"""
import argparse
import json
import multiprocessing
import os
import re
import shutil
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# runtime -> (side harness script, log file the readiness regex is matched against, readiness regex)
SIDE_HARNESS = {'asr_driver': ('side_asr.sh', 'driver.log', r'^clip '),
                'edgellm': ('side_edgellm.sh', 'e2e.log', r'Runtime tensors successfully allocated'),
                'trtllm': ('side_trtllm.sh', 'side.log', r'^READY')}
ROW_LOOP_RC_WHY = {2: 'engine unreadable / input load failed',
                   3: 'the harness never armed the grid (start file timeout)',
                   4: 'no frames completed'}
# executor daemons a side load may leave outside its own process group
STRAY_DAEMON_PATTERNS = ('mpi4py.futures.server', 'orted --hnp', 'torch/_inductor/compile_worker')
MPS_CONTROL_PATTERN = '^nvidia-cuda-mps-control'


def failed(row, rc, why, extra=None):
    record = {'name': row['name'], 'driver': 'row_loop_cpp', 'status': 'FAILED', 'rc': rc, 'why': why,
              'target_hz': float(row['hz']), 'deadline_ms': float(row['deadline_ms']),
              'achieved_hz': 0.0, 'n': 0, 'p50_ms': None, 'p99_ms': None, 'max_ms': None,
              'miss_frac': 1.0, 'overrun_frac': 1.0, 'stream_priority': row.get('prio', 0)}
    record.update(extra or {})
    return record


def worker(row, row_loop, seconds, result_queue, device_uuid, mps_pct, timeout_s):
    """One row_loop process for one frame row; posts the row's JSON (or a FAILED record) to result_queue."""
    if device_uuid:
        os.environ['CUDA_VISIBLE_DEVICES'] = device_uuid
    if mps_pct:
        os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = str(mps_pct)
    out_json = os.path.join(os.environ['ROW_LOOP_OUT'], f"row_{row['name']}.json")
    stderr_path = out_json.replace('.json', '.stderr.txt')
    extra = {'mps_pct': mps_pct, 'device': device_uuid or 'default'}
    cmd = [row_loop, row['engine'], row['name'], f"{row['hz']:g}", f"{row['deadline_ms']:g}", str(seconds),
           out_json] + list(row.get('run_flags') or [])
    try:
        with open(stderr_path, 'w') as stderr_file:
            rc = subprocess.run(cmd, stderr=stderr_file, stdout=subprocess.DEVNULL,
                                timeout=timeout_s).returncode
    except subprocess.TimeoutExpired:
        why = f'row_loop exceeded {timeout_s}s (hung in deserialize/enqueue?)'
        result_queue.put(failed(row, -2, why, extra))
        return
    if rc == 0 and os.path.exists(out_json):
        try:
            record = json.load(open(out_json))
        except Exception as exc:
            result_queue.put(failed(row, rc, f'row JSON unreadable: {exc}', extra))
            return
        record.update(extra)
        result_queue.put(record)
        return
    why = ROW_LOOP_RC_WHY.get(rc, '')
    result_queue.put(failed(row, rc, f'row_loop rc={rc} {why}; see {stderr_path}'.strip(), extra))


def wait_ready(specs, timeout_s):
    """specs: (file, regex, proc). Block until every file matches, a side process exits, or timeout.
    -> {file: seconds to ready | None}"""
    t0 = time.time()
    pending = list(specs)
    ready_after = {}
    while pending and time.time() - t0 < timeout_s:
        for spec in list(pending):
            path, regex, proc = spec
            try:
                if re.search(regex, open(path, errors='ignore').read(), re.M):
                    ready_after[path] = round(time.time() - t0, 1)
                    pending.remove(spec)
                    continue
            except FileNotFoundError:
                pass
            if proc.poll() is not None:
                ready_after[path] = None
                pending.remove(spec)
        if pending:
            time.sleep(0.5)
    for path, _, _ in pending:
        ready_after[path] = None
    return ready_after


def mps_control_running():
    return subprocess.run(['pgrep', '-f', MPS_CONTROL_PATTERN], capture_output=True).returncode == 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--resolved', required=True)
    parser.add_argument('--mode', required=True, choices=['plain', 'mps', 'streams', 'mig'])
    parser.add_argument('--out', required=True)
    parser.add_argument('--row-loop', required=True)
    parser.add_argument('--seconds', type=float, default=90)
    parser.add_argument('--paced-solo', default='',
                        help='dir of paced_solo/<row>@<hz>.json (contention reference)')
    parser.add_argument('--row-device', action='append', default=[], help='row=UUID (mig)')
    parser.add_argument('--grid-s', type=float, default=float(os.environ.get('ROW_GRID_S', '60')),
                        help='max seconds to wait for every frame row to be warm before arming the grid')
    side_wait_help = ('seconds to wait for side loads after the rows (default seconds + 120: the finite side '
                      'batteries are sized to 1.3 x the run at the solo rate, so this tolerates a ~2.3x '
                      'slowdown under contention before the side load is terminated and recorded FAILED)')
    parser.add_argument('--side-wait', type=float, default=None, help=side_wait_help)
    parser.add_argument('--ready-timeout', type=float, default=600,
                        help='max seconds for a side load to report ready')
    return parser.parse_args()


class ArmRun:
    """State of one arm run: resolved mix, side-load processes, the MPS daemon and the output record."""

    def __init__(self, args):
        self.args = args
        self.out_dir = os.path.abspath(args.out)
        self.mix = json.load(open(args.resolved))
        if self.mix.get('errors'):
            sys.exit(f'{self.mix["mix"]}: resolved with errors: {self.mix["errors"]}')
        self.frames = [row for row in self.mix['rows'] if row['role'] == 'frame']
        self.sides = [row for row in self.mix['rows'] if row['role'] == 'side']
        self.row_device = dict(kv.split('=', 1) for kv in args.row_device if kv)
        if args.mode == 'mig' and any(row['name'] not in self.row_device for row in self.frames):
            sys.exit('mig: every frame row needs --row-device row=UUID')
        self.side_wait = args.side_wait if args.side_wait is not None else args.seconds + 120
        self.start_file = os.path.join(self.out_dir, 'row_start_ns.txt')
        self.row_timeout = args.grid_s + args.seconds + 120
        self.side_procs = []
        self.mps_started = False
        self.paced_solo = self._load_paced_solo()
        mps_pct = {row['name']: row.get('mps_pct') for row in self.mix['rows'] if row.get('mps_pct')}
        self.out = {'mix': self.mix['mix'], 'mode': args.mode,
                    'arm': {'name': args.mode, 'seconds': args.seconds, 'grid_s': args.grid_s,
                            'side_wait_s': self.side_wait, 'mps_pct': mps_pct,
                            'stream_priority': {row['name']: row.get('prio', 0) for row in self.frames},
                            'row_device': self.row_device},
                    'resolved_mix': os.path.abspath(args.resolved), 'seconds': args.seconds,
                    'side_rows': [row['name'] for row in self.sides], 'side_cmds': [], 'side_ready': {},
                    'rows': [], 'status': 'RUNNING'}

    def _load_paced_solo(self):
        solo = {}
        if not self.args.paced_solo:
            return solo
        for row in self.frames:
            path = os.path.join(self.args.paced_solo, f"{row['name']}@{row['hz']:g}.json")
            if not os.path.exists(path):
                continue
            try:
                solo[row['name']] = json.load(open(path))
            except Exception as exc:
                print(f'[warn] paced solo {path}: {exc}', file=sys.stderr)
        return solo

    def dump(self):
        json.dump(self.out, open(os.path.join(self.out_dir, 'concurrent_mix.json'), 'w'), indent=1)

    def fail(self, status, rc):
        self.out['status'] = status
        self.kill_side(signal.SIGKILL)
        self.mps_quit()
        self.dump()
        sys.exit(rc)

    # ---- MPS daemon
    def mps_start(self):
        # the pipe dir holds a Unix socket (path limit 108 chars): short, in /tmp; logs live in the arm dir
        pipe_dir = f'/tmp/nvmps_{self.args.mode}_{os.getpid()}'
        log_dir = os.path.join(self.out_dir, 'mps_log')
        os.makedirs(pipe_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.environ['CUDA_MPS_PIPE_DIRECTORY'] = pipe_dir
        os.environ['CUDA_MPS_LOG_DIRECTORY'] = log_dir
        if mps_control_running():
            self.out['status'] = ('FAILED: an MPS control daemon is already running (a previous arm did not '
                                  'quit it: echo quit | nvidia-cuda-mps-control)')
            self.dump()
            sys.exit(3)
        start_rc = subprocess.run(['nvidia-cuda-mps-control', '-d']).returncode
        time.sleep(1.5)
        self.mps_started = True
        alive = mps_control_running() and os.path.exists(os.path.join(pipe_dir, 'control'))
        self.out['arm']['mps_daemon'] = {'start_rc': start_rc, 'control_alive': alive, 'pipe_dir': pipe_dir,
                                         'log_dir': log_dir}
        if not alive:
            self.out['status'] = 'FAILED: MPS control daemon did not start'
            self.dump()
            self.mps_quit()
            sys.exit(3)

    def mps_quit(self):
        if not self.mps_started:
            return
        try:
            subprocess.run('echo quit | nvidia-cuda-mps-control', shell=True, timeout=30)
        except Exception:
            pass
        time.sleep(1)
        if mps_control_running():
            subprocess.run(['pkill', '-f', '^nvidia-cuda-mps-server'])
            subprocess.run(['pkill', '-f', MPS_CONTROL_PATTERN])
        self.mps_started = False
        shutil.rmtree(os.environ.get('CUDA_MPS_PIPE_DIRECTORY', '/nonexistent'), ignore_errors=True)

    def mps_verify(self):
        server_log = os.path.join(self.out['arm']['mps_daemon']['log_dir'], 'server.log')
        text = open(server_log, errors='ignore').read() if os.path.exists(server_log) else ''
        clients_seen = 'connect' in text.lower() or 'client' in text.lower()
        self.out['arm']['mps_verified'] = bool(text.strip()) and clients_seen
        self.out['arm']['mps_server_log_lines'] = len(text.splitlines())

    # ---- side loads
    def kill_side(self, sig=signal.SIGTERM):
        for proc in self.side_procs:
            try:
                os.killpg(os.getpgid(proc.pid), sig)
            except Exception:
                pass

    def _side_env(self, row):
        env = dict(os.environ)
        if row.get('mps_pct') and self.args.mode == 'mps':
            env['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = str(row['mps_pct'])
        if self.args.paced_solo:   # the measured solo request time sizes the side battery
            env['PACED_SOLO_SIDE'] = os.path.join(self.args.paced_solo, f"side_{row['name']}")
        # lets the side harness window its statistics to the frame run
        env['ROW_START_FILE'] = self.start_file
        return env

    def launch_side_loads(self):
        """Start every side row in its own process group, then block until each reports ready."""
        specs = []
        for row in self.sides:
            runtime = row.get('runtime')
            harness = SIDE_HARNESS.get(runtime)
            if not harness:
                self.out['rows'].append(failed(row, -1, f'no side-load harness for runtime {runtime!r}'))
                continue
            script, log_name, ready_regex = harness
            side_dir = os.path.join(self.out_dir, 'side', row['name'])
            os.makedirs(side_dir, exist_ok=True)
            cmd = [os.path.join(HERE, script), os.path.abspath(self.args.resolved), row['name'], side_dir,
                   f'{self.args.seconds:g}']
            self.out['side_cmds'].append(' '.join(cmd))
            proc = subprocess.Popen(cmd, env=self._side_env(row), start_new_session=True,
                                    stdout=open(os.path.join(side_dir, 'side.log'), 'w'),
                                    stderr=subprocess.STDOUT)
            self.side_procs.append(proc)
            specs.append((os.path.join(side_dir, log_name), ready_regex, proc))
        if not specs:
            return
        self.out['side_ready'] = wait_ready(specs, self.args.ready_timeout)
        if any(value is None for value in self.out['side_ready'].values()):
            self.fail('FAILED: side load never became ready ' + json.dumps(self.out['side_ready']), 4)

    def reap_side_loads(self):
        """Let the finite side loads finish and dump their summaries, then tear down what remains."""
        t0 = time.time()
        for proc in self.side_procs:
            try:
                proc.wait(timeout=max(0.0, self.side_wait - (time.time() - t0)))
            except Exception:
                pass
        self.kill_side(signal.SIGTERM)
        for _ in range(50):
            if all(proc.poll() is not None for proc in self.side_procs):
                break
            time.sleep(0.5)
        self.kill_side(signal.SIGKILL)
        for pattern in STRAY_DAEMON_PATTERNS:
            subprocess.run(['pkill', '-KILL', '-f', pattern], capture_output=True)
        self.out['side_rc'] = [proc.poll() for proc in self.side_procs]
        self.out['side_wait_used_s'] = round(time.time() - t0, 1)
        for row in self.sides:
            self.out['rows'].append(self._side_record(row))

    def _side_record(self, row):
        result_path = os.path.join(self.out_dir, 'side', row['name'], 'side_result.json')
        record = {'name': row['name'], 'role': 'side', 'runtime': row.get('runtime'), 'status': 'FAILED',
                  'why': f'no side_result.json under side/{row["name"]}',
                  'mps_pct': row.get('mps_pct') if self.args.mode == 'mps' else None}
        if not os.path.exists(result_path):
            return record
        try:
            record.update(json.load(open(result_path)))
            record.setdefault('status', 'OK')
        except Exception as exc:
            record['why'] = f'side_result.json unreadable: {exc}'
        return record

    # ---- frame rows
    def _ready_file(self, row):
        return os.path.join(self.out_dir, f"ready_{row['name']}")

    def clear_grid_files(self):
        for path in [self.start_file] + [self._ready_file(row) for row in self.frames]:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    def run_frames_streams(self):
        """One process, one context, one stream per row at its priority; row_loop arms the shared start."""
        prio_range = subprocess.run([self.args.row_loop, '--prio-range'], capture_output=True, text=True)
        lo, hi = prio_range.stdout.split()[:2]
        self.out['arm']['stream_priority_range'] = [int(lo), int(hi)]
        out_json = os.path.join(self.out_dir, 'multi.json')
        row_specs = [f"{row['name']}|{row['engine']}|{row['hz']:g}|{row['deadline_ms']:g}|"
                     f"{row.get('prio', 0)}|{' '.join(row.get('run_flags') or [])}" for row in self.frames]
        cmd = [self.args.row_loop, '--multi', str(self.args.seconds), out_json] + row_specs
        try:
            with open(os.path.join(self.out_dir, 'multi.stderr.txt'), 'w') as stderr_file:
                rc = subprocess.run(cmd, stderr=stderr_file, stdout=subprocess.DEVNULL,
                                    timeout=self.row_timeout).returncode
        except subprocess.TimeoutExpired:
            rc = -2
        if rc != 0 or not os.path.exists(out_json):
            return [failed(row, rc, f'row_loop --multi rc={rc}; see {self.out_dir}/multi.stderr.txt')
                    for row in self.frames]
        multi = json.load(open(out_json))
        self.out['row_start_ns'] = multi.get('row_start_ns')
        results = multi.get('rows', [])
        for record in results:
            record.setdefault('device', 'default')
            record['mps_pct'] = None
        return results

    def _arm_grid(self, procs, t0):
        """Write the shared start time once every row is warm (or the grid wait expires)."""
        args = self.args
        while (time.time() - t0 < args.grid_s
               and not all(os.path.exists(self._ready_file(row)) for row in self.frames)
               and any(proc.is_alive() for proc in procs)):
            time.sleep(0.1)
        ready = [row['name'] for row in self.frames if os.path.exists(self._ready_file(row))]
        grid_ns = time.monotonic_ns() + int(1e9)
        open(self.start_file, 'w').write(f'{grid_ns}\n')
        self.out['row_start_ns'] = grid_ns
        self.out['rows_ready_s'] = round(time.time() - t0, 1)
        self.out['rows_ready'] = ready
        # a finite side load that already exited never overlapped the frames: the cell measures nothing
        # (the verdict marks it invalid)
        self.out['side_exited_before_start'] = [row['name'] for row, proc in zip(self.sides, self.side_procs)
                                               if proc.poll() is not None]
        if len(ready) < len(self.frames):
            print(f'[warn] grid armed with only {len(ready)}/{len(self.frames)} rows warm after '
                  f'{args.grid_s}s: {ready}', file=sys.stderr)

    def run_frames_processes(self):
        """One row_loop process per row; this harness arms the shared grid once every row is warm."""
        args = self.args
        os.environ['ROW_READY_DIR'] = self.out_dir
        os.environ['ROW_START_FILE'] = self.start_file
        os.environ['ROW_START_TIMEOUT_S'] = str(int(args.grid_s + 30))
        mps_pct = {row['name']: row.get('mps_pct') for row in self.frames} if args.mode == 'mps' else {}
        context = multiprocessing.get_context('spawn')
        result_queue = context.Queue()
        procs = [context.Process(target=worker,
                                 args=(row, args.row_loop, args.seconds, result_queue,
                                       self.row_device.get(row['name']), mps_pct.get(row['name']),
                                       self.row_timeout))
                 for row in self.frames]
        for proc in procs:
            proc.start()
        self._arm_grid(procs, time.time())
        results = []
        for _ in procs:
            try:
                results.append(result_queue.get(timeout=self.row_timeout + 60))
            except Exception:
                break
        for proc in procs:
            proc.join(timeout=30)
        for proc in procs:
            if proc.is_alive():
                proc.kill()
        posted = {record['name'] for record in results}
        results += [failed(row, -3, 'no result posted (worker died)')
                    for row in self.frames if row['name'] not in posted]
        return results

    # ---- assembly
    def finish(self, frame_results):
        order = {row['name']: index for index, row in enumerate(self.frames)}
        frame_results.sort(key=lambda record: order.get(record['name'], 99))
        for record in frame_results:
            solo_p99 = (self.paced_solo.get(record['name']) or {}).get('p99_ms')
            record['role'] = 'frame'
            record['paced_solo_p99_ms'] = solo_p99
            contended = solo_p99 and record.get('p99_ms')
            record['contention_factor'] = (record['p99_ms'] / solo_p99) if contended else None
        self.out['rows'] = frame_results + self.out['rows']
        self.out['sum_paced_solo_p99_ms'] = sum((record.get('paced_solo_p99_ms') or 0)
                                                for record in frame_results)
        self.out['sum_target_time_share'] = sum(
            record['target_hz'] * (record.get('paced_solo_p99_ms') or 0) / 1000 for record in frame_results)
        bad = [record['name'] for record in self.out['rows'] if record.get('status') == 'FAILED']
        self.out['status'] = 'OK' if not bad else 'FAILED: ' + ', '.join(bad)
        self.dump()
        self.print_summary(frame_results)
        sys.exit(0 if self.out['status'] == 'OK' else 5)

    def print_summary(self, frame_results):
        print(f"[{self.args.mode}] {self.mix['mix']}: status {self.out['status']}")
        print(f"{'row':28s} {'prio':>4s} {'target Hz':>9s} {'achieved':>9s} {'p50':>7s} {'p99':>7s} "
              f"{'paced solo':>10s} {'x':>5s} {'miss%':>6s} {'overrun%':>8s}")
        for record in frame_results:
            print(f"{record['name']:28s} {record.get('stream_priority', 0):4d} {record['target_hz']:9.1f} "
                  f"{record['achieved_hz']:9.1f} {record['p50_ms'] or float('nan'):7.2f} "
                  f"{record['p99_ms'] or float('nan'):7.2f} {record['paced_solo_p99_ms'] or 0:10.2f} "
                  f"{record['contention_factor'] or 0:5.2f} {100 * record['miss_frac']:6.1f} "
                  f"{100 * record['overrun_frac']:8.1f}")
        for record in self.out['rows']:
            if record.get('role') == 'side':
                print(f"{record['name']:28s} side {record.get('runtime')}: {record.get('status')} "
                      f"{record.get('summary', record.get('why', ''))}")


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    args.out = os.path.abspath(args.out)
    os.environ['ROW_LOOP_OUT'] = args.out
    os.environ['ROW_TRACE_DIR'] = args.out
    run = ArmRun(args)

    def on_term(signum, _frame):
        run.out['status'] = f'FAILED: signal {signum} before completion'
        run.kill_side(signal.SIGKILL)
        run.mps_quit()
        run.dump()
        os._exit(143)

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    if args.mode == 'mps':
        run.mps_start()
    run.launch_side_loads()
    run.clear_grid_files()
    if args.mode == 'streams':
        frame_results = run.run_frames_streams()
    else:
        frame_results = run.run_frames_processes()
    run.reap_side_loads()
    if args.mode == 'mps':
        run.mps_verify()
        run.mps_quit()
    run.finish(frame_results)


if __name__ == '__main__':
    main()
