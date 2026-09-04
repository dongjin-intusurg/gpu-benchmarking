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
import argparse, json, multiprocessing as mp, os, re, shutil, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SIDE = {'asr_driver': ('side_asr.sh', 'driver.log', r'^clip '),
        'edgellm': ('side_edgellm.sh', 'e2e.log', r'Runtime tensors successfully allocated'),
        'trtllm': ('side_trtllm.sh', 'side.log', r'^READY')}


def failed(row, rc, why, extra=None):
    d = {'name': row['name'], 'driver': 'row_loop_cpp', 'status': 'FAILED', 'rc': rc, 'why': why, 'target_hz': float(row['hz']),
         'deadline_ms': float(row['deadline_ms']), 'achieved_hz': 0.0, 'n': 0, 'p50_ms': None, 'p99_ms': None, 'max_ms': None,
         'miss_frac': 1.0, 'overrun_frac': 1.0, 'stream_priority': row.get('prio', 0)}
    d.update(extra or {}); return d


def worker(row, row_loop, seconds, outq, device_uuid, mps_pct, timeout_s):
    if device_uuid: os.environ['CUDA_VISIBLE_DEVICES'] = device_uuid
    if mps_pct: os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = str(mps_pct)
    out = os.environ['ROW_LOOP_OUT']; outj = os.path.join(out, f"row_{row['name']}.json"); errp = outj.replace('.json', '.stderr.txt')
    extra = {'mps_pct': mps_pct, 'device': device_uuid or 'default'}
    cmd = [row_loop, row['engine'], row['name'], f"{row['hz']:g}", f"{row['deadline_ms']:g}", str(seconds), outj] + list(row.get('run_flags') or [])
    try:
        with open(errp, 'w') as ef: rc = subprocess.run(cmd, stderr=ef, stdout=subprocess.DEVNULL, timeout=timeout_s).returncode
    except subprocess.TimeoutExpired:
        outq.put(failed(row, -2, f'row_loop exceeded {timeout_s}s (hung in deserialize/enqueue?)', extra)); return
    if rc == 0 and os.path.exists(outj):
        try:
            r = json.load(open(outj)); r.update(extra); outq.put(r); return
        except Exception as ex:
            outq.put(failed(row, rc, f'row JSON unreadable: {ex}', extra)); return
    why = {2: 'engine unreadable / input load failed', 3: 'the harness never armed the grid (start file timeout)', 4: 'no frames completed'}.get(rc, '')
    outq.put(failed(row, rc, f'row_loop rc={rc} {why}; see {errp}'.strip(), extra))


def wait_ready(specs, timeout_s, procs):
    """specs: (file, regex, proc). Block until every file matches, a side process exits, or timeout. -> {file: seconds|None}"""
    t0 = time.time(); left = list(specs); got = {}
    while left and time.time() - t0 < timeout_s:
        for f, rx, p in list(left):
            try:
                if re.search(rx, open(f, errors='ignore').read(), re.M): got[f] = round(time.time() - t0, 1); left.remove((f, rx, p)); continue
            except FileNotFoundError:
                pass
            if p.poll() is not None: got[f] = None; left.remove((f, rx, p))
        if left: time.sleep(0.5)
    for f, _, _ in left: got[f] = None
    return got


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--resolved', required=True); ap.add_argument('--mode', required=True, choices=['plain', 'mps', 'streams', 'mig'])
    ap.add_argument('--out', required=True); ap.add_argument('--row-loop', required=True); ap.add_argument('--seconds', type=float, default=90)
    ap.add_argument('--paced-solo', default='', help='dir of paced_solo/<row>@<hz>.json (contention reference)')
    ap.add_argument('--row-device', action='append', default=[], help='row=UUID (mig)')
    ap.add_argument('--grid-s', type=float, default=float(os.environ.get('ROW_GRID_S', '60')), help='max seconds to wait for every frame row to be warm before arming the grid')
    ap.add_argument('--side-wait', type=float, default=None, help='seconds to wait for side loads after the rows (default seconds + 120: the finite side batteries are sized to 1.3 x the run at the solo rate, so this tolerates a ~2.3x slowdown under contention before the side load is terminated and recorded FAILED)')
    ap.add_argument('--ready-timeout', type=float, default=600, help='max seconds for a side load to report ready')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); a.out = os.path.abspath(a.out); os.environ['ROW_LOOP_OUT'] = a.out; os.environ['ROW_TRACE_DIR'] = a.out
    mix = json.load(open(a.resolved))
    if mix.get('errors'): sys.exit(f'{mix["mix"]}: resolved with errors: {mix["errors"]}')
    frames = [r for r in mix['rows'] if r['role'] == 'frame']; sides = [r for r in mix['rows'] if r['role'] == 'side']
    dev = dict(kv.split('=', 1) for kv in a.row_device if kv)
    if a.mode == 'mig' and any(r['name'] not in dev for r in frames): sys.exit('mig: every frame row needs --row-device row=UUID')
    side_wait = a.side_wait if a.side_wait is not None else a.seconds + 120
    solo = {}
    for r in frames:
        f = os.path.join(a.paced_solo, f"{r['name']}@{r['hz']:g}.json") if a.paced_solo else ''
        if f and os.path.exists(f):
            try: solo[r['name']] = json.load(open(f))
            except Exception as ex: print(f'[warn] paced solo {f}: {ex}', file=sys.stderr)
    state = {'side': [], 'mps': False}
    out = {'mix': mix['mix'], 'mode': a.mode, 'arm': {'name': a.mode, 'seconds': a.seconds, 'grid_s': a.grid_s, 'side_wait_s': side_wait,
                                                      'mps_pct': {r['name']: r.get('mps_pct') for r in mix['rows'] if r.get('mps_pct')},
                                                      'stream_priority': {r['name']: r.get('prio', 0) for r in frames}, 'row_device': dev},
           'resolved_mix': os.path.abspath(a.resolved), 'seconds': a.seconds, 'side_rows': [r['name'] for r in sides], 'side_cmds': [], 'side_ready': {},
           'rows': [], 'status': 'RUNNING'}
    def dump(): json.dump(out, open(os.path.join(a.out, 'concurrent_mix.json'), 'w'), indent=1)
    def mps_quit():
        if not state['mps']: return
        try: subprocess.run('echo quit | nvidia-cuda-mps-control', shell=True, timeout=30)
        except Exception: pass
        time.sleep(1)
        if subprocess.run(['pgrep', '-f', '^nvidia-cuda-mps-control'], capture_output=True).returncode == 0:
            subprocess.run(['pkill', '-f', '^nvidia-cuda-mps-server']); subprocess.run(['pkill', '-f', '^nvidia-cuda-mps-control'])
        state['mps'] = False; shutil.rmtree(os.environ.get('CUDA_MPS_PIPE_DIRECTORY', '/nonexistent'), ignore_errors=True)
    def kill_side(sig=signal.SIGTERM):
        for s in state['side']:
            try: os.killpg(os.getpgid(s.pid), sig)
            except Exception: pass
    def on_term(signum, frame):
        out['status'] = f'FAILED: signal {signum} before completion'; kill_side(signal.SIGKILL); mps_quit(); dump(); os._exit(143)
    signal.signal(signal.SIGTERM, on_term); signal.signal(signal.SIGINT, on_term)

    if a.mode == 'mps':
        # the pipe dir holds a Unix socket (path limit 108 chars): short, in /tmp; logs live in the arm dir
        pipe = f'/tmp/nvmps_{a.mode}_{os.getpid()}'; logd = os.path.join(a.out, 'mps_log'); os.makedirs(pipe, exist_ok=True); os.makedirs(logd, exist_ok=True)
        os.environ['CUDA_MPS_PIPE_DIRECTORY'] = pipe; os.environ['CUDA_MPS_LOG_DIRECTORY'] = logd
        if subprocess.run(['pgrep', '-f', '^nvidia-cuda-mps-control'], capture_output=True).returncode == 0:
            out['status'] = 'FAILED: an MPS control daemon is already running (a previous arm did not quit it: echo quit | nvidia-cuda-mps-control)'; dump(); sys.exit(3)
        rc = subprocess.run(['nvidia-cuda-mps-control', '-d']).returncode; time.sleep(1.5); state['mps'] = True
        up = subprocess.run(['pgrep', '-f', '^nvidia-cuda-mps-control'], capture_output=True).returncode == 0 and os.path.exists(os.path.join(pipe, 'control'))
        out['arm']['mps_daemon'] = {'start_rc': rc, 'control_alive': up, 'pipe_dir': pipe, 'log_dir': logd}
        if not up: out['status'] = 'FAILED: MPS control daemon did not start'; dump(); mps_quit(); sys.exit(3)

    # side loads first (their own process groups), then wait for every one to report ready
    specs = []
    for r in sides:
        rt = r.get('runtime'); spec = SIDE.get(rt)
        if not spec:
            out['rows'].append(failed(r, -1, f'no side-load harness for runtime {rt!r}')); continue
        script, logname, rx = spec; sd = os.path.join(a.out, 'side', r['name']); os.makedirs(sd, exist_ok=True)
        env = dict(os.environ)
        if r.get('mps_pct') and a.mode == 'mps': env['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = str(r['mps_pct'])
        if a.paced_solo: env['PACED_SOLO_SIDE'] = os.path.join(a.paced_solo, f"side_{r['name']}")   # measured solo request time sizes the side battery
        env['ROW_START_FILE'] = os.path.join(a.out, 'row_start_ns.txt')   # lets the side harness window its statistics to the frame run
        cmd = [os.path.join(HERE, script), os.path.abspath(a.resolved), r['name'], sd, f'{a.seconds:g}']
        out['side_cmds'].append(' '.join(cmd))
        p = subprocess.Popen(cmd, env=env, start_new_session=True, stdout=open(os.path.join(sd, 'side.log'), 'w'), stderr=subprocess.STDOUT)
        state['side'].append(p); specs.append((os.path.join(sd, logname), rx, p))
    if specs:
        out['side_ready'] = wait_ready(specs, a.ready_timeout, state['side'])
        if any(v is None for v in out['side_ready'].values()):
            out['status'] = 'FAILED: side load never became ready ' + json.dumps(out['side_ready']); kill_side(signal.SIGKILL); mps_quit(); dump(); sys.exit(4)

    sf = os.path.join(a.out, 'row_start_ns.txt')
    for f in [sf] + [os.path.join(a.out, f"ready_{r['name']}") for r in frames]:
        try: os.remove(f)
        except FileNotFoundError: pass
    tmo = a.grid_s + a.seconds + 120; res = []
    if a.mode == 'streams':
        # one process, one context, one stream per row at its priority; row_loop arms its own shared start
        lo, hi = subprocess.run([a.row_loop, '--prio-range'], capture_output=True, text=True).stdout.split()[:2]
        out['arm']['stream_priority_range'] = [int(lo), int(hi)]
        outj = os.path.join(a.out, 'multi.json')
        cmd = [a.row_loop, '--multi', str(a.seconds), outj] + [f"{r['name']}|{r['engine']}|{r['hz']:g}|{r['deadline_ms']:g}|{r.get('prio', 0)}|{' '.join(r.get('run_flags') or [])}" for r in frames]
        try:
            with open(os.path.join(a.out, 'multi.stderr.txt'), 'w') as ef: rc = subprocess.run(cmd, stderr=ef, stdout=subprocess.DEVNULL, timeout=tmo).returncode
        except subprocess.TimeoutExpired:
            rc = -2
        if rc == 0 and os.path.exists(outj):
            m = json.load(open(outj)); out['row_start_ns'] = m.get('row_start_ns'); res = m.get('rows', [])
            for r in res: r.setdefault('device', 'default'); r['mps_pct'] = None
        else:
            res = [failed(r, rc, f'row_loop --multi rc={rc}; see {a.out}/multi.stderr.txt') for r in frames]
    else:
        os.environ['ROW_READY_DIR'] = a.out; os.environ['ROW_START_FILE'] = sf; os.environ['ROW_START_TIMEOUT_S'] = str(int(a.grid_s + 30))
        pct = {r['name']: r.get('mps_pct') for r in frames} if a.mode == 'mps' else {}
        ctx = mp.get_context('spawn'); q = ctx.Queue()
        procs = [ctx.Process(target=worker, args=(r, a.row_loop, a.seconds, q, dev.get(r['name']), pct.get(r['name']), tmo)) for r in frames]
        for p in procs: p.start()
        t0 = time.time()
        while time.time() - t0 < a.grid_s and not all(os.path.exists(os.path.join(a.out, f"ready_{r['name']}")) for r in frames) and any(p.is_alive() for p in procs): time.sleep(0.1)
        ready = [r['name'] for r in frames if os.path.exists(os.path.join(a.out, f"ready_{r['name']}"))]
        grid_ns = time.monotonic_ns() + int(1e9); open(sf, 'w').write(f'{grid_ns}\n')
        out['row_start_ns'] = grid_ns; out['rows_ready_s'] = round(time.time() - t0, 1); out['rows_ready'] = ready
        # a finite side load that has already exited never overlapped the frames: the cell measures nothing (verdict: invalid)
        out['side_exited_before_start'] = [r['name'] for r, p in zip(sides, state['side']) if p.poll() is not None]
        if len(ready) < len(frames): print(f'[warn] grid armed with only {len(ready)}/{len(frames)} rows warm after {a.grid_s}s: {ready}', file=sys.stderr)
        for _ in procs:
            try: res.append(q.get(timeout=tmo + 60))
            except Exception: break
        for p in procs: p.join(timeout=30)
        for p in procs:
            if p.is_alive(): p.kill()
        got = {r['name'] for r in res}
        res += [failed(r, -3, 'no result posted (worker died)') for r in frames if r['name'] not in got]
    # let the finite side loads finish and dump their summaries, then tear down what remains (process groups)
    t0 = time.time()
    for s in state['side']:
        try: s.wait(timeout=max(0.0, side_wait - (time.time() - t0)))
        except Exception: pass
    kill_side(signal.SIGTERM)
    for _ in range(50):
        if all(s.poll() is not None for s in state['side']): break
        time.sleep(0.5)
    kill_side(signal.SIGKILL)
    for pat in ('mpi4py.futures.server', 'orted --hnp', 'torch/_inductor/compile_worker'):   # executor daemons outside the side load's group
        subprocess.run(['pkill', '-KILL', '-f', pat], capture_output=True)
    out['side_rc'] = [s.poll() for s in state['side']]; out['side_wait_used_s'] = round(time.time() - t0, 1)
    for r in sides:
        sj = os.path.join(a.out, 'side', r['name'], 'side_result.json')
        rec = {'name': r['name'], 'role': 'side', 'runtime': r.get('runtime'), 'status': 'FAILED', 'why': f'no side_result.json under side/{r["name"]}', 'mps_pct': r.get('mps_pct') if a.mode == 'mps' else None}
        if os.path.exists(sj):
            try: rec.update(json.load(open(sj))); rec.setdefault('status', 'OK')
            except Exception as ex: rec['why'] = f'side_result.json unreadable: {ex}'
        out['rows'].append(rec)
    if a.mode == 'mps':
        srv = os.path.join(out['arm']['mps_daemon']['log_dir'], 'server.log')
        txt = open(srv, errors='ignore').read() if os.path.exists(srv) else ''
        out['arm']['mps_verified'] = bool(txt.strip()) and ('connect' in txt.lower() or 'client' in txt.lower()); out['arm']['mps_server_log_lines'] = len(txt.splitlines())
        mps_quit()
    order = {r['name']: i for i, r in enumerate(frames)}; res.sort(key=lambda r: order.get(r['name'], 99))
    for r in res:
        s = solo.get(r['name']) or {}; sp = s.get('p99_ms'); r['role'] = 'frame'; r['paced_solo_p99_ms'] = sp
        r['contention_factor'] = (r['p99_ms'] / sp) if (sp and r.get('p99_ms')) else None
    out['rows'] = res + out['rows']
    out['sum_paced_solo_p99_ms'] = sum((r.get('paced_solo_p99_ms') or 0) for r in res)
    out['sum_target_time_share'] = sum(r['target_hz'] * (r.get('paced_solo_p99_ms') or 0) / 1000 for r in res)
    bad = [r['name'] for r in out['rows'] if r.get('status') == 'FAILED']
    out['status'] = 'OK' if not bad else 'FAILED: ' + ', '.join(bad)
    dump()
    print(f"[{a.mode}] {mix['mix']}: status {out['status']}")
    print(f"{'row':28s} {'prio':>4s} {'target Hz':>9s} {'achieved':>9s} {'p50':>7s} {'p99':>7s} {'paced solo':>10s} {'x':>5s} {'miss%':>6s} {'overrun%':>8s}")
    for r in res:
        print(f"{r['name']:28s} {r.get('stream_priority', 0):4d} {r['target_hz']:9.1f} {r['achieved_hz']:9.1f} {r['p50_ms'] or float('nan'):7.2f} {r['p99_ms'] or float('nan'):7.2f} "
              f"{r['paced_solo_p99_ms'] or 0:10.2f} {r['contention_factor'] or 0:5.2f} {100*r['miss_frac']:6.1f} {100*r['overrun_frac']:8.1f}")
    for r in out['rows']:
        if r.get('role') == 'side': print(f"{r['name']:28s} side {r.get('runtime')}: {r.get('status')} {r.get('summary', r.get('why', ''))}")
    sys.exit(0 if out['status'] == 'OK' else 5)


if __name__ == '__main__': main()
