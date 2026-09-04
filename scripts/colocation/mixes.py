#!/usr/bin/env python3
"""Mix registration for the co-location stage: read, resolve, validate.

A mix is one CSV under configs/colocation/ whose rows are stage-4 row names:

    row,role,hz,deadline_ms,prio,mps_pct,note
    depth_int8,frame,30,33.3,0,,paced by row_loop at 30 Hz
    tracker_int8,frame,30,33.3,-5,,stream priority (streams arm only)
    asr_e2e,side,0,50,,35,real-time ASR driver; 35 % MPS thread cap (mps arm only)
    vlm_nvfp4,side,X,100,,,back-to-back generative battery; X = spec rate undecided

  role   frame  -> an engine row (kind 'engine') paced by row_loop at hz; in the makespan
         side   -> the model's own runtime (kind 'e2e' ASR driver in real time, kind
                   'generative' battery back-to-back); reported by its own harness
  hz     frame rows: the mix rate (> 0). side rows: 0 or a number for the composed
         table; X = "rate not decided" (charged back-to-back, time_share_at_spec on the
         stage-4 rate grid). X is illegal on a frame row.
  deadline_ms  the mix deadline (overrides the row's stage-4 placeholder)
  prio   CUDA stream priority, streams arm only (numerically lower = higher); blank = 0
  mps_pct  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE for this row's process, mps arm only

Engines, run flags, precision and the side-load inputs are resolved from the registry
row records (engines/<tag>/registry_rows/*.jsonl); solo p99 / bytes / VRAM from the
newest stage-4 results.json. Nothing is re-typed in the mix.

    mixes.py validate --mix-dir D --rows-dir R --results J --platform P [--prio-range lo,hi] [--mix m]...
    mixes.py resolve  --mix F --rows-dir R --results J --platform P --out resolved.json
    mixes.py list     --mix-dir D --rows-dir R --results J --platform P
"""
import argparse, csv, glob, json, os, sys

ROLES = ('frame', 'side')
ARMS_BY_PLATFORM = {'jetson': ['plain', 'mps', 'streams', 'mig'], 'discrete': ['plain', 'mps', 'streams', 'mig']}
HZ_GRID_FALLBACK = [1, 2, 5, 10, 20, 30]


def read_mix(path):
    rows = []
    with open(path, newline='') as f:
        lines = [(i, l) for i, l in enumerate(f, 1) if l.strip() and not l.lstrip().startswith('#')]
    if not lines:
        raise ValueError(f'{path}: empty')
    rd = csv.DictReader([l for _, l in lines])
    need = {'row', 'role', 'hz', 'deadline_ms'}
    if not rd.fieldnames or not need <= set(rd.fieldnames):
        raise ValueError(f'{path}: header must contain {sorted(need)} (got {rd.fieldnames})')
    if True:
        for (i, _), r in zip(lines[1:], rd):
            if r.get(None):
                raise ValueError(f'{path}:{i}: too many fields (an unquoted comma?)')
            r = {k: (v or '').strip() for k, v in r.items()}
            if not r['row']:
                continue
            r['_line'] = i
            rows.append(r)
    return rows


def load_registry_rows(rows_dir):
    recs = {}
    for f in sorted(glob.glob(os.path.join(rows_dir, '*.jsonl'))):
        for l in open(f):
            if l.strip():
                r = json.loads(l); recs[r['row']] = r
    return recs


def load_results(path):
    R = json.load(open(path))
    return R, {r['name']: r for r in R.get('rows', [])}


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def resolve(mix_path, registry, results_rows, platform, prio_range=None):
    """-> (resolved dict, errors list). Every error is collected; the dict is complete when errors == []."""
    name = os.path.splitext(os.path.basename(mix_path))[0]
    errs = []; out = {'mix': name, 'path': os.path.abspath(mix_path), 'platform': platform, 'rows': []}
    try:
        rows = read_mix(mix_path)
    except Exception as e:
        return out, [str(e)]
    if not rows:
        return out, [f'{name}: no rows']
    seen = set()
    for r in rows:
        where = f'{name}:{r["_line"]} {r["row"]}'
        if r['row'] in seen:
            errs.append(f'{where}: duplicate row'); continue
        seen.add(r['row'])
        rec = registry.get(r['row'])
        if rec is None:
            errs.append(f'{where}: not a registered stage-4 row (engines/<tag>/registry_rows)'); continue
        if r['role'] not in ROLES:
            errs.append(f'{where}: role must be one of {ROLES}, got {r["role"]!r}'); continue
        kind = rec.get('kind')
        if r['role'] == 'frame' and kind != 'engine':
            errs.append(f'{where}: role frame needs an engine row (row_loop paces TensorRT engines); this row is kind {kind!r} - use role side')
        if r['role'] == 'side' and kind == 'engine':
            errs.append(f'{where}: role side needs an e2e or generative row (its own runtime); an engine row is paced as a frame row')
        hz_is_x = r['hz'].upper() == 'X'
        hz = None if hz_is_x else _num(r['hz'])
        if hz_is_x and r['role'] == 'frame':
            errs.append(f'{where}: X (rate undecided) is only legal on a side row; a frame row needs a pacing rate')
        if not hz_is_x and (hz is None or hz < 0):
            errs.append(f'{where}: hz must be a number >= 0 or X, got {r["hz"]!r}')
        if r['role'] == 'frame' and hz is not None and hz <= 0:
            errs.append(f'{where}: a frame row needs hz > 0')
        dl = _num(r['deadline_ms'])
        if dl is None or dl <= 0:
            errs.append(f'{where}: deadline_ms must be > 0, got {r["deadline_ms"]!r}')
        prio = 0
        if r.get('prio'):
            p = _num(r['prio'])
            if p is None or p != int(p):
                errs.append(f'{where}: prio must be an integer, got {r["prio"]!r}')
            else:
                prio = int(p)
                if prio_range and not (min(prio_range) <= prio <= max(prio_range)):
                    errs.append(f'{where}: prio {prio} outside this device\'s stream priority range {prio_range}')
        if r.get('prio') and r['role'] != 'frame':
            errs.append(f'{where}: prio applies to frame rows only (streams arm)')
        mps_pct = None
        if r.get('mps_pct'):
            m = _num(r['mps_pct'])
            if m is None or not (1 <= m <= 100):
                errs.append(f'{where}: mps_pct must be 1..100, got {r["mps_pct"]!r}')
            else:
                mps_pct = int(m)
        solo = results_rows.get(r['row'])
        if solo is None:
            errs.append(f'{where}: not in the stage-4 results.json (run stage 4 for it first - solo p99/bytes/VRAM come from there; SOLO_RESULTS=<results.json> selects another run)')
        row = {'name': r['row'], 'role': r['role'], 'kind': kind, 'runtime': rec.get('runtime') or {'engine': 'tensorrt', 'e2e': 'asr_driver'}.get(kind),
               'model': rec.get('model'), 'precision': rec.get('precision'), 'hz': hz, 'hz_is_x': hz_is_x, 'deadline_ms': dl,
               'prio': prio, 'mps_pct': mps_pct, 'note': r.get('note', ''), 'registry': rec}
        if kind == 'engine':
            row['engine'] = rec.get('engine'); row['run_flags'] = (rec.get('run_flags') or '').split()
            if not rec.get('engine') or not os.path.exists(rec['engine']):
                errs.append(f'{where}: engine file missing: {rec.get("engine")}')
        if solo is not None:
            s = solo.get('solo') or {}
            row['solo'] = {'latency_ms': solo.get('latency_ms'), 'latency_source': solo.get('latency_source'),
                           'bytes_per_frame_MB': solo.get('bytes_per_frame_MB'), 'vram_mb': solo.get('vram_mb'),
                           'hz_stage4': solo.get('hz'), 'deadline_ms_stage4': solo.get('deadline_ms'),
                           'arch_gflops': solo.get('arch_gflops'), 'N': s.get('N'), 'cause': s.get('cause'),
                           'N_vs_hz': s.get('N_vs_hz'), 'max_hz_at_N1': s.get('max_hz_at_N1'), 'max_hz_bound_by': s.get('max_hz_bound_by')}
            if r['role'] == 'side':
                # the harness rows stage 4 derived for this runtime: <row>_e2e / <row>_decode (generative),
                # <model>_e2e / <model>_decode (e2e driver rows are registered under the model's e2e name)
                base = [r['row']] + ([rec['model']] if rec.get('model') else [])
                e2e = next((results_rows[f'{b}_e2e'] for b in base if f'{b}_e2e' in results_rows), None)
                dec = next((results_rows[f'{b}_decode'] for b in base if f'{b}_decode' in results_rows), None)
                if kind == 'e2e' and e2e is None:
                    e2e = solo
                row['solo_e2e'] = e2e and {'latency_ms': e2e.get('latency_ms'), 'latency_source': e2e.get('latency_source'), 'vram_mb': e2e.get('vram_mb'),
                                           'bytes_per_frame_MB': e2e.get('bytes_per_frame_MB'), 'detail': e2e.get('e2e'), 'solo': e2e.get('solo')}
                row['solo_decode'] = dec and {'latency_ms': dec.get('latency_ms'), 'hz': dec.get('hz'), 'bytes_per_frame_MB': dec.get('bytes_per_frame_MB'),
                                              'detail': dec.get('e2e'), 'solo': dec.get('solo')}
                if kind == 'generative' and (e2e is None or dec is None):
                    errs.append(f'{where}: stage-4 results lack {r["row"]}_e2e / {r["row"]}_decode (the battery rows) - re-run stage 4 with the battery')
                if kind == 'e2e' and dec is None:
                    errs.append(f'{where}: stage-4 results lack the {rec.get("model")}_decode row')
        out['rows'].append(row)
    frames = [x for x in out['rows'] if x['role'] == 'frame']
    if not frames:
        errs.append(f'{name}: a mix needs at least one frame row')
    out['frame_rows'] = [x['name'] for x in frames]; out['side_rows'] = [x['name'] for x in out['rows'] if x['role'] == 'side']
    return out, errs


def prio_range_of(row_loop):
    """query the device once; None when the binary is absent (validation then skips the range check)"""
    import subprocess
    if not row_loop or not os.path.exists(row_loop):
        return None
    try:
        o = subprocess.run([row_loop, '--prio-range'], capture_output=True, text=True, timeout=60).stdout.split()
        return (int(o[0]), int(o[1])) if len(o) == 2 else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    for c in ('validate', 'resolve', 'list'):
        p = sp.add_parser(c)
        p.add_argument('--rows-dir', required=True); p.add_argument('--results', required=True); p.add_argument('--platform', required=True)
        p.add_argument('--prio-range', default='', help='lo,hi from row_loop --prio-range')
        if c == 'resolve':
            p.add_argument('--mix', required=True); p.add_argument('--out', required=True)
        else:
            p.add_argument('--mix-dir', required=True); p.add_argument('--mix', action='append', default=[])
    a = ap.parse_args()
    registry = load_registry_rows(a.rows_dir)
    if not registry:
        sys.exit(f'no registry rows under {a.rows_dir} - run stage 4 (build) first')
    if not os.path.exists(a.results):
        sys.exit(f'stage-4 results not found: {a.results}')
    R, rr = load_results(a.results)
    pr = tuple(int(x) for x in a.prio_range.split(',')) if a.prio_range else None
    if a.cmd == 'resolve':
        res, errs = resolve(a.mix, registry, rr, a.platform, pr)
        res['stage4_results'] = os.path.abspath(a.results); res['errors'] = errs
        json.dump(res, open(a.out, 'w'), indent=1)
        for e in errs: print('ERROR', e, file=sys.stderr)
        sys.exit(1 if errs else 0)
    files = sorted(glob.glob(os.path.join(a.mix_dir, '*.csv')))
    files = [f for f in files if os.path.basename(f) != 'template.csv']
    if a.mix:
        want = set(a.mix); files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in want]
        missing = want - {os.path.splitext(os.path.basename(f))[0] for f in files}
        if missing:
            sys.exit(f'unknown mix(es) {sorted(missing)} - registered: {[os.path.splitext(os.path.basename(f))[0] for f in sorted(glob.glob(os.path.join(a.mix_dir, "*.csv")))]}')
    if not files:
        sys.exit(f'no mixes registered under {a.mix_dir} (copy template.csv to <mix>.csv)')
    all_errs = []
    for f in files:
        res, errs = resolve(f, registry, rr, a.platform, pr); all_errs += errs
        if a.cmd == 'list':
            print(f'\nmix {res["mix"]}  ({len(res["rows"])} rows: {len(res["frame_rows"])} frame, {len(res["side_rows"])} side)' + ('  INVALID' if errs else ''))
            print(f'  {"row":30} {"role":6} {"kind":10} {"hz":>7} {"dl ms":>6} {"prio":>4} {"mps%":>4}  {"solo p99":>8}  engine / runtime')
            for x in res['rows']:
                hz = 'X' if x['hz_is_x'] else f'{x["hz"]:g}'
                s = x.get('solo') or {}
                tgt = x.get('engine') or (x['registry'].get('llm_dir') or x['registry'].get('engine_dir') or '')
                print(f'  {x["name"]:30} {x["role"]:6} {str(x["kind"]):10} {hz:>7} {x["deadline_ms"] or 0:6g} {x["prio"]:4d} {str(x["mps_pct"] or ""):>4}  {s.get("latency_ms") or 0:8.2f}  {tgt}')
    for e in all_errs: print('ERROR', e, file=sys.stderr)
    if a.cmd == 'validate' and not all_errs:
        print(f'{len(files)} mix(es) valid: {", ".join(os.path.splitext(os.path.basename(f))[0] for f in files)}')
    sys.exit(1 if all_errs else 0)


if __name__ == '__main__':
    main()
