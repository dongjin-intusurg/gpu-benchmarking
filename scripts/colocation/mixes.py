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
import argparse
import csv
import glob
import json
import os
import sys

ROLES = ('frame', 'side')
ARMS_BY_PLATFORM = {'jetson': ['plain', 'mps', 'streams', 'mig'],
                    'discrete': ['plain', 'mps', 'streams', 'mig']}
HZ_GRID_FALLBACK = [1, 2, 5, 10, 20, 30]
REQUIRED_COLUMNS = {'row', 'role', 'hz', 'deadline_ms'}
DEFAULT_RUNTIME_BY_KIND = {'engine': 'tensorrt', 'e2e': 'asr_driver'}


def mix_name_of(path):
    return os.path.splitext(os.path.basename(path))[0]


def read_mix(path):
    """Rows of one mix CSV (comment and blank lines skipped), each carrying its 1-based '_line'."""
    with open(path, newline='') as handle:
        numbered = [(number, line) for number, line in enumerate(handle, 1)
                    if line.strip() and not line.lstrip().startswith('#')]
    if not numbered:
        raise ValueError(f'{path}: empty')
    reader = csv.DictReader([line for _, line in numbered])
    if not reader.fieldnames or not REQUIRED_COLUMNS <= set(reader.fieldnames):
        raise ValueError(f'{path}: header must contain {sorted(REQUIRED_COLUMNS)} (got {reader.fieldnames})')
    rows = []
    for (line_number, _), record in zip(numbered[1:], reader):
        if record.get(None):
            raise ValueError(f'{path}:{line_number}: too many fields (an unquoted comma?)')
        record = {key: (value or '').strip() for key, value in record.items()}
        if not record['row']:
            continue
        record['_line'] = line_number
        rows.append(record)
    return rows


def load_registry_rows(rows_dir):
    records = {}
    for path in sorted(glob.glob(os.path.join(rows_dir, '*.jsonl'))):
        for line in open(path):
            if line.strip():
                record = json.loads(line)
                records[record['row']] = record
    return records


def load_results(path):
    results = json.load(open(path))
    return results, {row['name']: row for row in results.get('rows', [])}


def _num(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _solo_summary(solo):
    score = solo.get('solo') or {}
    return {'latency_ms': solo.get('latency_ms'), 'latency_source': solo.get('latency_source'),
            'bytes_per_frame_MB': solo.get('bytes_per_frame_MB'), 'vram_mb': solo.get('vram_mb'),
            'hz_stage4': solo.get('hz'), 'deadline_ms_stage4': solo.get('deadline_ms'),
            'arch_gflops': solo.get('arch_gflops'), 'N': score.get('N'), 'cause': score.get('cause'),
            'N_vs_hz': score.get('N_vs_hz'), 'max_hz_at_N1': score.get('max_hz_at_N1'),
            'max_hz_bound_by': score.get('max_hz_bound_by')}


def _attach_side_harness_rows(row, registry_record, results_rows, solo, where, errors):
    """The harness rows stage 4 derived for a side row's runtime: <row>_e2e / <row>_decode (generative);
    e2e driver rows are registered under the model's name, hence the <model>_e2e / <model>_decode fallback."""
    kind = row['kind']
    bases = [row['name']] + ([registry_record['model']] if registry_record.get('model') else [])
    e2e = next((results_rows[f'{base}_e2e'] for base in bases if f'{base}_e2e' in results_rows), None)
    decode = next((results_rows[f'{base}_decode'] for base in bases if f'{base}_decode' in results_rows),
                  None)
    if kind == 'e2e' and e2e is None:
        e2e = solo
    row['solo_e2e'] = e2e and {'latency_ms': e2e.get('latency_ms'),
                               'latency_source': e2e.get('latency_source'), 'vram_mb': e2e.get('vram_mb'),
                               'bytes_per_frame_MB': e2e.get('bytes_per_frame_MB'),
                               'detail': e2e.get('e2e'), 'solo': e2e.get('solo')}
    row['solo_decode'] = decode and {'latency_ms': decode.get('latency_ms'), 'hz': decode.get('hz'),
                                     'bytes_per_frame_MB': decode.get('bytes_per_frame_MB'),
                                     'detail': decode.get('e2e'), 'solo': decode.get('solo')}
    if kind == 'generative' and (e2e is None or decode is None):
        errors.append(f'{where}: stage-4 results lack {row["name"]}_e2e / {row["name"]}_decode '
                      f'(the battery rows) - re-run stage 4 with the battery')
    if kind == 'e2e' and decode is None:
        errors.append(f'{where}: stage-4 results lack the {registry_record.get("model")}_decode row')


def _resolve_row(record, registry_record, results_rows, prio_range, where, errors):
    """One mix row -> its resolved dict. Every problem is appended to errors; the row is built regardless."""
    role = record['role']
    kind = registry_record.get('kind')
    if role == 'frame' and kind != 'engine':
        errors.append(f'{where}: role frame needs an engine row (row_loop paces TensorRT engines); '
                      f'this row is kind {kind!r} - use role side')
    if role == 'side' and kind == 'engine':
        errors.append(f'{where}: role side needs an e2e or generative row (its own runtime); '
                      f'an engine row is paced as a frame row')

    hz_is_x = record['hz'].upper() == 'X'
    hz = None if hz_is_x else _num(record['hz'])
    if hz_is_x and role == 'frame':
        errors.append(f'{where}: X (rate undecided) is only legal on a side row; '
                      f'a frame row needs a pacing rate')
    if not hz_is_x and (hz is None or hz < 0):
        errors.append(f'{where}: hz must be a number >= 0 or X, got {record["hz"]!r}')
    if role == 'frame' and hz is not None and hz <= 0:
        errors.append(f'{where}: a frame row needs hz > 0')

    deadline_ms = _num(record['deadline_ms'])
    if deadline_ms is None or deadline_ms <= 0:
        errors.append(f'{where}: deadline_ms must be > 0, got {record["deadline_ms"]!r}')

    prio = 0
    if record.get('prio'):
        prio_value = _num(record['prio'])
        if prio_value is None or prio_value != int(prio_value):
            errors.append(f'{where}: prio must be an integer, got {record["prio"]!r}')
        else:
            prio = int(prio_value)
            if prio_range and not (min(prio_range) <= prio <= max(prio_range)):
                errors.append(f'{where}: prio {prio} outside this device\'s stream priority range '
                              f'{prio_range}')
    if record.get('prio') and role != 'frame':
        errors.append(f'{where}: prio applies to frame rows only (streams arm)')

    mps_pct = None
    if record.get('mps_pct'):
        mps_value = _num(record['mps_pct'])
        if mps_value is None or not (1 <= mps_value <= 100):
            errors.append(f'{where}: mps_pct must be 1..100, got {record["mps_pct"]!r}')
        else:
            mps_pct = int(mps_value)

    solo = results_rows.get(record['row'])
    if solo is None:
        errors.append(f'{where}: not in the stage-4 results.json (run stage 4 for it first - '
                      f'solo p99/bytes/VRAM come from there; '
                      f'SOLO_RESULTS=<results.json> selects another run)')

    row = {'name': record['row'], 'role': role, 'kind': kind,
           'runtime': registry_record.get('runtime') or DEFAULT_RUNTIME_BY_KIND.get(kind),
           'model': registry_record.get('model'), 'precision': registry_record.get('precision'),
           'hz': hz, 'hz_is_x': hz_is_x, 'deadline_ms': deadline_ms, 'prio': prio, 'mps_pct': mps_pct,
           'note': record.get('note', ''), 'registry': registry_record}
    if kind == 'engine':
        row['engine'] = registry_record.get('engine')
        row['run_flags'] = (registry_record.get('run_flags') or '').split()
        if not registry_record.get('engine') or not os.path.exists(registry_record['engine']):
            errors.append(f'{where}: engine file missing: {registry_record.get("engine")}')
    if solo is not None:
        row['solo'] = _solo_summary(solo)
        if role == 'side':
            _attach_side_harness_rows(row, registry_record, results_rows, solo, where, errors)
    return row


def resolve(mix_path, registry, results_rows, platform, prio_range=None):
    """-> (resolved dict, errors list). Every error is collected; the dict is complete when errors == []."""
    name = mix_name_of(mix_path)
    out = {'mix': name, 'path': os.path.abspath(mix_path), 'platform': platform, 'rows': []}
    try:
        records = read_mix(mix_path)
    except Exception as exc:
        return out, [str(exc)]
    if not records:
        return out, [f'{name}: no rows']
    errors = []
    seen = set()
    for record in records:
        where = f'{name}:{record["_line"]} {record["row"]}'
        if record['row'] in seen:
            errors.append(f'{where}: duplicate row')
            continue
        seen.add(record['row'])
        registry_record = registry.get(record['row'])
        if registry_record is None:
            errors.append(f'{where}: not a registered stage-4 row (engines/<tag>/registry_rows)')
            continue
        if record['role'] not in ROLES:
            errors.append(f'{where}: role must be one of {ROLES}, got {record["role"]!r}')
            continue
        out['rows'].append(_resolve_row(record, registry_record, results_rows, prio_range, where, errors))
    out['frame_rows'] = [row['name'] for row in out['rows'] if row['role'] == 'frame']
    out['side_rows'] = [row['name'] for row in out['rows'] if row['role'] == 'side']
    if not out['frame_rows']:
        errors.append(f'{name}: a mix needs at least one frame row')
    return out, errors


def print_mix_table(resolved, errors):
    """The "mix <name>  (N rows: F frame, S side)" header line is grepped by shell scripts: keep it."""
    invalid = '  INVALID' if errors else ''
    print(f'\nmix {resolved["mix"]}  ({len(resolved["rows"])} rows: {len(resolved["frame_rows"])} frame, '
          f'{len(resolved["side_rows"])} side)' + invalid)
    print(f'  {"row":30} {"role":6} {"kind":10} {"hz":>7} {"dl ms":>6} {"prio":>4} {"mps%":>4}  '
          f'{"solo p99":>8}  engine / runtime')
    for row in resolved['rows']:
        hz = 'X' if row['hz_is_x'] else f'{row["hz"]:g}'
        solo = row.get('solo') or {}
        registry_record = row['registry']
        registry_dir = registry_record.get('llm_dir') or registry_record.get('engine_dir') or ''
        target = row.get('engine') or registry_dir
        print(f'  {row["name"]:30} {row["role"]:6} {str(row["kind"]):10} {hz:>7} '
              f'{row["deadline_ms"] or 0:6g} {row["prio"]:4d} {str(row["mps_pct"] or ""):>4}  '
              f'{solo.get("latency_ms") or 0:8.2f}  {target}')


def select_mix_files(mix_dir, wanted):
    """Registered mixes (template.csv excluded), narrowed to the --mix subset when one is given."""
    registered = sorted(glob.glob(os.path.join(mix_dir, '*.csv')))
    files = [path for path in registered if os.path.basename(path) != 'template.csv']
    if wanted:
        files = [path for path in files if mix_name_of(path) in set(wanted)]
        missing = set(wanted) - {mix_name_of(path) for path in files}
        if missing:
            registered_names = [mix_name_of(path) for path in registered]
            sys.exit(f'unknown mix(es) {sorted(missing)} - registered: {registered_names}')
    if not files:
        sys.exit(f'no mixes registered under {mix_dir} (copy template.csv to <mix>.csv)')
    return files


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='cmd', required=True)
    for command in ('validate', 'resolve', 'list'):
        sub = subparsers.add_parser(command)
        sub.add_argument('--rows-dir', required=True)
        sub.add_argument('--results', required=True)
        sub.add_argument('--platform', required=True)
        sub.add_argument('--prio-range', default='', help='lo,hi from row_loop --prio-range')
        if command == 'resolve':
            sub.add_argument('--mix', required=True)
            sub.add_argument('--out', required=True)
        else:
            sub.add_argument('--mix-dir', required=True)
            sub.add_argument('--mix', action='append', default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    registry = load_registry_rows(args.rows_dir)
    if not registry:
        sys.exit(f'no registry rows under {args.rows_dir} - run stage 4 (build) first')
    if not os.path.exists(args.results):
        sys.exit(f'stage-4 results not found: {args.results}')
    _, results_rows = load_results(args.results)
    prio_range = tuple(int(bound) for bound in args.prio_range.split(',')) if args.prio_range else None

    if args.cmd == 'resolve':
        resolved, errors = resolve(args.mix, registry, results_rows, args.platform, prio_range)
        resolved['stage4_results'] = os.path.abspath(args.results)
        resolved['errors'] = errors
        json.dump(resolved, open(args.out, 'w'), indent=1)
        for error in errors:
            print('ERROR', error, file=sys.stderr)
        sys.exit(1 if errors else 0)

    files = select_mix_files(args.mix_dir, args.mix)
    all_errors = []
    for path in files:
        resolved, errors = resolve(path, registry, results_rows, args.platform, prio_range)
        all_errors += errors
        if args.cmd == 'list':
            print_mix_table(resolved, errors)
    for error in all_errors:
        print('ERROR', error, file=sys.stderr)
    if args.cmd == 'validate' and not all_errors:
        print(f'{len(files)} mix(es) valid: {", ".join(mix_name_of(path) for path in files)}')
    sys.exit(1 if all_errors else 0)


if __name__ == '__main__':
    main()
