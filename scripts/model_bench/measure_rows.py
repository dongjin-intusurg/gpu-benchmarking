#!/usr/bin/env python3
"""Row plumbing for measure_models.sh - everything that is parsing, not running.

  select  --rows-dir D [--only m]... [--kind K]        -> JSON list (or --shell lines / --table)
  mix     --rows-dir D [--only m]... --out mix.csv     -> the 7-column mix of engine rows; prints the count
  e2e / edgellm / trtllm  --row R ... --vram V --out rows.json  -> compute_budgets.py input rows
  merge   --budgets B [--engine-results E] --provenance P --out results.json --report report.md
Every row carries name, latency_ms (the latency of record), hz, deadline_ms, arch_gflops,
bytes_per_frame_MB (or None + bytes_source), vram_mb, latency_source, plus a detail block.
"""
import argparse
import glob
import json
import os
import re
import shlex
import statistics
import sys

# Spec rate for the generative rows is not fixed yet: MODEL_HZ is a placeholder. Every generative /
# e2e row therefore also carries N at this rate grid plus the highest rate the row sustains solo at
# N >= 1, so the verdict can be read off once the rate is decided (compute_budgets.py scores the grid).
HZ_GRID = [1, 2, 5, 10, 20, 30]


def load_rows(rows_dir, only=(), kind=None):
    rows = []
    for path in sorted(glob.glob(os.path.join(rows_dir, '*.jsonl'))):
        model = os.path.basename(path)[:-6]
        if only and model not in only:
            continue
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if kind and row.get('kind') != kind:
                continue
            rows.append(row)
    return rows


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    return values[min(len(values) - 1, int(round(fraction * (len(values) - 1))))]


def median_if_present(records, key):
    if not all(key in record for record in records):
        return None
    return statistics.median([record[key] for record in records])


def read_vram(path):
    try:
        sample = json.load(open(path))
        return sample.get('peak_mb'), sample.get('source')
    except Exception:
        return None, 'none'


def read_drift(path):
    """drift_report.py verdict for the row window -> the clock_integrity block the engine rows
    carry; a FAIL invalidates the row (measurement_valid False)."""
    try:
        drift = json.load(open(path))
    except Exception:
        return {'verdict': None, 'note': 'drift.json missing - no under-load clock evidence recorded'}, True
    integrity = {'verdict': drift.get('verdict'), 'pct_at_target': drift.get('pct_at_target'),
                 'reference_clock_mhz': drift.get('reference_clock_mhz'),
                 'throttle_reasons_seen': drift.get('throttle_reasons_seen', drift.get('throttle_reasons')),
                 'clamp_events': drift.get('clamp_events',
                                           drift.get('clamp_event_count', drift.get('oc_clamp_events')))}
    return integrity, integrity['verdict'] != 'FAIL'


def stamp_rows(rows, drift_path):
    integrity, valid = read_drift(drift_path)
    for row in rows:
        row['clock_integrity'] = integrity
        row['measurement_valid'] = valid
    return rows


def write_rows(rows, args):
    json.dump(stamp_rows(rows, args.drift), open(args.out, 'w'), indent=1)


def table_line(row):
    what = {'engine': row.get('engine'), 'e2e': row.get('src'),
            'generative': row.get('llm_dir') or row.get('serving_dir')}.get(row.get('kind'), '')
    return (f"{row['row']:32} {row.get('kind', ''):10} {str(row.get('precision', '')):8} "
            f"{row.get('runtime', 'tensorrt'):9} hz={row.get('hz')} deadline={row.get('deadline_ms')}  "
            f"{what}")


def shell_line(row):
    """KEY=value pairs for `eval`; nested fields are not exportable and are skipped."""
    pairs = []
    for key, value in row.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            pairs.append(f'{key.upper()}={shlex.quote(str("" if value is None else value))}')
    return ' '.join(pairs)


def cmd_select(args):
    rows = load_rows(args.rows_dir, args.only, args.kind)
    if args.table:
        for row in rows:
            print(table_line(row))
    elif args.shell:
        for row in rows:
            print(shell_line(row))
    else:
        print(json.dumps(rows, indent=1))


def cmd_mix(args):
    rows = load_rows(args.rows_dir, args.only, 'engine')
    with open(args.out, 'w') as out:
        out.write('name,onnx,precision,hz,deadline_ms,arch_gflops,extra\n')
        for row in rows:
            out.write(f"{row['row']},{row['engine']},{row['precision']},{row['hz']},{row['deadline_ms']},"
                      f"{row.get('arch_gflops') or 0},{row.get('run_flags') or ''}\n")
    print(len(rows))


def e2e_detail(records, spec, jsonl_path):
    totals = [record['gpu_total_ms'] for record in records]
    seconds = [record.get('seconds') for record in records if record.get('seconds')]
    tokens = [record.get('n_tokens') for record in records if record.get('n_tokens')]
    speech_tokens_per_s = None
    if seconds and tokens and len(seconds) == len(tokens):
        speech_tokens_per_s = sum(tokens) / sum(seconds)
    return {'clips': len({record.get('clip') for record in records}), 'records': len(records),
            'repeats': spec.get('repeats'),
            'ttft_ms_median': statistics.median([record['ttft_ms'] for record in records]),
            'ttft_ms_p99': percentile([record['ttft_ms'] for record in records], .99),
            'encoder_ms_median': median_if_present(records, 'encoder_ms'),
            'first_step_ms_median': median_if_present(records, 'first_step_ms'),
            'decode_ms_median': median_if_present(records, 'decode_ms_median'),
            'decode_ms_p99_max': max(record['decode_ms_p99'] for record in records),
            'gpu_total_ms_median': statistics.median(totals), 'gpu_total_ms_p99': percentile(totals, .99),
            'wall_ms_median': statistics.median([record['wall_ms'] for record in records]),
            'rtf_wall_median': median_if_present(records, 'rtf_wall'),
            'speech_tokens_per_s': speech_tokens_per_s,
            'jsonl': jsonl_path}


def cmd_e2e(args):
    spec = json.load(open(args.row))
    records = [json.loads(line) for line in open(args.jsonl) if line.strip()]
    if not records:
        sys.exit(f'{args.jsonl}: the driver wrote no records')
    required = ('ttft_ms', 'decode_ms_p99', 'gpu_total_ms', 'wall_ms')
    missing = [key for key in required if any(key not in record for record in records)]
    if missing:
        sys.exit(f'{args.jsonl}: records lack required fields {missing} (the e2e contract needs {required})')
    vram_mb, vram_source = read_vram(args.vram)
    model = spec['model']
    hz = float(spec.get('hz') or 0)
    deadline_ms = float(spec.get('deadline_ms') or spec.get('model_deadline_ms') or 0)
    detail = e2e_detail(records, spec, args.jsonl)
    speech_rate = detail['speech_tokens_per_s']
    step_row = {'name': f'{model}_e2e', 'kind': 'e2e', 'precision': spec.get('precision'),
                'runtime': 'tensorrt-cpp-driver',
                'latency_ms': detail['gpu_total_ms_p99'], 'latency_source': 'e2e-gpu-total-p99',
                'hz': hz, 'deadline_ms': deadline_ms, 'arch_gflops': float(spec.get('arch_gflops') or 0),
                'bytes_per_frame_MB': None,
                'bytes_source': 'none (no per-request byte counter for a multi-engine driver)',
                'vram_mb': vram_mb, 'vram_source': vram_source, 'e2e': detail}
    # the decode loop is the row the per-token deadline judges: p99 of one decode step, charged at
    # the speech-rate token demand
    decode_row = {'name': f'{model}_decode', 'kind': 'e2e', 'precision': spec.get('precision'),
                  'runtime': 'tensorrt-cpp-driver',
                  'latency_ms': detail['decode_ms_p99_max'],
                  'latency_source': 'e2e-decode-step-p99 (max over clips)',
                  'hz': round(speech_rate, 3) if speech_rate else 0.0,
                  'hz_source': 'measured speech token rate (sum tokens / sum clip seconds)',
                  'deadline_ms': float(spec.get('model_deadline_ms') or deadline_ms), 'arch_gflops': 0.0,
                  'bytes_per_frame_MB': None, 'bytes_source': 'none',
                  'vram_mb': vram_mb, 'vram_source': vram_source,
                  'e2e': {'decode_ms_median': detail['decode_ms_median'],
                          'decode_ms_p99_max': detail['decode_ms_p99_max'],
                          'ttft_ms_median': detail['ttft_ms_median'], 'jsonl': args.jsonl}}
    write_rows([step_row, decode_row], args)
    print(f"  {step_row['name']:26} gpu_total p99 {step_row['latency_ms']:.3f} ms  "
          f"ttft {detail['ttft_ms_median']:.1f}  rtf {detail['rtf_wall_median']}")
    print(f"  {decode_row['name']:26} decode step p99 {decode_row['latency_ms']:.3f} ms  "
          f"at {decode_row['hz']} tok/s")


def bench_ms(path):
    try:
        match = re.search(r'E2E Time \(actual performance\): ([0-9.]+)', open(path, errors='ignore').read())
        return float(match.group(1)) if match else None
    except FileNotFoundError:
        return None


def read_llm_profile(path):
    """The llm_inference battery profile (real prompts + image, the runtime's own stage timers)."""
    profile = json.load(open(path))
    stages = {stage['stage_id']: stage['gpu_time_stats'] for stage in profile.get('stages', [])}
    generation = profile.get('generation', {})
    ttft_ms = sum(stages[key]['median_ms'] for key in ('vision_encoder', 'llm_prefill') if key in stages)
    if 'llm_generation' in stages:
        ttft_ms += stages['llm_generation']['median_ms']
    return {'ttft_ms': ttft_ms,
            'ttft_basis': 'vision_encoder + llm_prefill + first llm_generation step (stage medians)',
            'decode_ms_median': stages.get('llm_generation', {}).get('median_ms'),
            'decode_ms_p99': stages.get('llm_generation', {}).get('p99_ms'),
            'tokens_per_second': generation.get('tokens_per_second'),
            'generated_tokens': generation.get('generated_tokens'),
            'prefill_ms_median': stages.get('llm_prefill', {}).get('median_ms'),
            'visual_ms_median': stages.get('vision_encoder', {}).get('median_ms'),
            'peak_unified_memory_mb': profile.get('peak_unified_memory_mb'), 'profile': path,
            'stages': stages}


def edgellm_e2e_rows(spec, profile, chunk, step_row, engine_mb):
    """The battery pair: the chunk step at p99 of every stage, and the decode loop the per-token
    deadline judges - the same pair the C++ driver rows produce for the ASR model."""
    stage_stats = profile['stages']

    def stage_p99(key):
        return float(stage_stats.get(key, {}).get('p99_ms') or 0)

    step_p99 = stage_p99('vision_encoder') + stage_p99('llm_prefill') + chunk * stage_p99('llm_generation')
    common = {'kind': 'e2e', 'runtime': 'edgellm-llm_inference', 'precision': spec['precision']}
    e2e_row = {'name': f"{spec['row']}_e2e", **common,
               'latency_ms': round(step_p99, 3),
               'latency_source': f'llm_inference battery step = vision p99 + prefill p99 + {chunk} x decode '
                                 'p99 (stage timers)',
               'hz': float(spec['hz']), 'deadline_ms': float(spec['deadline_ms']),
               'arch_gflops': float(spec.get('arch_gflops') or 0),
               'hz_grid': HZ_GRID, 'bytes_per_frame_MB': step_row['bytes_per_frame_MB'],
               'bytes_source': step_row['bytes_source'],
               'vram_mb': step_row['vram_mb'], 'vram_source': step_row['vram_source'],
               'e2e': dict(profile, chunk=chunk, step_ms_p99=round(step_p99, 3), battery=spec.get('battery'),
                           image=spec.get('image'))}
    # modal: the decode loop runs at its own token rate, not the frame rate
    decode_row = {'name': f"{spec['row']}_decode", **common,
                  'latency_ms': round(profile['decode_ms_p99'], 3),
                  'latency_source': 'llm_inference battery decode step p99',
                  'hz': round(float(profile.get('tokens_per_second') or 0), 3),
                  'deadline_ms': float(spec['deadline_ms']),
                  'arch_gflops': 0.0, 'hz_grid': HZ_GRID,
                  'bytes_per_frame_MB': round(engine_mb, 1) if engine_mb else None,
                  'bytes_source': 'weights-stream-estimate: llm engine bytes per token (KV traffic '
                                  'excluded - a lower bound)',
                  'vram_mb': step_row['vram_mb'], 'vram_source': step_row['vram_source'],
                  'e2e': {'decode_ms_median': profile.get('decode_ms_median'),
                          'decode_ms_p99': profile['decode_ms_p99'],
                          'tokens_per_second': profile.get('tokens_per_second'),
                          'ttft_ms': profile['ttft_ms']}}
    return [e2e_row, decode_row]


def cmd_edgellm(args):
    spec = json.load(open(args.row))
    bench_dir = args.bench_dir
    has_visual = bool(spec.get('visual_dir'))
    visual_ms = bench_ms(os.path.join(bench_dir, 'visual.log')) if has_visual else 0.0
    prefill_ms = bench_ms(os.path.join(bench_dir, 'prefill.log'))
    reuse_ms = bench_ms(os.path.join(bench_dir, 'reuse.log'))
    decode_ms = bench_ms(os.path.join(bench_dir, 'decode.log'))
    if prefill_ms is None or decode_ms is None or (has_visual and visual_ms is None):
        sys.exit(f'{bench_dir}: llm_bench output incomplete (visual={visual_ms} prefill={prefill_ms} '
                 f'reuse={reuse_ms} decode={decode_ms})')
    chunk = int(spec.get('chunk') or 8)
    best_prefill_ms = min(value for value in (prefill_ms, reuse_ms) if value is not None)
    step_ms = (visual_ms or 0) + best_prefill_ms + chunk * decode_ms
    vram_mb, vram_source = read_vram(args.vram)
    profile = None
    if args.profile and os.path.exists(args.profile):
        profile = read_llm_profile(args.profile)
        if profile['peak_unified_memory_mb']:
            vram_mb = max(vram_mb or 0, profile['peak_unified_memory_mb'])
            vram_source = 'edgellm-profile-peak-unified'
    engine_mb = float(spec.get('engine_bytes') or 0) / 1e6
    visual_mb = float(spec.get('visual_bytes') or 0) / 1e6
    bytes_mb = engine_mb * (chunk + 1) + visual_mb if engine_mb else None
    # a resident engine can never occupy less than its own bytes: when the battery did not run
    # (sampler saw a process that died early) the sampled delta is meaningless, so the weights
    # are the floor of record
    if engine_mb and (vram_mb or 0) < engine_mb + visual_mb:
        vram_mb = round(engine_mb + visual_mb, 1)
        vram_source = 'engine-bytes-floor (llm + visual engine files; battery peak unavailable)'
    step_row = {'name': spec['row'], 'kind': 'generative', 'runtime': 'edgellm',
                'precision': spec['precision'], 'hz_grid': HZ_GRID,
                'latency_ms': round(step_ms, 3),
                'latency_source': f'llm_bench step = visual + best(prefill,reuse) + {chunk} x decode',
                'hz': float(spec['hz']), 'deadline_ms': float(spec['deadline_ms']),
                'arch_gflops': float(spec.get('arch_gflops') or 0),
                'bytes_per_frame_MB': round(bytes_mb, 1) if bytes_mb else None,
                'bytes_source': 'weights-stream-estimate: llm engine bytes x (chunk+1) + visual engine bytes '
                                '(KV traffic excluded - a lower bound)',
                'vram_mb': vram_mb, 'vram_source': vram_source,
                'generative': {'visual_ms': visual_ms, 'prefill_ms': prefill_ms, 'prefill_reuse_ms': reuse_ms,
                               'decode_ms': decode_ms,
                               'tokens_per_s_bench': round(1000.0 / decode_ms, 2), 'chunk': chunk,
                               'context_len': spec.get('context_len'), 'reuse_len': spec.get('reuse_len'),
                               'ttft_ms_bench': round((visual_ms or 0) + best_prefill_ms + decode_ms, 2),
                               'engine_mb': round(engine_mb, 1), 'visual_mb': round(visual_mb, 1),
                               'e2e': profile}}
    rows = [step_row]
    if profile and profile.get('decode_ms_p99'):
        rows += edgellm_e2e_rows(spec, profile, chunk, step_row, engine_mb)
    write_rows(rows, args)
    battery_note = ''
    if profile:
        battery_note = f"  e2e ttft {profile['ttft_ms']:.1f} tok/s {profile['tokens_per_second']:.1f}"
    print(f"  {step_row['name']:26} step {step_ms:.1f} ms (vis {visual_ms} + prefill {prefill_ms}/{reuse_ms} "
          f"+ {chunk}x{decode_ms})  tok/s {1000 / decode_ms:.1f}" + battery_note)
    if len(rows) > 1:
        print(f"  {rows[1]['name']:26} e2e step p99 {rows[1]['latency_ms']:.1f} ms   "
              f"{rows[2]['name']:26} decode p99 {rows[2]['latency_ms']:.2f} ms at {rows[2]['hz']} tok/s")


# --------------------------------------------------------------------- trtllm
def trtllm_side_rows(spec, row, chunk, compose, engine_mb, vram_mb, vram_source):
    """The <row>_e2e / <row>_decode pair every generative runtime exports, so a TensorRT-LLM model
    can be a stage-5 side row: the measured chunk step (what the deadline judges) and the per-token
    decode loop at its own token rate."""
    decode_ms = float(compose['decode_ms'])
    step_row = {'name': f"{spec['row']}_e2e", 'kind': 'e2e', 'runtime': 'trtllm', 'precision': spec['precision'],
                'latency_ms': row['latency_ms'],
                'latency_source': f'RequestPerfMetrics total_ms p99, chunk {chunk} (reuse on) - the measured step',
                'hz': float(spec['hz']), 'deadline_ms': float(spec['deadline_ms']),
                'arch_gflops': float(spec.get('arch_gflops') or 0),
                'hz_grid': HZ_GRID, 'bytes_per_frame_MB': row['bytes_per_frame_MB'],
                'bytes_source': row['bytes_source'],
                'vram_mb': vram_mb, 'vram_source': vram_source,
                'e2e': dict(row['generative'], chunk=int(chunk), step_ms_p99=row['latency_ms'],
                            image=spec.get('image'))}
    decode_row = {'name': f"{spec['row']}_decode", 'kind': 'e2e', 'runtime': 'trtllm',
                  'precision': spec['precision'],
                  'latency_ms': round(decode_ms, 3),
                  'latency_source': 'RequestPerfMetrics decode ms/token (sweep, reuse on)',
                  'hz': round(1000.0 / decode_ms, 3), 'deadline_ms': float(spec['deadline_ms']),
                  'arch_gflops': 0.0, 'hz_grid': HZ_GRID,
                  'bytes_per_frame_MB': round(engine_mb, 1) if engine_mb else None,
                  'bytes_source': 'weights-stream-estimate: checkpoint bytes per token (KV traffic excluded - '
                                  'a lower bound)',
                  'vram_mb': vram_mb, 'vram_source': vram_source,
                  'e2e': {'decode_ms': decode_ms, 'tokens_per_second': round(1000.0 / decode_ms, 2),
                          'ttft_ms': compose.get('ttft_ms')}}
    return [step_row, decode_row]


def cmd_trtllm(args):
    spec = json.load(open(args.row))
    sweep = json.load(open(args.sweep))
    chunk = str(spec.get('chunk') or 8)
    chunk_pass = (sweep.get('chunks') or {}).get(chunk) or {}
    compose = sweep.get('compose') or {}
    if not chunk_pass.get('total_ms_p99'):
        sys.exit(f'{args.sweep}: no chunk-{chunk} pass with total_ms_p99')
    vram_mb, vram_source = read_vram(args.vram)
    engine_mb = float(spec.get('engine_bytes') or 0) / 1e6
    decode_ms = compose.get('decode_ms')
    row = {'name': spec['row'], 'kind': 'generative', 'runtime': 'trtllm', 'precision': spec['precision'],
           'hz_grid': HZ_GRID,
           'latency_ms': chunk_pass['total_ms_p99'],
           'latency_source': f'RequestPerfMetrics total_ms p99, chunk {chunk} (reuse on)',
           'hz': float(spec['hz']), 'deadline_ms': float(spec['deadline_ms']),
           'arch_gflops': float(spec.get('arch_gflops') or 0),
           'bytes_per_frame_MB': round(engine_mb * (int(chunk) + 1), 1) if engine_mb else None,
           'bytes_source': 'weights-stream-estimate: checkpoint bytes x (chunk+1) (KV traffic excluded - '
                           'a lower bound)',
           'vram_mb': vram_mb, 'vram_source': vram_source,
           'generative': {'ttft_ms': compose.get('ttft_ms'),
                          'prefill_reuse_ms': compose.get('prefill_reuse_ms'),
                          'prefill_cold_ms': compose.get('prefill_cold_ms'), 'decode_ms': decode_ms,
                          'tokens_per_s_bench': round(1000.0 / decode_ms, 2) if decode_ms else None,
                          'step_ms_p50': chunk_pass.get('total_ms_p50'), 'n': chunk_pass.get('n'),
                          'chunk': int(chunk), 'sweep': args.sweep}}
    rows = [row]
    if decode_ms:
        rows += trtllm_side_rows(spec, row, chunk, compose, engine_mb, vram_mb, vram_source)
    write_rows(rows, args)
    derived = f"  -> +{rows[1]['name']} / {rows[2]['name']}" if len(rows) > 1 else ''
    print(f"  {row['name']:26} step p99 {row['latency_ms']:.1f} ms  ttft {compose.get('ttft_ms')}  "
          f"decode {decode_ms} ms/tok" + derived)


def engine_row_summary(model):
    return {'name': model['name'], 'kind': 'engine', 'runtime': 'tensorrt',
            'precision': model.get('precision'),
            'latency_ms': model.get('p99_ms'), 'latency_source': model.get('p99_source', 'trtexec p99'),
            'hz': model.get('hz'), 'deadline_ms': model.get('deadline_ms'),
            'arch_gflops': model.get('arch_gflops'),
            'bytes_per_frame_MB': model.get('bytes_per_frame_MB'), 'bytes_source': model.get('bytes_source'),
            'vram_mb': model.get('vram_budget_mb') or model.get('vram_mb'),
            'vram_source': model.get('vram_source'),
            'solo': model.get('solo') or {}, 'clock_integrity': model.get('clock_integrity'),
            'measurement_valid': model.get('measurement_valid')}


DEVICE_ROLLUP_KEYS = ('U_max', 'C', 'L', 'N', 'shortfall_cause_if_N_lt_1', 'workload_constant_gflops_per_s',
                      'score_tflops_at_deadline', 'budgets', 'binding_budget', 'bw_demand_is_upper_bound')


def merged_results(budgets, engine_results, provenance, rows, engine_results_path):
    merged = {'suite': engine_results.get('suite', 'per-model-solo'),
              'convention': engine_results.get('convention'), 'regime': engine_results.get('regime'),
              'composition': 'solo (one row at a time; N/C/L per row, never summed across rows)',
              'device': provenance.get('device'), 'platform': provenance.get('platform'),
              'device_tag': provenance.get('device_tag'),
              'provenance': provenance,
              'preflight': engine_results.get('preflight'),
              'clock_integrity': engine_results.get('clock_integrity'),
              'ceilings': engine_results.get('ceilings'),
              'bw_ceiling_source': (engine_results.get('bw_ceiling_source')
                                    or budgets.get('bw_ceiling_source')),
              'bw_eff_gbps': budgets.get('bw_eff_gbps'),
              'vram_capacity_mb': engine_results.get('vram_capacity_mb') or budgets.get('vram_capacity_mb'),
              'vram_capacity_source': (engine_results.get('vram_capacity_source')
                                       or budgets.get('vram_capacity_source')),
              'engine_rows_results': engine_results_path,
              'rows': rows,
              # the per-engine block keeps its full schema for the existing readers
              'models': engine_results.get('models', [])
              + [dict(row, p99_ms=row.get('latency_ms'), p99_source=row.get('latency_source'))
                 for row in budgets.get('rows', [])]}
    for key in DEVICE_ROLLUP_KEYS:
        if key in engine_results:
            merged[key] = engine_results[key]
    if 'N' in engine_results:
        merged['device_rollup_scope'] = ('engine rows only (the trtexec mix); generative and e2e rows are '
                                         'solo verdicts')
    return merged


def verdict_table(rows):
    lines = ['| row | kind | prec | latency ms | source | hz | deadline | U_max | C | L | N | cause | '
             'budget |',
             '|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|']
    for row in rows:
        solo = row.get('solo') or {}
        prefix = (f"| {row['name']} | {row.get('kind')} | {row.get('precision')} | ")
        if not solo:
            lines.append(prefix + f"{row.get('latency_ms')} | {row.get('latency_source', '')} | "
                         f"{row.get('hz')} | {row.get('deadline_ms')} | - | - | - | - | "
                         f"{row.get('error', 'unscored')} | |")
            continue
        if solo.get('budget_complete', True):
            completeness = 'complete'
        else:
            completeness = 'incomplete: ' + ','.join(solo.get('budgets_not_measured', []))
        if row.get('measurement_valid') is False:
            completeness += ' INVALID(drift)'
        lines.append(prefix + f"{row.get('latency_ms'):.3f} | {row.get('latency_source', '')} | "
                     f"{row.get('hz')} | {row.get('deadline_ms')} | {solo.get('U_max')} | {solo.get('C')} | "
                     f"{solo.get('L')} | **{solo.get('N')}** | {solo.get('cause')} | {completeness} |")
    return lines


def generative_table(rows):
    generative_rows = [row for row in rows if row.get('kind') == 'generative']
    if not generative_rows:
        return []
    lines = ['', '## Generative rows (TTFT / decode)', '',
             '| row | runtime | prec | visual ms | prefill ms | reuse ms | decode ms/tok | tok/s | TTFT ms | '
             'e2e TTFT ms | e2e tok/s | e2e decode p99 | peak mem MB |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in generative_rows:
        bench = row.get('generative') or {}
        battery = bench.get('e2e') or {}
        lines.append(f"| {row['name']} | {row.get('runtime')} | {row.get('precision')} | "
                     f"{bench.get('visual_ms')} | "
                     f"{bench.get('prefill_ms') or bench.get('prefill_cold_ms')} | "
                     f"{bench.get('prefill_reuse_ms')} | {bench.get('decode_ms')} | "
                     f"{bench.get('tokens_per_s_bench')} | "
                     f"{bench.get('ttft_ms_bench') or bench.get('ttft_ms')} | "
                     f"{battery.get('ttft_ms')} | {battery.get('tokens_per_second')} | "
                     f"{battery.get('decode_ms_p99')} | {row.get('vram_mb')} |")
    return lines


def rate_sweep_table(rows):
    sweep_rows = [row for row in rows if (row.get('solo') or {}).get('N_vs_hz')]
    if not sweep_rows:
        return []
    grid = list(sweep_rows[0]['solo']['N_vs_hz'].keys())
    lines = ['', '## Rate sweep - generative and e2e rows (spec rate not fixed: MODEL_HZ is a placeholder)',
             '',
             'N at each candidate rate with the period (1000/rate) as the deadline and the same measured '
             'budgets; **max rate @ N>=1** = the highest solo rate the row sustains, and which budget binds '
             'there. Rates are Hz for step rows and tokens/s for decode rows. Read the verdict off the '
             'column once the rate is decided.', '',
             '| row | latency ms | ' + ' | '.join(f'N@{rate}' for rate in grid)
             + ' | max rate @ N>=1 | binds at max |',
             '|---|---:|' + '---:|' * len(grid) + '---:|---|']
    for row in sweep_rows:
        solo = row['solo']
        lines.append(f"| {row['name']} | {row['latency_ms']:.1f} | "
                     + ' | '.join(str(solo['N_vs_hz'][rate]) for rate in grid)
                     + f" | {solo.get('max_hz_at_N1')} | {solo.get('max_hz_bound_by')} |")
    return lines


def e2e_driver_table(rows):
    driver_rows = [row for row in rows
                   if row.get('kind') == 'e2e' and row['name'].endswith('_e2e')
                   and 'clips' in (row.get('e2e') or {})]
    if not driver_rows:
        return []
    lines = ['', '## End-to-end driver rows', '',
             '| row | clips x repeats | encoder ms | first step ms | TTFT ms | decode ms median / p99 max | '
             'gpu total median / p99 | RTF | speech tok/s |',
             '|---|---:|---:|---:|---:|---|---|---:|---:|']
    for row in driver_rows:
        detail = row['e2e']
        lines.append(f"| {row['name']} | {detail['clips']} x {detail['repeats']} | "
                     f"{detail.get('encoder_ms_median')} | {detail.get('first_step_ms_median')} | "
                     f"{detail['ttft_ms_median']:.1f} | {detail.get('decode_ms_median')} / "
                     f"{detail['decode_ms_p99_max']:.2f} | "
                     f"{detail['gpu_total_ms_median']:.1f} / {detail['gpu_total_ms_p99']:.1f} | "
                     f"{detail.get('rtf_wall_median')} | {detail.get('speech_tokens_per_s')} |")
    return lines


def report_lines(merged, rows, provenance, engine_results_path):
    lines = ['# Stage 4 - solo per-model measurement', '',
             f"device: {provenance.get('device')}  platform: {provenance.get('platform')}  "
             f"tag: {provenance.get('device_tag')}  date: {provenance.get('date')}",
             f"ceilings: {provenance.get('ceilings_json')}",
             f"bw_eff {merged.get('bw_eff_gbps')} GB/s ({merged.get('bw_ceiling_source')})  "
             f"vram cap {merged.get('vram_capacity_mb')} MB ({merged.get('vram_capacity_source')})",
             '', 'N = min(L, C); C = 1/U_max over time, DRAM-bandwidth and VRAM budgets; '
             'L = deadline / latency of record.',
             'cause: latency-limited (L < C, needs faster silicon) vs throughput-limited (C < L, buyable '
             'with capacity).', '']
    lines += verdict_table(rows)
    lines += generative_table(rows)
    lines += rate_sweep_table(rows)
    lines += e2e_driver_table(rows)
    if engine_results_path:
        lines += ['', f'Per-engine detail (nsys, NCU roofline, clock drift): '
                      f'{os.path.dirname(engine_results_path)}/report.md']
    return lines


def load_json_or(path, default):
    if path and os.path.exists(path):
        return json.load(open(path))
    return default


def cmd_merge(args):
    budgets = load_json_or(args.budgets, {'rows': []})
    engine_results = load_json_or(args.engine_results, {})
    provenance = load_json_or(args.provenance, {})
    rows = ([engine_row_summary(model) for model in engine_results.get('models', [])]
            + list(budgets.get('rows', [])))
    merged = merged_results(budgets, engine_results, provenance, rows, args.engine_results)
    json.dump(merged, open(args.out, 'w'), indent=1)
    lines = report_lines(merged, rows, provenance, args.engine_results)
    open(args.report, 'w').write('\n'.join(lines) + '\n')
    print(f"  {len(rows)} rows -> {args.out}")


def add_measurement_flags(parser, *extra):
    """--row first, then the subcommand's own inputs, then the shared --vram/--drift/--out."""
    parser.add_argument('--row', required=True)
    for flag, kwargs in extra:
        parser.add_argument(flag, **kwargs)
    parser.add_argument('--vram', default='')
    parser.add_argument('--drift', default='')
    parser.add_argument('--out', required=True)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='cmd', required=True)

    select = subparsers.add_parser('select')
    select.add_argument('--rows-dir', required=True)
    select.add_argument('--only', action='append', default=[])
    select.add_argument('--kind')
    select.add_argument('--shell', action='store_true')
    select.add_argument('--table', action='store_true')
    select.set_defaults(run=cmd_select)

    mix = subparsers.add_parser('mix')
    mix.add_argument('--rows-dir', required=True)
    mix.add_argument('--only', action='append', default=[])
    mix.add_argument('--out', required=True)
    mix.set_defaults(run=cmd_mix)

    e2e = subparsers.add_parser('e2e')
    add_measurement_flags(e2e, ('--jsonl', {'required': True}))
    e2e.set_defaults(run=cmd_e2e)

    edgellm = subparsers.add_parser('edgellm')
    add_measurement_flags(edgellm, ('--bench-dir', {'required': True}), ('--profile', {'default': ''}))
    edgellm.set_defaults(run=cmd_edgellm)

    trtllm = subparsers.add_parser('trtllm')
    add_measurement_flags(trtllm, ('--sweep', {'required': True}))
    trtllm.set_defaults(run=cmd_trtllm)

    merge = subparsers.add_parser('merge')
    merge.add_argument('--budgets', default='')
    merge.add_argument('--engine-results', default='')
    merge.add_argument('--provenance', default='')
    merge.add_argument('--out', required=True)
    merge.add_argument('--report', required=True)
    merge.set_defaults(run=cmd_merge)

    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
