#!/usr/bin/env python3
"""Row plumbing for measure_models.sh - everything that is parsing, not running.

  select  --rows-dir D [--only m]... [--kind K]        -> JSON list (or --shell lines)
  mix     --rows-dir D [--only m]... --out mix.csv     -> the 7-column mix of engine rows
  e2e     --row R --jsonl F --vram V --out rows.json   -> <model>_e2e + <model>_decode rows
  edgellm --row R --bench-dir D [--profile P] --vram V --out rows.json
  trtllm  --row R --sweep S --vram V --out rows.json
  merge   --budgets B [--engine-results E] --provenance P --out results.json --report report.md

The row dicts written by e2e/edgellm/trtllm are compute_budgets.py inputs:
name, latency_ms (the latency of record), hz, deadline_ms, arch_gflops,
bytes_per_frame_MB (or None + bytes_source), vram_mb, latency_source, plus
the detail block a reader needs to trust the number.
"""
import argparse, glob, json, os, re, shlex, statistics as st, sys


def load_rows(rows_dir, only=(), kind=None):
    rows = []
    for f in sorted(glob.glob(os.path.join(rows_dir, '*.jsonl'))):
        model = os.path.basename(f)[:-6]
        if only and model not in only:
            continue
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if kind and r.get('kind') != kind:
                continue
            rows.append(r)
    return rows


def pctl(v, p):
    v = sorted(v)
    if not v:
        return None
    return v[min(len(v) - 1, int(round(p * (len(v) - 1))))]


def read_vram(path):
    try:
        d = json.load(open(path))
        return d.get('peak_mb'), d.get('source')
    except Exception:
        return None, 'none'


# ---------------------------------------------------------------- select / mix
def cmd_select(a):
    rows = load_rows(a.rows_dir, a.only, a.kind)
    if a.table:
        for r in rows:
            what = {'engine': r.get('engine'), 'e2e': r.get('src'),
                    'generative': r.get('llm_dir') or r.get('serving_dir')}.get(r.get('kind'), '')
            print(f"{r['row']:32} {r.get('kind',''):10} {str(r.get('precision','')):8} "
                  f"{r.get('runtime','tensorrt'):9} hz={r.get('hz')} deadline={r.get('deadline_ms')}  {what}")
    elif a.shell:
        for r in rows:
            print(' '.join(f'{k.upper()}={shlex.quote(str("" if v is None else v))}' for k, v in r.items()
                           if isinstance(v, (str, int, float, bool)) or v is None))
    else:
        print(json.dumps(rows, indent=1))


def cmd_mix(a):
    rows = load_rows(a.rows_dir, a.only, 'engine')
    with open(a.out, 'w') as f:
        f.write('name,onnx,precision,hz,deadline_ms,arch_gflops,extra\n')
        for r in rows:
            f.write(f"{r['row']},{r['engine']},{r['precision']},{r['hz']},{r['deadline_ms']},"
                    f"{r.get('arch_gflops') or 0},{r.get('run_flags') or ''}\n")
    print(len(rows))


# ------------------------------------------------------------------------ e2e
def cmd_e2e(a):
    r = json.load(open(a.row))
    recs = [json.loads(l) for l in open(a.jsonl) if l.strip()]
    if not recs:
        sys.exit(f'{a.jsonl}: the driver wrote no records')
    need = ('ttft_ms', 'decode_ms_p99', 'gpu_total_ms', 'wall_ms')
    miss = [k for k in need if any(k not in x for x in recs)]
    if miss:
        sys.exit(f'{a.jsonl}: records lack required fields {miss} (the e2e contract needs {need})')
    vram, vsrc = read_vram(a.vram)
    tot = [x['gpu_total_ms'] for x in recs]
    secs = [x.get('seconds') for x in recs if x.get('seconds')]
    toks = [x.get('n_tokens') for x in recs if x.get('n_tokens')]
    model = r['model']
    hz = float(r.get('hz') or 0)
    dl = float(r.get('deadline_ms') or r.get('model_deadline_ms') or 0)
    detail = {'clips': len({x.get('clip') for x in recs}), 'records': len(recs), 'repeats': r.get('repeats'),
              'ttft_ms_median': st.median([x['ttft_ms'] for x in recs]),
              'ttft_ms_p99': pctl([x['ttft_ms'] for x in recs], .99),
              'encoder_ms_median': st.median([x['encoder_ms'] for x in recs]) if all('encoder_ms' in x for x in recs) else None,
              'first_step_ms_median': st.median([x['first_step_ms'] for x in recs]) if all('first_step_ms' in x for x in recs) else None,
              'decode_ms_median': st.median([x['decode_ms_median'] for x in recs]) if all('decode_ms_median' in x for x in recs) else None,
              'decode_ms_p99_max': max(x['decode_ms_p99'] for x in recs),
              'gpu_total_ms_median': st.median(tot), 'gpu_total_ms_p99': pctl(tot, .99),
              'wall_ms_median': st.median([x['wall_ms'] for x in recs]),
              'rtf_wall_median': st.median([x['rtf_wall'] for x in recs]) if all('rtf_wall' in x for x in recs) else None,
              'speech_tokens_per_s': (sum(toks) / sum(secs)) if secs and toks and len(secs) == len(toks) else None,
              'jsonl': a.jsonl}
    rows = [{'name': f'{model}_e2e', 'kind': 'e2e', 'precision': r.get('precision'), 'runtime': 'tensorrt-cpp-driver',
             'latency_ms': detail['gpu_total_ms_p99'], 'latency_source': 'e2e-gpu-total-p99',
             'hz': hz, 'deadline_ms': dl, 'arch_gflops': float(r.get('arch_gflops') or 0),
             'bytes_per_frame_MB': None, 'bytes_source': 'none (no per-request byte counter for a multi-engine driver)',
             'vram_mb': vram, 'vram_source': vsrc, 'e2e': detail},
            # the decode loop is the row the per-token deadline judges: p99 of
            # one decode step, charged at the speech-rate token demand
            {'name': f'{model}_decode', 'kind': 'e2e', 'precision': r.get('precision'), 'runtime': 'tensorrt-cpp-driver',
             'latency_ms': detail['decode_ms_p99_max'], 'latency_source': 'e2e-decode-step-p99 (max over clips)',
             'hz': round(detail['speech_tokens_per_s'], 3) if detail['speech_tokens_per_s'] else 0.0,
             'hz_source': 'measured speech token rate (sum tokens / sum clip seconds)',
             'deadline_ms': float(r.get('model_deadline_ms') or dl), 'arch_gflops': 0.0,
             'bytes_per_frame_MB': None, 'bytes_source': 'none',
             'vram_mb': vram, 'vram_source': vsrc,
             'e2e': {'decode_ms_median': detail['decode_ms_median'], 'decode_ms_p99_max': detail['decode_ms_p99_max'],
                     'ttft_ms_median': detail['ttft_ms_median'], 'jsonl': a.jsonl}}]
    json.dump(rows, open(a.out, 'w'), indent=1)
    print(f"  {rows[0]['name']:26} gpu_total p99 {rows[0]['latency_ms']:.3f} ms  ttft {detail['ttft_ms_median']:.1f}  rtf {detail['rtf_wall_median']}")
    print(f"  {rows[1]['name']:26} decode step p99 {rows[1]['latency_ms']:.3f} ms  at {rows[1]['hz']} tok/s")


# -------------------------------------------------------------------- edgellm
def bench_ms(path):
    try:
        m = re.search(r'E2E Time \(actual performance\): ([0-9.]+)', open(path, errors='ignore').read())
        return float(m.group(1)) if m else None
    except FileNotFoundError:
        return None


def cmd_edgellm(a):
    r = json.load(open(a.row))
    d = a.bench_dir
    vis = bench_ms(os.path.join(d, 'visual.log')) if r.get('visual_dir') else 0.0
    pre = bench_ms(os.path.join(d, 'prefill.log'))
    reuse = bench_ms(os.path.join(d, 'reuse.log'))
    dec = bench_ms(os.path.join(d, 'decode.log'))
    if pre is None or dec is None or (r.get('visual_dir') and vis is None):
        sys.exit(f'{d}: llm_bench output incomplete (visual={vis} prefill={pre} reuse={reuse} decode={dec})')
    chunk = int(r.get('chunk') or 8)
    best_pre = min(x for x in (pre, reuse) if x is not None)
    step = (vis or 0) + best_pre + chunk * dec
    vram, vsrc = read_vram(a.vram)
    prof = None
    if a.profile and os.path.exists(a.profile):
        p = json.load(open(a.profile))
        stages = {s['stage_id']: s['gpu_time_stats'] for s in p.get('stages', [])}
        gen = p.get('generation', {})
        prof = {'ttft_ms': sum(stages[k]['median_ms'] for k in ('vision_encoder', 'llm_prefill') if k in stages)
                           + (stages['llm_generation']['median_ms'] if 'llm_generation' in stages else 0),
                'ttft_basis': 'vision_encoder + llm_prefill + first llm_generation step (stage medians)',
                'decode_ms_median': stages.get('llm_generation', {}).get('median_ms'),
                'decode_ms_p99': stages.get('llm_generation', {}).get('p99_ms'),
                'tokens_per_second': gen.get('tokens_per_second'), 'generated_tokens': gen.get('generated_tokens'),
                'prefill_ms_median': stages.get('llm_prefill', {}).get('median_ms'),
                'visual_ms_median': stages.get('vision_encoder', {}).get('median_ms'),
                'peak_unified_memory_mb': p.get('peak_unified_memory_mb'), 'profile': a.profile}
        if p.get('peak_unified_memory_mb'):
            vram = max(vram or 0, p['peak_unified_memory_mb']); vsrc = 'edgellm-profile-peak-unified'
    eb = float(r.get('engine_bytes') or 0) / 1e6; vb = float(r.get('visual_bytes') or 0) / 1e6
    bytes_mb = eb * (chunk + 1) + vb if eb else None
    row = {'name': r['row'], 'kind': 'generative', 'runtime': 'edgellm', 'precision': r['precision'],
           'latency_ms': round(step, 3), 'latency_source': f'llm_bench step = visual + best(prefill,reuse) + {chunk} x decode',
           'hz': float(r['hz']), 'deadline_ms': float(r['deadline_ms']), 'arch_gflops': float(r.get('arch_gflops') or 0),
           'bytes_per_frame_MB': round(bytes_mb, 1) if bytes_mb else None,
           'bytes_source': 'weights-stream-estimate: llm engine bytes x (chunk+1) + visual engine bytes (KV traffic excluded - a lower bound)',
           'vram_mb': vram, 'vram_source': vsrc,
           'generative': {'visual_ms': vis, 'prefill_ms': pre, 'prefill_reuse_ms': reuse, 'decode_ms': dec,
                          'tokens_per_s_bench': round(1000.0 / dec, 2), 'chunk': chunk,
                          'context_len': r.get('context_len'), 'reuse_len': r.get('reuse_len'),
                          'ttft_ms_bench': round((vis or 0) + best_pre + dec, 2),
                          'engine_mb': round(eb, 1), 'visual_mb': round(vb, 1), 'e2e': prof}}
    json.dump([row], open(a.out, 'w'), indent=1)
    print(f"  {row['name']:26} step {step:.1f} ms (vis {vis} + prefill {pre}/{reuse} + {chunk}x{dec})  "
          f"tok/s {1000/dec:.1f}" + (f"  e2e ttft {prof['ttft_ms']:.1f} tok/s {prof['tokens_per_second']:.1f}" if prof else ''))


# --------------------------------------------------------------------- trtllm
def cmd_trtllm(a):
    r = json.load(open(a.row))
    s = json.load(open(a.sweep))
    chunk = str(r.get('chunk') or 8)
    c = (s.get('chunks') or {}).get(chunk) or {}
    comp = s.get('compose') or {}
    if not c.get('total_ms_p99'):
        sys.exit(f'{a.sweep}: no chunk-{chunk} pass with total_ms_p99')
    vram, vsrc = read_vram(a.vram)
    eb = float(r.get('engine_bytes') or 0) / 1e6
    row = {'name': r['row'], 'kind': 'generative', 'runtime': 'trtllm', 'precision': r['precision'],
           'latency_ms': c['total_ms_p99'], 'latency_source': f'RequestPerfMetrics total_ms p99, chunk {chunk} (reuse on)',
           'hz': float(r['hz']), 'deadline_ms': float(r['deadline_ms']), 'arch_gflops': float(r.get('arch_gflops') or 0),
           'bytes_per_frame_MB': round(eb * (int(chunk) + 1), 1) if eb else None,
           'bytes_source': 'weights-stream-estimate: checkpoint bytes x (chunk+1) (KV traffic excluded - a lower bound)',
           'vram_mb': vram, 'vram_source': vsrc,
           'generative': {'ttft_ms': comp.get('ttft_ms'), 'prefill_reuse_ms': comp.get('prefill_reuse_ms'),
                          'prefill_cold_ms': comp.get('prefill_cold_ms'), 'decode_ms': comp.get('decode_ms'),
                          'tokens_per_s_bench': round(1000.0 / comp['decode_ms'], 2) if comp.get('decode_ms') else None,
                          'step_ms_p50': c.get('total_ms_p50'), 'n': c.get('n'), 'chunk': int(chunk), 'sweep': a.sweep}}
    json.dump([row], open(a.out, 'w'), indent=1)
    print(f"  {row['name']:26} step p99 {row['latency_ms']:.1f} ms  ttft {comp.get('ttft_ms')}  decode {comp.get('decode_ms')} ms/tok")


# ---------------------------------------------------------------------- merge
def engine_row_summary(m):
    s = m.get('solo') or {}
    return {'name': m['name'], 'kind': 'engine', 'runtime': 'tensorrt', 'precision': m.get('precision'),
            'latency_ms': m.get('p99_ms'), 'latency_source': m.get('p99_source', 'trtexec p99'),
            'hz': m.get('hz'), 'deadline_ms': m.get('deadline_ms'), 'arch_gflops': m.get('arch_gflops'),
            'bytes_per_frame_MB': m.get('bytes_per_frame_MB'), 'bytes_source': m.get('bytes_source'),
            'vram_mb': m.get('vram_budget_mb') or m.get('vram_mb'), 'vram_source': m.get('vram_source'),
            'solo': s, 'measurement_valid': m.get('measurement_valid')}


def cmd_merge(a):
    B = json.load(open(a.budgets)) if a.budgets and os.path.exists(a.budgets) else {'rows': []}
    E = json.load(open(a.engine_results)) if a.engine_results and os.path.exists(a.engine_results) else {}
    prov = json.load(open(a.provenance)) if a.provenance and os.path.exists(a.provenance) else {}
    rows = [engine_row_summary(m) for m in E.get('models', [])] + list(B.get('rows', []))
    out = {'suite': E.get('suite', 'per-model-solo'), 'convention': E.get('convention'), 'regime': E.get('regime'),
           'composition': 'solo (one row at a time; N/C/L per row, never summed across rows)',
           'device': prov.get('device'), 'platform': prov.get('platform'), 'device_tag': prov.get('device_tag'),
           'provenance': prov,
           'preflight': E.get('preflight'), 'clock_integrity': E.get('clock_integrity'),
           'ceilings': E.get('ceilings'), 'bw_ceiling_source': E.get('bw_ceiling_source') or B.get('bw_ceiling_source'),
           'bw_eff_gbps': B.get('bw_eff_gbps'),
           'vram_capacity_mb': E.get('vram_capacity_mb') or B.get('vram_capacity_mb'),
           'vram_capacity_source': E.get('vram_capacity_source') or B.get('vram_capacity_source'),
           'engine_rows_results': a.engine_results,
           'rows': rows,
           # the per-engine block keeps its full schema for the existing readers
           'models': E.get('models', []) + [dict(r, p99_ms=r.get('latency_ms'), p99_source=r.get('latency_source'))
                                             for r in B.get('rows', [])]}
    for k in ('U_max', 'C', 'L', 'N', 'shortfall_cause_if_N_lt_1', 'workload_constant_gflops_per_s', 'score_tflops_at_deadline',
              'budgets', 'binding_budget', 'bw_demand_is_upper_bound'):
        if k in E:
            out[k] = E[k]
    if 'N' in E:
        out['device_rollup_scope'] = 'engine rows only (the trtexec mix); generative and e2e rows are solo verdicts'
    json.dump(out, open(a.out, 'w'), indent=1)

    L = ['# Stage 4 - solo per-model measurement', '',
         f"device: {prov.get('device')}  platform: {prov.get('platform')}  tag: {prov.get('device_tag')}  "
         f"date: {prov.get('date')}", f"ceilings: {prov.get('ceilings_json')}",
         f"bw_eff {out.get('bw_eff_gbps')} GB/s ({out.get('bw_ceiling_source')})  vram cap {out.get('vram_capacity_mb')} MB ({out.get('vram_capacity_source')})",
         '', 'N = min(L, C); C = 1/U_max over time, DRAM-bandwidth and VRAM budgets; L = deadline / latency of record.',
         'cause: latency-limited (L < C, needs faster silicon) vs throughput-limited (C < L, buyable with capacity).', '',
         '| row | kind | prec | latency ms | source | hz | deadline | U_max | C | L | N | cause | budget |',
         '|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|']
    for r in rows:
        s = r.get('solo') or {}
        if not s:
            L.append(f"| {r['name']} | {r.get('kind')} | {r.get('precision')} | {r.get('latency_ms')} | {r.get('latency_source','')} | "
                     f"{r.get('hz')} | {r.get('deadline_ms')} | - | - | - | - | {r.get('error','unscored')} | |")
            continue
        comp = 'complete' if s.get('budget_complete', True) else 'incomplete: ' + ','.join(s.get('budgets_not_measured', []))
        if r.get('measurement_valid') is False:
            comp += ' INVALID(drift)'
        L.append(f"| {r['name']} | {r.get('kind')} | {r.get('precision')} | {r.get('latency_ms'):.3f} | {r.get('latency_source','')} | "
                 f"{r.get('hz')} | {r.get('deadline_ms')} | {s.get('U_max')} | {s.get('C')} | {s.get('L')} | **{s.get('N')}** | {s.get('cause')} | {comp} |")
    gen = [r for r in rows if r.get('kind') == 'generative']
    if gen:
        L += ['', '## Generative rows (TTFT / decode)', '',
              '| row | runtime | prec | visual ms | prefill ms | reuse ms | decode ms/tok | tok/s | TTFT ms | e2e TTFT ms | e2e tok/s | e2e decode p99 | peak mem MB |',
              '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in gen:
            g = r.get('generative') or {}; e = g.get('e2e') or {}
            L.append(f"| {r['name']} | {r.get('runtime')} | {r.get('precision')} | {g.get('visual_ms')} | {g.get('prefill_ms') or g.get('prefill_cold_ms')} | "
                     f"{g.get('prefill_reuse_ms')} | {g.get('decode_ms')} | {g.get('tokens_per_s_bench')} | {g.get('ttft_ms_bench') or g.get('ttft_ms')} | "
                     f"{e.get('ttft_ms')} | {e.get('tokens_per_second')} | {e.get('decode_ms_p99')} | {r.get('vram_mb')} |")
    e2e = [r for r in rows if r.get('kind') == 'e2e' and r['name'].endswith('_e2e')]
    if e2e:
        L += ['', '## End-to-end driver rows', '',
              '| row | clips x repeats | encoder ms | first step ms | TTFT ms | decode ms median / p99 max | gpu total median / p99 | RTF | speech tok/s |',
              '|---|---:|---:|---:|---:|---|---|---:|---:|']
        for r in e2e:
            d = r['e2e']
            L.append(f"| {r['name']} | {d['clips']} x {d['repeats']} | {d.get('encoder_ms_median')} | {d.get('first_step_ms_median')} | "
                     f"{d['ttft_ms_median']:.1f} | {d.get('decode_ms_median')} / {d['decode_ms_p99_max']:.2f} | "
                     f"{d['gpu_total_ms_median']:.1f} / {d['gpu_total_ms_p99']:.1f} | {d.get('rtf_wall_median')} | {d.get('speech_tokens_per_s')} |")
    if a.engine_results:
        L += ['', f'Per-engine detail (nsys, NCU roofline, clock drift): {os.path.dirname(a.engine_results)}/report.md']
    open(a.report, 'w').write('\n'.join(L) + '\n')
    print(f"  {len(rows)} rows -> {a.out}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('select'); p.add_argument('--rows-dir', required=True); p.add_argument('--only', action='append', default=[])
    p.add_argument('--kind'); p.add_argument('--shell', action='store_true'); p.add_argument('--table', action='store_true'); p.set_defaults(f=cmd_select)
    p = sp.add_parser('mix'); p.add_argument('--rows-dir', required=True); p.add_argument('--only', action='append', default=[])
    p.add_argument('--out', required=True); p.set_defaults(f=cmd_mix)
    p = sp.add_parser('e2e'); p.add_argument('--row', required=True); p.add_argument('--jsonl', required=True)
    p.add_argument('--vram', default=''); p.add_argument('--out', required=True); p.set_defaults(f=cmd_e2e)
    p = sp.add_parser('edgellm'); p.add_argument('--row', required=True); p.add_argument('--bench-dir', required=True)
    p.add_argument('--profile', default=''); p.add_argument('--vram', default=''); p.add_argument('--out', required=True); p.set_defaults(f=cmd_edgellm)
    p = sp.add_parser('trtllm'); p.add_argument('--row', required=True); p.add_argument('--sweep', required=True)
    p.add_argument('--vram', default=''); p.add_argument('--out', required=True); p.set_defaults(f=cmd_trtllm)
    p = sp.add_parser('merge'); p.add_argument('--budgets', default=''); p.add_argument('--engine-results', default='')
    p.add_argument('--provenance', default=''); p.add_argument('--out', required=True); p.add_argument('--report', required=True); p.set_defaults(f=cmd_merge)
    a = ap.parse_args()
    a.f(a)


if __name__ == '__main__':
    main()
