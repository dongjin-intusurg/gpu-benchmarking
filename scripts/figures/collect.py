#!/usr/bin/env python3
"""Stage 7.0: read the newest stage 3-6 runs of one device into a single data.json.

    collect.py --device <cfg.json> [--ceilings <raw/results.json>] [--solo <results.json>]
               [--coloc <run dir>] [--power <run dir>] --root <repo root> --out data.json

Every block is optional: a missing stage yields null and a note in 'missing'.
The output is the only input of render.py and report.py, and it is written to
be diffable: sorted keys, floats rounded to six significant digits, NaN -> null,
dates confined to 'sources'. Numbers are derived with the same helpers the
stages use (bandwidth denominator, floors, N bounds); nothing is re-modelled.
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.dirname(HERE)
for sub in ('model_bench', 'colocation', 'device_ceilings'):
    sys.path.insert(0, os.path.join(KIT, sub))

from floors import load_device, t_floor, n_at, CEIL_KEY  # noqa: E402
from compose_mix import HEADROOM  # noqa: E402
from compute_budgets import bw_eff_from_ceilings  # noqa: E402
from validate_ceilings import DATASHEET_KEYS, load_json, trt_best  # noqa: E402,F401
from write_ceilings_report import (sweep_points, sustained_throughputs, best_over_buffers,  # noqa: E402
                                   same_size_burst, sustained_metric, haircut_worker_key, num)

SCHEMA = 'figures/v1'
PRECISIONS = ('fp16', 'int8', 'fp8', 'fp32')
TORCH_METRIC = {'fp16': 'fp16_fp16acc_tflops', 'int8': 'int8_tops'}
UNIT = {'fp16': 'TFLOPS', 'int8': 'TOPS', 'fp8': 'TFLOPS', 'fp32': 'TFLOPS'}
TRACE_POINTS = 300
TOP_KERNELS = 10
MAX_KERNELS = 150
BANDWIDTH_KERNELS = ('copy_RW', 'read_only', 'write_only', 'triad_2R1W')


# ----------------------------------------------------------------------------- helpers
def clean(obj):
    """Round floats to 6 significant digits, NaN/inf -> None, -0.0 -> 0.0, recursively."""
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        value = float(f'{obj:.6g}')
        return 0.0 if value == 0 else value
    if isinstance(obj, int):
        return obj
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    return obj


def ratio(a, b):
    a, b = num(a), num(b)
    return a / b if a is not None and b else None


def median(values):
    values = sorted(values)
    if not values:
        return None
    n = len(values)
    return values[n // 2] if n % 2 else 0.5 * (values[n // 2 - 1] + values[n // 2])


def decimate(samples, limit):
    if len(samples) <= limit:
        return list(samples)
    step = len(samples) / float(limit)
    return [samples[int(i * step)] for i in range(limit)]


def relative(path, root):
    if not path:
        return None
    path = os.path.abspath(path)
    try:
        rel = os.path.relpath(path, root)
    except ValueError:
        rel = path
    return rel if not rel.startswith('..') else os.path.join(*path.split(os.sep)[-3:])


def same_file(a, b):
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.abspath(a or '') == os.path.abspath(b or '')


def short_status(status):
    """Statuses may embed log paths in braces; keep the sentence only."""
    if not isinstance(status, str):
        return status
    return status.split('{', 1)[0].strip().rstrip(':').strip()


def pick(d, keys):
    return {k: d.get(k) for k in keys if isinstance(d, dict)}


# ----------------------------------------------------------------------------- device
def device_block(cfg_path):
    cfg = load_json(cfg_path) or {}
    ds = cfg.get('datasheet') or {}
    datasheet = {p: num(ds.get(DATASHEET_KEYS[p])) for p in PRECISIONS}
    datasheet.update({'bw_gbps': num(ds.get('peak_bw_gbps')), 'tdp_w': num(ds.get('tdp_w')),
                      'source_url': ds.get('source_url'), 'retrieved': ds.get('retrieved')})
    return {
        'tag': cfg.get('device_tag'),
        'platform': cfg.get('platform'),
        'name': cfg.get('device_name_match') or cfg.get('device_id'),
        'module': cfg.get('module'),
        'config': os.path.basename(cfg_path),
        'datasheet': datasheet,
        'vram_physical_mb': num(cfg.get('vram_physical_mb')),
        'vram_budget_cap_mb': num(cfg.get('vram_budget_cap_mb')),
        'lock_targets_mhz': cfg.get('lock_targets_mhz'),
        'power_envelope_w': cfg.get('power_envelope_w'),
    }


# ----------------------------------------------------------------------------- ceilings
def regime_block(run_dir):
    """Preflight / lock / drift verdicts of a run directory, when present."""
    out = {}
    for name, path in (('preflight', 'provenance/preflight.json'),
                       ('lock', 'provenance/lock_verified.json'),
                       ('drift', 'clock_drift.json')):
        doc = load_json(os.path.join(run_dir, path)) if run_dir else None
        out[name] = doc.get('verdict') if isinstance(doc, dict) else None
        if name == 'preflight' and isinstance(doc, dict):
            out['smoke_only'] = doc.get('smoke_only')
    return out


def ceilings_block(raw_path, device):
    raw = load_json(raw_path)
    if not raw:
        return None, None, None
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(raw_path)))
    trt = load_json(os.path.join(os.path.dirname(raw_path), 'trt_compute.json'))
    gemm = raw.get('gemm_sweep') or {}
    ds = device['datasheet']
    sust_prec = (raw.get('meta') or {}).get('sustain_prec')
    sust_values = sustained_throughputs(raw)
    sust_median = median(sust_values)

    precisions = {}
    for p in PRECISIONS:
        entry = {'unit': UNIT[p], 'datasheet': ds.get(p)}
        if trt and isinstance(trt.get('precisions'), dict) and p in trt['precisions']:
            tp = trt['precisions'][p]
            entry['trt_best'] = num(tp.get('best'))
            entry['trt_at_n'] = tp.get('at_n')
            entry['trt_method'] = tp.get('best_method')
            entry['trt_sweep'] = [{'n': pt.get('n'), 'value': num(pt.get('tflops', pt.get('tops')))}
                                  for pt in (tp.get('points') or [])
                                  if isinstance(pt, dict) and pt.get('ok')]
        if p in TORCH_METRIC:
            pts = sweep_points(gemm, TORCH_METRIC[p])
            if pts:
                best_n, best_pt = max(pts, key=lambda np_: num(np_[1]['best']))
                entry['torch_burst_best'] = num(best_pt['best'])
                entry['torch_burst_at_n'] = best_n
                entry['torch_sweep'] = [{'n': n, 'best': num(pt.get('best')), 'median': num(pt.get('median')),
                                         'spread_clean_pct': num(pt.get('spread_clean_pct')),
                                         'clamped_reps': pt.get('clamped_reps')} for n, pt in pts]
            entry['torch_burst_same_n'] = same_size_burst(gemm, TORCH_METRIC[p])
        elif p == 'fp8':
            entry['torch_burst_best'] = entry['torch_burst_same_n'] = num(gemm.get('fp8_n4096_tflops'))
            entry['torch_burst_at_n'] = 4096 if entry['torch_burst_best'] is not None else None
        elif p == 'fp32':
            entry['torch_burst_best'] = entry['torch_burst_same_n'] = num(raw.get('cuda_fp32'))
        if p == sust_prec:
            entry['torch_sustained_median'] = sust_median
        entry['attain_trt'] = ratio(entry.get('trt_best'), ds.get(p))
        entry['attain_burst'] = ratio(entry.get('torch_burst_same_n'), ds.get(p))
        entry['attain_sustained'] = ratio(entry.get('torch_sustained_median'), ds.get(p))
        precisions[p] = entry

    shaped = {k: num(v) for k, v in (raw.get('shaped') or {}).items() if num(v) is not None}
    bandwidth = {}
    for kernel in BANDWIDTH_KERNELS:
        best, size = best_over_buffers(raw.get('bandwidth') or {}, kernel)
        if best is not None:
            bandwidth[kernel] = {'best_gbps': best, 'at_buffer': size, 'frac_datasheet': ratio(best, ds.get('bw_gbps'))}
    bw_by_buffer = {}
    for size, row in (raw.get('bandwidth') or {}).items():
        if isinstance(row, dict):
            bw_by_buffer[size] = {k: num(row.get(k)) for k in BANDWIDTH_KERNELS if num(row.get(k)) is not None}
    bw_eff, bw_source = bw_eff_from_ceilings(raw_path)
    haircut = {}
    for key in sorted((raw.get('haircut') or {}), key=haircut_worker_key):
        value = num((raw.get('haircut') or {})[key])
        if value is not None:
            haircut[str(haircut_worker_key(key))] = value

    sustained = None
    if sust_values:
        samples = [s for s in raw.get('sustained') if isinstance(s, dict)]
        trace = [{'t': num(s.get('t')), 'tflops': num(s.get('tflops')), 'gpu_mhz': num(s.get('gpu_mhz')),
                  'temp_c': num(s.get('max_temp_c'))} for s in decimate(samples, TRACE_POINTS)]
        burst_same_n = same_size_burst(gemm, sustained_metric(sust_prec))
        mhz = [num(s.get('gpu_mhz')) for s in samples if num(s.get('gpu_mhz')) is not None]
        temps = [num(s.get('max_temp_c')) for s in samples if num(s.get('max_temp_c')) is not None]
        sustained = {'precision': sust_prec, 'unit': UNIT.get(sust_prec, 'TFLOPS'),
                     'median': sust_median, 'min': min(sust_values), 'max': max(sust_values),
                     'n_samples': len(sust_values), 'duration_s': num(samples[-1].get('t')),
                     'burst_same_n': burst_same_n, 'sustained_over_burst': ratio(sust_median, burst_same_n),
                     'gpu_mhz_min': min(mhz) if mhz else None, 'temp_c_max': max(temps) if temps else None,
                     'trace': trace}

    meta = raw.get('meta') or {}
    block = {
        'precisions': precisions,
        'shaped_gemm_tflops': shaped,
        'bandwidth': bandwidth,
        'bandwidth_by_buffer': bw_by_buffer,
        'bw_eff_gbps': bw_eff,
        'bw_eff_source': bw_source,
        'haircut_gbps_by_workers': haircut,
        'sustained': sustained,
        'regime': regime_block(run_dir),
        'torch': meta.get('torch'),
        'gpu_mhz_start': num(meta.get('gpu_freq_mhz_start')),
    }
    return block, run_dir, meta.get('start')


# ----------------------------------------------------------------------------- solo
def floor_block(model, dev):
    """Latency floor and N ceiling of one row, with a note that says what bounded it."""
    bytes_mb, arch, ceiling = model.get('bytes_mb'), model.get('arch_gflops'), model.get('comp_ceiling')
    t_mem = bytes_mb / dev['bw'] if bytes_mb and dev.get('bw') else None
    t_comp = arch / ceiling if arch and ceiling else None
    if t_mem is None and t_comp is None:
        return {'t_floor_ms': None, 'bound_by': None, 'note': 'no floor: neither bytes/frame nor a compute term'}
    t_ms, _ = t_floor(model, dev)
    if t_comp is None:
        bound = 'memory'
        note = ('memory-only floor (architectural FLOPs not measured for this row)' if not arch
                else f"memory-only floor (no compute ceiling measured for {model.get('precision')})")
    elif t_mem is None:
        bound, note = 'compute', 'compute-only floor (bytes/frame not measured)'
    else:
        bound = 'memory' if t_mem >= t_comp else 'compute'
        note = f'{bound}-bound floor'
    L, C, N = n_at(t_ms, model, dev) if t_ms else (None, None, None)
    return {'t_floor_ms': t_ms, 't_mem_ms': t_mem, 't_comp_ms': t_comp, 'bound_by': bound, 'note': note,
            'comp_ceiling': ceiling, 'comp_ceiling_key': CEIL_KEY.get(model.get('precision')),
            'L': L, 'C': C, 'N_ceiling': N}


def kernels_block(solo_dir, roofline):
    detail = (roofline or {}).get('detail')
    if not detail:
        return None
    path = detail if os.path.isabs(detail) else os.path.join(solo_dir, 'engine_rows', detail)
    doc = load_json(path)
    kernels = (doc or {}).get('kernels') if isinstance(doc, dict) else None
    if not kernels:
        return None
    kernels = sorted((k for k in kernels if isinstance(k, dict)), key=lambda k: -(num(k.get('us')) or 0))
    return [{'label': k.get('label'), 'us': num(k.get('us')), 'intensity': num(k.get('intensity')),
             'achieved': num(k.get('achieved_tops')), 'roof': num(k.get('roof_tops')),
             'pct_of_roof': num(k.get('pct_of_roof')), 'pipe': k.get('pipe'), 'bound': k.get('bound')}
            for k in kernels[:MAX_KERNELS]]


def deadline_kind(row):
    """spec: the mix fixes hz and deadline; placeholder: a rate grid says the rate is open; none: hz 0."""
    if not row.get('hz'):
        return 'none'
    if row.get('hz_grid') or ((row.get('solo') or {}).get('N_vs_hz')):
        return 'placeholder'
    return 'spec'


EXTRA_KEYS = ('tokens_per_s_bench', 'ttft_ms_bench', 'decode_ms', 'prefill_ms', 'prefill_reuse_ms', 'visual_ms',
              'chunk', 'context_len', 'engine_mb', 'ttft_ms_median', 'ttft_ms_p99', 'decode_ms_median',
              'decode_ms_p99_max', 'gpu_total_ms_median', 'gpu_total_ms_p99', 'rtf_wall_median',
              'speech_tokens_per_s', 'clips')


def solo_row(row, model, dev_model, dev, solo_dir, kernels):
    name = row.get('name')
    solo = row.get('solo') or {}
    budgets = solo.get('budgets') or {}
    p99 = num(model.get('p99_ms')) or num(row.get('latency_ms'))
    deadline = num(row.get('deadline_ms'))
    n_vs_hz = solo.get('N_vs_hz')
    integrity = model.get('clock_integrity') if isinstance(model.get('clock_integrity'), dict) else {}
    entry = {
        'name': name, 'kind': row.get('kind'), 'runtime': row.get('runtime'), 'precision': row.get('precision'),
        'hz': num(row.get('hz')), 'hz_source': row.get('hz_source'), 'deadline_ms': deadline,
        'deadline_kind': deadline_kind(row),
        'p99_ms': p99, 'mean_ms': num(model.get('mean_ms')), 'latency_source': row.get('latency_source'),
        'arch_gflops': num(row.get('arch_gflops')),
        'bytes_MB': num(row.get('bytes_per_frame_MB')), 'bytes_source': row.get('bytes_source'),
        'bytes_physical_bound_MB': num(model.get('bytes_physical_bound_MB')),
        'vram_mb': num(row.get('vram_mb')), 'vram_source': row.get('vram_source'),
        'valid': row.get('measurement_valid'), 'clock_verdict': integrity.get('verdict'),
        'budgets': {'time': num(budgets.get('time_occupancy')), 'bw': num(budgets.get('dram_bandwidth')),
                    'vram': num(budgets.get('vram_footprint'))},
        'budgets_not_measured': solo.get('budgets_not_measured') or [],
        'U_max': num(solo.get('U_max')), 'C': num(solo.get('C')), 'L': num(solo.get('L')), 'N': num(solo.get('N')),
        'cause': solo.get('cause'), 'score_tflops': num(solo.get('score_tflops')),
        'modal': bool(solo.get('modal')), 'error': row.get('error'),
        'meets_deadline': (p99 <= deadline) if (p99 and deadline) else None,
        'N_vs_hz': sorted(([float(k), num(v)] for k, v in n_vs_hz.items()), key=lambda kv: kv[0]) if n_vs_hz else None,
        'max_hz_at_N1': num(solo.get('max_hz_at_N1')), 'max_hz_bound_by': solo.get('max_hz_bound_by'),
    }
    if dev_model:
        entry['floor'] = floor_block(dev_model, dev)
    roofline = model.get('roofline')
    if isinstance(roofline, dict):
        entry['roofline'] = {
            'time_pct_by_bound': roofline.get('time_pct_by_bound'),
            'time_pct_by_pipe': roofline.get('time_pct_by_pipe'),
            'byte_source': roofline.get('byte_source'),
            'ridge_ops_per_byte': (roofline.get('ceilings_used') or {}).get('ridge_ops_per_byte'),
            'top_kernels': [pick(k, ('label', 'us', 'time_pct', 'pipe', 'bound', 'pct_of_roof', 'intensity'))
                            for k in (roofline.get('top_kernels') or [])[:TOP_KERNELS]],
        }
        ks = kernels_block(solo_dir, roofline)
        if ks:
            kernels[name] = ks
    for extra in ('generative', 'e2e'):
        block = row.get(extra)
        if isinstance(block, dict):
            entry[extra] = {k: v for k, v in pick(block, EXTRA_KEYS).items() if v is not None}
    return entry


def solo_block(solo_path):
    doc = load_json(solo_path)
    if not doc:
        return None, None, None, None
    solo_dir = os.path.dirname(os.path.abspath(solo_path))
    dev = load_device(solo_path)
    dev_models = {m['name']: m for m in dev['models']}
    models = {m.get('name'): m for m in (doc.get('models') or []) if isinstance(m, dict)}
    rows, kernels = [], {}
    for row in sorted((r for r in doc.get('rows') or [] if isinstance(r, dict)), key=lambda r: r.get('name') or ''):
        name = row.get('name')
        rows.append(solo_row(row, models.get(name) or {}, dev_models.get(name), dev, solo_dir, kernels))

    ceilings = doc.get('ceilings') or {}
    integrity = doc.get('clock_integrity') or {}
    rollup = {k: doc.get(k) for k in ('U_max', 'C', 'L', 'N', 'binding_budget', 'budgets', 'score_tflops_at_deadline',
                                      'workload_constant_gflops_per_s', 'shortfall_cause_if_N_lt_1', 'device_rollup_scope')}
    ceilings_used = {k: v for k, v in ceilings.items() if isinstance(v, (int, float, str)) and k != 'ceilings_source'}
    provenance = doc.get('provenance') or {}
    block = {
        'bw_eff_gbps': num(doc.get('bw_eff_gbps')), 'bw_ceiling_source': doc.get('bw_ceiling_source'),
        'vram_capacity_mb': num(doc.get('vram_capacity_mb')), 'vram_capacity_source': doc.get('vram_capacity_source'),
        'run_valid': integrity.get('run_valid'), 'lock_verify': integrity.get('lock_verify'),
        'worst_model_verdict': integrity.get('worst_model_verdict'), 'excluded_models': integrity.get('excluded_models') or [],
        'regime': doc.get('regime'), 'ceilings_used': ceilings_used,
        'ceilings_source': (ceilings.get('ceilings_source') or {}).get('path'),
        'rollup': rollup, 'headroom': HEADROOM,
        'rows': rows, 'kernels': kernels,
    }
    return block, solo_dir, provenance.get('date'), provenance.get('device_config')


# ----------------------------------------------------------------------------- co-location
ROW_KEYS = ('hz', 'deadline_ms', 'stage4_solo_p99_ms', 'paced_solo_p99_ms', 'paced_solo_p50_ms', 'p50_ms', 'p99_ms',
            'max_ms', 'achieved_hz', 'contention_factor', 'p99_contended_pred', 'pred_error_frac', 'miss_frac',
            'overrun_frac', 'done_before_next_trigger_frac', 'fits', 'status', 'n')
SIDE_KEYS = ('runtime', 'status', 'fits', 'L_contended', 'L_basis', 'summary', 'solo_summary')
COMPOSED_KEYS = ('U_time', 'U_bw', 'U_vram', 'U_max', 'binding_budget', 'C', 'L', 'L_frame', 'N', 'cause',
                 'units_at_0.65', 'units_at_1.0', 'fits_one_device_at_0.65', 'budget_complete', 'sum_frame_p99_ms',
                 'period_ms')
COMPOSED_ROW_KEYS = ('p99_ms', 'hz', 'deadline_ms', 'bytes_per_frame_MB', 'vram_mb', 'time_share', 'bw_share',
                     'vram_share', 'L', 'solo_N', 'solo_cause', 'p99_contended_pred', 'others_time_share',
                     'meets_deadline_pred')


def coloc_cell(c):
    mk = c.get('makespan') if isinstance(c.get('makespan'), dict) else {}
    rows = {}
    for name, r in sorted((c.get('rows') or {}).items()):
        if isinstance(r, dict):
            rows[name] = dict(pick(r, ROW_KEYS), unfit=r.get('unfit') or [])
    sides = {}
    for name, s in sorted((c.get('side_rows') or {}).items()):
        if isinstance(s, dict):
            vs = {m: pick(v, ('contended', 'solo', 'ratio', 'basis'))
                  for m, v in sorted((s.get('vs_solo') or {}).items()) if isinstance(v, dict)}
            sides[name] = dict(pick(s, SIDE_KEYS), vs_solo=vs)
    return {
        'mix': c.get('mix'), 'arm': c.get('arm'), 'status': short_status(c.get('status')),
        'valid': c.get('valid'), 'fits': c.get('fits'),
        'invalid_reasons': [short_status(x) for x in c.get('invalid_reasons') or []],
        'unfit_reasons': c.get('unfit_reasons') or [],
        'period_ms': num(c.get('period_ms')), 'makespan_p99_ms': num(mk.get('makespan_p99_ms')),
        'makespan_p50_ms': num(mk.get('makespan_p50_ms')), 'makespan_basis': mk.get('makespan_basis'),
        'aligned': mk.get('aligned'), 'all_rows_done_frac': num(mk.get('all_rows_done_in_period_frac')),
        'overlap_frac_p50': num(mk.get('overlap_frac_of_period_p50')),
        'makespan_p99_over_period': num(c.get('makespan_p99_over_period')),
        'sum_paced_solo_p99_ms': num(c.get('sum_paced_solo_p99_ms')), 'free_time_ms': num(c.get('free_time_ms')),
        'N_measured': num(c.get('N_measured')), 'N_predicted': num(c.get('N_predicted')),
        'L_contended': num(c.get('L_contended')), 'C_composed': num(c.get('C_composed')),
        'cause': c.get('cause'), 'worst_row': c.get('worst_row'),
        'worst_contention_factor': num(c.get('worst_contention_factor')), 'drift': c.get('drift'),
        'rows': rows, 'side_rows': sides,
    }


def coloc_composed(run_dir, mixes):
    out = {}
    for mix in sorted(mixes or []):
        doc = load_json(os.path.join(run_dir, 'composed', f'{mix}.json'))
        if not isinstance(doc, dict):
            continue
        comp = doc.get('composed') or {}
        rows = []
        for r in doc.get('rows') or []:
            row = {'row': r.get('row'), 'role': r.get('role'), 'kind': r.get('kind'), 'precision': r.get('precision')}
            row.update(pick(r.get('metrics') or {}, COMPOSED_ROW_KEYS))
            row['incomplete'] = (r.get('metrics') or {}).get('incomplete') or []
            rows.append(row)
        out[mix] = dict(pick(comp, COMPOSED_KEYS), rows=rows, budgets_not_measured=comp.get('budgets_not_measured') or [])
    return out


def coloc_block(run_dir, kind):
    doc = load_json(os.path.join(run_dir, 'verdict.json'))
    if not doc:
        return None, None, None
    cells = sorted((c for c in doc.get('cells') or [] if isinstance(c, dict)),
                   key=lambda c: (c.get('mix') or '', c.get('arm') or ''))
    block = {
        'kind': kind, 'arms': doc.get('arms') or [], 'mixes': sorted(doc.get('mixes') or []),
        'run_seconds': num(doc.get('run_seconds')), 'rules': doc.get('rules'), 'lock_verified': doc.get('lock_verified'),
        'cells': [coloc_cell(c) for c in cells], 'composed': coloc_composed(run_dir, doc.get('mixes')),
    }
    return block, doc.get('date'), doc.get('stage4_results')


# ----------------------------------------------------------------------------- power
PACED_KEYS = ('row', 'hz', 'p50_ms', 'p99_ms', 'miss_frac', 'frames', 'wall_s', 'mean_w', 'vdd_gpu_mean_w', 'clock_mhz_min',
              'oc_events', 'throttled_frac', 'temp_c_max', 'j_per_frame', 'j_per_frame_marginal', 'drift')
CELL_KEYS = ('mix', 'arm', 'valid', 'fits', 'makespan_p99_ms', 'all_done_frac', 'N_measured', 'N_predicted', 'worst_row',
             'mean_w', 'vdd_gpu_mean_w', 'oc_events', 'throttled_frac', 'j_per_period', 'drift')
CURVE_KEYS = {'busy_pct': 'busy_pct', 'target_pct': 'target_pct', 'delivered': 'delivered_tops', 'kernel': 'kernel_tops',
              'module_w': 'module_w', 'vdd_gpu_w': 'vdd_gpu_w', 'gpu_mhz': 'gpu_mhz', 'gpu_mhz_min': 'gpu_mhz_min',
              'tj_c': 'tj_c', 'oc_events': 'oc_events', 'per_w': 'tops_per_w', 'per_w_above_idle': 'tops_per_w_above_idle'}


def power_curves(run_dir, point):
    doc = load_json(os.path.join(run_dir, point, 'per_watt', 'power_tops_sweep.json'))
    if not isinstance(doc, dict):
        return None
    out = {}
    for prec, block in sorted((doc.get('precisions') or {}).items()):
        pts = [pt for pt in (block or {}).get('points') or [] if isinstance(pt, dict)]
        out[prec] = {'unit': block.get('unit'),
                     'points': [{k: num(pt.get(src)) for k, src in CURVE_KEYS.items()} for pt in pts]}
    return {'n': doc.get('n'), 'idle_module_w': num(doc.get('idle_module_w')), 'precisions': out}


def power_point(p):
    per_watt = p.get('per_watt') or {}
    precs = {}
    for prec, b in sorted((per_watt.get('precisions') or {}).items()):
        precs[prec] = {'unit': b.get('unit'), 'fit': b.get('fit'), 'fit_unsaturated': b.get('fit_unsaturated'),
                       'knee': b.get('knee'), 'at50': b.get('at50'), 'at100': b.get('at100')}
    cells = {}
    for key, v in sorted((p.get('cells') or {}).items()):
        if isinstance(v, dict):
            cells[key] = dict(pick(v, CELL_KEYS),
                              invalid_reasons=[short_status(x) for x in v.get('invalid_reasons') or []],
                              rows={rn: pick(rv, ('p99_ms', 'miss_frac')) for rn, rv in sorted((v.get('rows') or {}).items())})
    return {
        'status': p.get('status'), 'why': short_status(p.get('why')), 'applied_mode': p.get('applied_mode'),
        'lock_targets_mhz': p.get('lock_targets_mhz'), 'lock_verdict': p.get('lock_verdict'),
        'preflight_verdict': p.get('preflight_verdict'), 'reference_clock_mhz': p.get('reference_clock_mhz'),
        'idle_w': num(p.get('idle_w')), 'idle_w_expected': num(p.get('idle_w_expected')),
        'paced_solo': {k: pick(v, PACED_KEYS) for k, v in sorted((p.get('paced_solo') or {}).items()) if isinstance(v, dict)},
        'cells': cells,
        'per_watt': {'idle_module_w': num(per_watt.get('idle_module_w')), 'n': per_watt.get('n'), 'precisions': precs},
        'vs_baseline': p.get('vs_baseline'),
    }


def power_block(run_dir):
    doc = load_json(os.path.join(run_dir, 'verdict.json'))
    if not doc:
        return None, None, None, None
    prov = load_json(os.path.join(run_dir, 'provenance', 'provenance.json')) or {}
    order = doc.get('points_order') or sorted(doc.get('points') or {})
    points = {name: power_point((doc.get('points') or {}).get(name) or {}) for name in order}
    curves = {name: power_curves(run_dir, name) for name in order}
    block = {
        'schema': doc.get('schema'), 'verdict': doc.get('verdict'), 'baseline': doc.get('baseline'),
        'points_order': order, 'rules': doc.get('rules'), 'checks': doc.get('checks') or [],
        'run_seconds': num(prov.get('run_seconds')), 'sweep_args': prov.get('sweep_args'),
        'points': points, 'curves': curves,
    }
    return block, doc.get('date'), prov.get('stage4_results'), prov.get('ceilings')


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--device', required=True, help='device config json')
    ap.add_argument('--ceilings', help='stage-3 raw/results.json')
    ap.add_argument('--solo', help='stage-4 results.json')
    ap.add_argument('--coloc', help='stage-5 run directory (verdict.json inside)')
    ap.add_argument('--coloc-kind', default='standalone', help="'standalone' or 'power-baseline'")
    ap.add_argument('--power', help='stage-6 run directory (verdict.json inside)')
    ap.add_argument('--root', default=os.path.dirname(KIT), help='paths in sources are written relative to this')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    root = os.path.abspath(args.root)

    device = device_block(args.device)
    sources, checks, missing = {}, [], []
    doc = {'schema': SCHEMA, 'device': device, 'headroom': HEADROOM,
           'ceilings': None, 'solo': None, 'coloc': None, 'power': None}

    if args.ceilings and os.path.exists(args.ceilings):
        doc['ceilings'], run_dir, start = ceilings_block(args.ceilings, device)
    if doc['ceilings']:
        sources['ceilings'] = {'path': relative(args.ceilings, root), 'run': os.path.basename(run_dir),
                               'date': start, 'kind': 'thorough'}
    else:
        missing.append('ceilings (stage 3): run ./scripts/run_ceilings.sh')

    if args.solo and os.path.exists(args.solo):
        doc['solo'], solo_dir, date, solo_cfg = solo_block(args.solo)
    if doc['solo']:
        sources['solo'] = {'path': relative(args.solo, root), 'run': os.path.basename(solo_dir), 'date': date, 'kind': 'solo'}
        recorded = doc['solo'].get('ceilings_source')
        if args.ceilings and recorded and not same_file(recorded, args.ceilings):
            checks.append(f"solo run scored against ceilings '{relative(recorded, root)}', "
                          f"figures use '{relative(args.ceilings, root)}'")
        if solo_cfg and os.path.basename(solo_cfg) != os.path.basename(args.device):
            checks.append(f"solo run used device config '{os.path.basename(solo_cfg)}', "
                          f"figures use '{os.path.basename(args.device)}'")
        if recorded:
            doc['solo']['ceilings_source'] = relative(recorded, root)
        if doc['solo'].get('run_valid') is False:
            checks.append('solo run_valid is false (clock integrity) - its numbers are not certified')
    else:
        missing.append('solo rows (stage 4): run ./scripts/run_model_bench.sh')

    if args.coloc and os.path.exists(os.path.join(args.coloc, 'verdict.json')):
        doc['coloc'], date, stage4 = coloc_block(args.coloc, args.coloc_kind)
    if doc['coloc']:
        sources['coloc'] = {'path': relative(os.path.join(args.coloc, 'verdict.json'), root),
                            'run': os.path.basename(os.path.abspath(args.coloc)), 'date': date, 'kind': args.coloc_kind}
        if args.solo and stage4 and not same_file(stage4, args.solo):
            checks.append(f"co-location run composed from '{relative(stage4, root)}', figures use '{relative(args.solo, root)}'")
    else:
        missing.append('co-location (stage 5): run ./scripts/run_colocation.sh')

    if args.power and os.path.exists(os.path.join(args.power, 'verdict.json')):
        doc['power'], date, stage4, ceilings = power_block(args.power)
    if doc['power']:
        sources['power'] = {'path': relative(os.path.join(args.power, 'verdict.json'), root),
                            'run': os.path.basename(os.path.abspath(args.power)), 'date': date, 'kind': 'power'}
        if args.solo and stage4 and not same_file(stage4, args.solo):
            checks.append(f"power run predicted from '{relative(stage4, root)}', figures use '{relative(args.solo, root)}'")
        if args.ceilings and ceilings and not same_file(ceilings, args.ceilings):
            checks.append(f"power run referenced ceilings '{relative(ceilings, root)}', "
                          f"figures use '{relative(args.ceilings, root)}'")
    else:
        missing.append('power (stage 6): run ./scripts/run_power.sh')

    doc['sources'], doc['checks'], doc['missing'] = sources, checks, missing
    text = json.dumps(clean(doc), sort_keys=True, indent=1, allow_nan=False)
    with open(args.out, 'w') as fh:
        fh.write(text + '\n')
    for c in checks:
        print(f'  check: {c}')
    for m in missing:
        print(f'  missing: {m}')
    n_rows = len((doc['solo'] or {}).get('rows') or [])
    n_cells = len((doc['coloc'] or {}).get('cells') or [])
    n_points = len((doc['power'] or {}).get('points') or {})
    print(f"  -> {args.out}  (ceilings {'yes' if doc['ceilings'] else 'no'}, {n_rows} rows, "
          f"{n_cells} cells, {n_points} power points)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
