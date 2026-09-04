#!/usr/bin/env python3
"""Composed (arithmetic) budget of a mix from the stage-4 solo numbers - no GPU.

The same three gauges as a solo row, summed over the residents of one device:

  frame row      time_share = p99/1e3 x hz         bw_share = bytes_MB/1e3 x hz / bw_eff
  ASR side       time_share = rtf_wall_median (the real-time driver's GPU duty) with the
                 decode-rate decomposition recorded (speech tok/s x decode p99);
                 bw_share from the decoder_past / encoder engine rows' bytes at the real
                 speech rate when stage 4 measured them (else the budget is incomplete)
  generative     numeric hz: time_share = step_p99/1e3 x hz; X: time_share = 1.0
  side           (back-to-back at the achievable rate) with time_share_at_spec on the
                 stage-4 rate grid; bw from the step bytes at the charged rate
  every row      vram_share = vram_mb / capacity

  U_* = sums, U_max, C = 1/U_max, L = min(deadline/p99) (L_frame over frame rows only),
  N = min(L, C), cause, units_at_0.65 / units_at_1.0, fits_one_device_at_0.65.
  Per row the modal prediction: p99_contended_pred = solo p99 / (1 - U_time of the OTHERS),
  null when the others already saturate the device; meets_deadline_pred.

Inputs are read, never typed: the resolved mix (mixes.py resolve), the stage-4
results.json it names (bw_eff_gbps, vram_capacity_mb, ceilings) and the device config
as the fallback for the capacity constants (compute_budgets.py helpers).

    compose_mix.py --resolved <mix.json> --device <cfg> [--ceilings <json>] --out composed.json
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model_bench'))
from compute_budgets import bw_eff_from_ceilings, vram_capacity_mb   # noqa: E402

HEADROOM = 0.65
HZ_GRID_FALLBACK = [1, 2, 5, 10, 20, 30]


def frame_row(row, bw_eff, vram_cap):
    solo = row['solo']
    p99 = solo['latency_ms']
    hz = row['hz']
    bytes_mb = solo.get('bytes_per_frame_MB')
    vram_mb = solo.get('vram_mb') or 0.0
    metrics = {'p99_ms': p99, 'hz': hz, 'deadline_ms': row['deadline_ms'], 'bytes_per_frame_MB': bytes_mb,
               'vram_mb': vram_mb, 'time_share': p99 / 1e3 * hz,
               'bw_share': (bytes_mb / 1e3 * hz / bw_eff) if (bytes_mb and bw_eff) else None,
               'vram_share': vram_mb / vram_cap if vram_cap else None, 'L': row['deadline_ms'] / p99,
               'solo_N': solo.get('N'), 'solo_cause': solo.get('cause'), 'incomplete': []}
    if bytes_mb is None:
        metrics['incomplete'].append('bytes_per_frame')
    return metrics


def _mean_clip_seconds(jsonl_path):
    try:
        seconds = [json.loads(line).get('seconds') for line in open(jsonl_path) if line.strip()]
    except Exception:
        return None
    seconds = [value for value in seconds if value]
    return sum(seconds) / len(seconds) if seconds else None


def asr_row(row, bw_eff, vram_cap, results_rows):
    e2e = row.get('solo_e2e') or {}
    detail = e2e.get('detail') or {}
    decode = row.get('solo_decode') or {}
    rtf = detail.get('rtf_wall_median')
    tokens_per_s = detail.get('speech_tokens_per_s')
    decode_p99 = decode.get('latency_ms') or detail.get('decode_ms_p99_max')
    vram_mb = e2e.get('vram_mb') or 0.0
    decode_only_share = (tokens_per_s * decode_p99 / 1e3) if (tokens_per_s and decode_p99) else None
    metrics = {'kind': 'asr', 'rtf_wall_median': rtf, 'speech_tokens_per_s': tokens_per_s,
               'decode_ms_p99': decode_p99, 'ttft_ms_median': detail.get('ttft_ms_median'),
               'encoder_ms_median': detail.get('encoder_ms_median'),
               'gpu_total_ms_p99': e2e.get('latency_ms'), 'vram_mb': vram_mb, 'deadline_ms': row['deadline_ms'],
               'time_share': rtf, 'time_share_basis': 'rtf_wall_median: GPU duty of the real-time driver',
               'time_share_decode_only': decode_only_share,
               'vram_share': vram_mb / vram_cap if vram_cap else None,
               'L': (row['deadline_ms'] / decode_p99) if decode_p99 else None, 'incomplete': []}
    if rtf is None:
        metrics['incomplete'].append('rtf (no e2e row)')

    # bandwidth: the engine rows stage 4 measured for this model, charged at the real speech rate
    model = row.get('model')
    decoder_past = results_rows.get(f'{model}_decoder_past')
    encoder = results_rows.get(f'{model}_encoder')
    bw_gbps = 0.0
    parts = {}
    if decoder_past and decoder_past.get('bytes_per_frame_MB') and tokens_per_s:
        parts['decoder_past'] = decoder_past['bytes_per_frame_MB'] / 1e3 * tokens_per_s
        bw_gbps += parts['decoder_past']
    else:
        metrics['incomplete'].append('bandwidth (no decoder_past engine row bytes)')
    clip_seconds = None
    encoder_bytes_known = bool(encoder and encoder.get('bytes_per_frame_MB'))
    if encoder_bytes_known and detail.get('jsonl') and os.path.exists(detail['jsonl']):
        clip_seconds = _mean_clip_seconds(detail['jsonl'])
        if clip_seconds:
            parts['encoder'] = encoder['bytes_per_frame_MB'] / 1e3 / clip_seconds
            bw_gbps += parts['encoder']
    metrics['bw_gbps_parts'] = parts
    metrics['clip_seconds_mean'] = clip_seconds
    metrics['bw_share'] = (bw_gbps / bw_eff) if (bw_eff and 'decoder_past' in parts) else None
    return metrics


def gen_row(row, bw_eff, vram_cap, hz_grid):
    solo = row['solo']
    e2e = row.get('solo_e2e') or {}
    e2e_detail = e2e.get('detail') or {}
    decode = row.get('solo_decode') or {}
    step_p99 = solo['latency_ms']
    bytes_mb = solo.get('bytes_per_frame_MB')
    vram_mb = solo.get('vram_mb') or 0.0
    achievable_hz = 1e3 / step_p99
    metrics = {'kind': 'generative', 'step_p99_ms': step_p99, 'step_source': solo.get('latency_source'),
               'request_p99_ms': e2e.get('latency_ms'), 'ttft_ms': e2e_detail.get('ttft_ms'),
               'tokens_per_s': e2e_detail.get('tokens_per_second'), 'decode_ms_p99': decode.get('latency_ms'),
               'achievable_hz': achievable_hz, 'hz_is_x': row['hz_is_x'], 'spec_hz': row['hz'],
               'deadline_ms': row['deadline_ms'], 'bytes_per_step_MB': bytes_mb, 'vram_mb': vram_mb,
               'vram_share': vram_mb / vram_cap if vram_cap else None, 'L': row['deadline_ms'] / step_p99,
               'solo_N': solo.get('N'), 'solo_cause': solo.get('cause'),
               'max_hz_at_N1': solo.get('max_hz_at_N1'), 'incomplete': []}
    if row['hz_is_x']:
        metrics['time_share'] = 1.0
        metrics['charged_hz'] = achievable_hz
        metrics['time_share_at_spec'] = {str(hz): step_p99 / 1e3 * hz for hz in hz_grid}
        metrics['time_share_basis'] = 'X: back-to-back at the achievable step rate'
    else:
        metrics['time_share'] = step_p99 / 1e3 * row['hz']
        metrics['charged_hz'] = row['hz']
        metrics['time_share_basis'] = f'step p99 x {row["hz"]:g} Hz'
    metrics['bw_share'] = (bytes_mb / 1e3 * metrics['charged_hz'] / bw_eff) if (bytes_mb and bw_eff) else None
    if bytes_mb is None:
        metrics['incomplete'].append('bytes_per_step')
    return metrics


def _share(row, key):
    value = row['metrics'].get(key)
    return 0.0 if value is None else value


def _cause(n_value, l_value, c_value):
    if n_value is None or n_value >= 1:
        return None
    if l_value is not None and l_value <= (c_value or math.inf):
        return 'latency-limited'
    return 'throughput-limited'


def compose(rows):
    u_time = sum(_share(row, 'time_share') for row in rows)
    u_bw = sum(_share(row, 'bw_share') for row in rows)
    u_vram = sum(_share(row, 'vram_share') for row in rows)
    incomplete = sorted({f'{row["row"]}: {item}' for row in rows for item in row['metrics']['incomplete']})
    u_max = max(u_time, u_bw, u_vram) if rows else 0.0
    l_values = [row['metrics']['L'] for row in rows if row['metrics'].get('L')]
    l_value = min(l_values) if l_values else None
    l_frame_values = [row['metrics']['L'] for row in rows
                      if row['role'] == 'frame' and row['metrics'].get('L')]
    l_frame = min(l_frame_values) if l_frame_values else None
    c_value = 1 / u_max if u_max else None
    n_value = min(l_value, c_value) if (l_value and c_value) else c_value
    budgets = (('time', u_time), ('bandwidth', u_bw), ('vram', u_vram))
    binding = max(budgets, key=lambda named_share: named_share[1])[0] if rows else None
    for row in rows:
        metrics = row['metrics']
        others = u_time - _share(row, 'time_share')
        p99 = metrics.get('p99_ms') or metrics.get('step_p99_ms') or metrics.get('decode_ms_p99')
        metrics['p99_contended_pred'] = (p99 / (1 - others)) if (p99 and others < 1) else None
        metrics['others_time_share'] = others
        metrics['meets_deadline_pred'] = ((metrics['p99_contended_pred'] <= metrics['deadline_ms'])
                                          if metrics['p99_contended_pred'] else None)
    return {'U_time': u_time, 'U_bw': u_bw, 'U_vram': u_vram, 'U_max': u_max, 'binding_budget': binding,
            'C': c_value, 'L': l_value, 'L_frame': l_frame, 'N': n_value,
            'cause': _cause(n_value, l_value, c_value),
            'units_at_0.65': math.ceil(u_time / HEADROOM) if u_time else 0,
            'units_at_1.0': math.ceil(u_time) if u_time else 0,
            'fits_one_device_at_0.65': u_time <= HEADROOM and u_bw <= HEADROOM and u_vram <= 1.0,
            'budget_complete': not incomplete, 'budgets_not_measured': incomplete,
            'sum_frame_p99_ms': sum(row['metrics']['p99_ms'] for row in rows if row['role'] == 'frame'),
            'period_ms': min((1e3 / row['metrics']['hz'] for row in rows if row['role'] == 'frame'),
                             default=None)}


def row_metrics(row, bw_eff, vram_cap, results_rows, hz_grid):
    if row['role'] == 'frame':
        return frame_row(row, bw_eff, vram_cap)
    if row['kind'] == 'e2e':
        return asr_row(row, bw_eff, vram_cap, results_rows)
    return gen_row(row, bw_eff, vram_cap, hz_grid)


def print_summary(mix_name, composed, rows):
    incomplete = '' if composed['budget_complete'] else f'  INCOMPLETE: {composed["budgets_not_measured"]}'
    cause = '  ' + composed['cause'] if composed['cause'] else ''
    print(f'{mix_name}: U_time {composed["U_time"]:.3f}  U_bw {composed["U_bw"]:.3f}  '
          f'U_vram {composed["U_vram"]:.4f}  '
          f'-> U_max {composed["U_max"]:.3f} ({composed["binding_budget"]})  '
          f'C {composed["C"]:.2f}  L {composed["L"]:.2f} (frame rows {composed["L_frame"] or 0:.2f})  '
          f'N {composed["N"]:.2f}{cause}  '
          f'units@0.65 {composed["units_at_0.65"]}  '
          f'fits@0.65 {composed["fits_one_device_at_0.65"]}' + incomplete)
    for row in rows:
        metrics = row['metrics']
        pred = metrics.get('p99_contended_pred')
        bw_share = metrics['bw_share'] if metrics['bw_share'] is not None else float('nan')
        print(f'  {row["row"]:28} {row["role"]:5} time {metrics["time_share"] or 0:.3f}  bw {bw_share:.3f}  '
              f'vram {metrics["vram_share"] or 0:.4f}  L {metrics["L"] or 0:.2f}  '
              f'contended p99 pred {pred if pred is None else round(pred, 2)} '
              f'(deadline {metrics["deadline_ms"]:g})')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--resolved', required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--ceilings', default='')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    mix = json.load(open(args.resolved))
    if mix.get('errors'):
        sys.exit(f'{mix["mix"]}: resolved with errors - fix the mix first: {mix["errors"]}')
    results = json.load(open(mix['stage4_results']))
    results_rows = {row['name']: row for row in results.get('rows', [])}
    bw_eff = results.get('bw_eff_gbps')
    bw_source = results.get('bw_ceiling_source')
    if not bw_eff:
        bw_eff, bw_source = bw_eff_from_ceilings(args.ceilings)
    vram_cap = results.get('vram_capacity_mb')
    vram_source = results.get('vram_capacity_source')
    if not vram_cap:
        vram_cap, vram_source = vram_capacity_mb(args.device)
    hz_grid = next((row.get('hz_grid') for row in results.get('rows', []) if row.get('hz_grid')),
                   HZ_GRID_FALLBACK)

    rows = [{'row': row['name'], 'role': row['role'], 'kind': row['kind'], 'precision': row.get('precision'),
             'metrics': row_metrics(row, bw_eff, vram_cap, results_rows, hz_grid)} for row in mix['rows']]
    composed = compose(rows)
    out = {'mix': mix['mix'], 'platform': mix['platform'], 'stage4_results': mix['stage4_results'],
           'bw_eff_gbps': bw_eff, 'bw_eff_source': bw_source, 'vram_capacity_mb': vram_cap,
           'vram_capacity_source': vram_source, 'ceilings': results.get('ceilings'), 'headroom': HEADROOM,
           'hz_grid': hz_grid, 'rows': rows, 'composed': composed}
    json.dump(out, open(args.out, 'w'), indent=1)
    print_summary(mix['mix'], composed, rows)


if __name__ == '__main__':
    main()
