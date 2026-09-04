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
import argparse, json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model_bench'))
from compute_budgets import bw_eff_from_ceilings, vram_capacity_mb   # noqa: E402

HEADROOM = 0.65


def frame_row(x, bw_eff, cap):
    s = x['solo']; p99 = s['latency_ms']; hz = x['hz']; B = s.get('bytes_per_frame_MB'); v = s.get('vram_mb') or 0.0
    m = {'p99_ms': p99, 'hz': hz, 'deadline_ms': x['deadline_ms'], 'bytes_per_frame_MB': B, 'vram_mb': v,
         'time_share': p99 / 1e3 * hz, 'bw_share': (B / 1e3 * hz / bw_eff) if (B and bw_eff) else None,
         'vram_share': v / cap if cap else None, 'L': x['deadline_ms'] / p99, 'solo_N': s.get('N'), 'solo_cause': s.get('cause'),
         'incomplete': []}
    if B is None: m['incomplete'].append('bytes_per_frame')
    return m


def asr_row(x, bw_eff, cap, results_rows):
    e = x.get('solo_e2e') or {}; d = e.get('detail') or {}; dec = x.get('solo_decode') or {}
    rtf = d.get('rtf_wall_median'); tok_s = d.get('speech_tokens_per_s'); dp99 = (dec.get('latency_ms') or d.get('decode_ms_p99_max'))
    v = e.get('vram_mb') or 0.0
    m = {'kind': 'asr', 'rtf_wall_median': rtf, 'speech_tokens_per_s': tok_s, 'decode_ms_p99': dp99, 'ttft_ms_median': d.get('ttft_ms_median'),
         'encoder_ms_median': d.get('encoder_ms_median'), 'gpu_total_ms_p99': e.get('latency_ms'), 'vram_mb': v, 'deadline_ms': x['deadline_ms'],
         'time_share': rtf, 'time_share_basis': 'rtf_wall_median: GPU duty of the real-time driver',
         'time_share_decode_only': (tok_s * dp99 / 1e3) if (tok_s and dp99) else None,
         'vram_share': v / cap if cap else None, 'L': (x['deadline_ms'] / dp99) if dp99 else None, 'incomplete': []}
    if rtf is None: m['incomplete'].append('rtf (no e2e row)')
    # bandwidth: the engine rows stage 4 measured for this model, charged at the real speech rate
    model = x.get('model'); bw = 0.0; parts = {}
    past = results_rows.get(f'{model}_decoder_past'); enc = results_rows.get(f'{model}_encoder')
    if past and past.get('bytes_per_frame_MB') and tok_s:
        parts['decoder_past'] = past['bytes_per_frame_MB'] / 1e3 * tok_s; bw += parts['decoder_past']
    else:
        m['incomplete'].append('bandwidth (no decoder_past engine row bytes)')
    clips = d.get('clips'); clip_s = None
    if enc and enc.get('bytes_per_frame_MB') and d.get('jsonl') and os.path.exists(d['jsonl']):
        try:
            secs = [json.loads(l).get('seconds') for l in open(d['jsonl']) if l.strip()]
            secs = [s for s in secs if s]
            clip_s = sum(secs) / len(secs) if secs else None
        except Exception:
            clip_s = None
        if clip_s:
            parts['encoder'] = enc['bytes_per_frame_MB'] / 1e3 / clip_s; bw += parts['encoder']
    m['bw_gbps_parts'] = parts; m['clip_seconds_mean'] = clip_s
    m['bw_share'] = (bw / bw_eff) if (bw_eff and 'decoder_past' in parts) else None
    return m


def gen_row(x, bw_eff, cap, hz_grid):
    s = x['solo']; e = x.get('solo_e2e') or {}; dec = x.get('solo_decode') or {}
    step = s['latency_ms']; B = s.get('bytes_per_frame_MB'); v = s.get('vram_mb') or 0.0
    ach = 1e3 / step
    ed = e.get('detail') or {}
    m = {'kind': 'generative', 'step_p99_ms': step, 'step_source': s.get('latency_source'), 'request_p99_ms': e.get('latency_ms'),
         'ttft_ms': ed.get('ttft_ms'), 'tokens_per_s': ed.get('tokens_per_second'), 'decode_ms_p99': dec.get('latency_ms'),
         'achievable_hz': ach, 'hz_is_x': x['hz_is_x'], 'spec_hz': x['hz'], 'deadline_ms': x['deadline_ms'],
         'bytes_per_step_MB': B, 'vram_mb': v, 'vram_share': v / cap if cap else None, 'L': x['deadline_ms'] / step,
         'solo_N': s.get('N'), 'solo_cause': s.get('cause'), 'max_hz_at_N1': s.get('max_hz_at_N1'), 'incomplete': []}
    if x['hz_is_x']:
        m['time_share'] = 1.0; m['charged_hz'] = ach
        m['time_share_at_spec'] = {str(h): step / 1e3 * h for h in hz_grid}
        m['time_share_basis'] = 'X: back-to-back at the achievable step rate'
    else:
        m['time_share'] = step / 1e3 * x['hz']; m['charged_hz'] = x['hz']
        m['time_share_basis'] = f'step p99 x {x["hz"]:g} Hz'
    m['bw_share'] = (B / 1e3 * m['charged_hz'] / bw_eff) if (B and bw_eff) else None
    if B is None: m['incomplete'].append('bytes_per_step')
    return m


def compose(rows):
    def g(r, k):
        v = r['metrics'].get(k); return 0.0 if v is None else v
    U_t = sum(g(r, 'time_share') for r in rows); U_b = sum(g(r, 'bw_share') for r in rows); U_v = sum(g(r, 'vram_share') for r in rows)
    incomplete = sorted({f'{r["row"]}: {i}' for r in rows for i in r['metrics']['incomplete']})
    Umax = max(U_t, U_b, U_v) if rows else 0.0
    Ls = [r['metrics']['L'] for r in rows if r['metrics'].get('L')]; L = min(Ls) if Ls else None
    Lf = [r['metrics']['L'] for r in rows if r['role'] == 'frame' and r['metrics'].get('L')]; L_frame = min(Lf) if Lf else None
    C = 1 / Umax if Umax else None; N = min(L, C) if (L and C) else C
    cause = None
    if N is not None and N < 1:
        cause = 'latency-limited' if (L is not None and L <= (C or math.inf)) else 'throughput-limited'
    binding = max((('time', U_t), ('bandwidth', U_b), ('vram', U_v)), key=lambda t: t[1])[0] if rows else None
    for r in rows:
        m = r['metrics']; others = U_t - g(r, 'time_share')
        p = m.get('p99_ms') or m.get('step_p99_ms') or m.get('decode_ms_p99')
        m['p99_contended_pred'] = (p / (1 - others)) if (p and others < 1) else None
        m['others_time_share'] = others
        m['meets_deadline_pred'] = (m['p99_contended_pred'] <= m['deadline_ms']) if m['p99_contended_pred'] else None
    return {'U_time': U_t, 'U_bw': U_b, 'U_vram': U_v, 'U_max': Umax, 'binding_budget': binding, 'C': C, 'L': L, 'L_frame': L_frame,
            'N': N, 'cause': cause, 'units_at_0.65': math.ceil(U_t / HEADROOM) if U_t else 0, 'units_at_1.0': math.ceil(U_t) if U_t else 0,
            'fits_one_device_at_0.65': U_t <= HEADROOM and U_b <= HEADROOM and U_v <= 1.0,
            'budget_complete': not incomplete, 'budgets_not_measured': incomplete,
            'sum_frame_p99_ms': sum(r['metrics']['p99_ms'] for r in rows if r['role'] == 'frame'),
            'period_ms': min((1e3 / r['metrics']['hz'] for r in rows if r['role'] == 'frame'), default=None)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--resolved', required=True); ap.add_argument('--device', required=True)
    ap.add_argument('--ceilings', default=''); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    mix = json.load(open(a.resolved))
    if mix.get('errors'):
        sys.exit(f'{mix["mix"]}: resolved with errors - fix the mix first: {mix["errors"]}')
    R = json.load(open(mix['stage4_results'])); rr = {r['name']: r for r in R.get('rows', [])}
    bw_eff = R.get('bw_eff_gbps'); bw_src = R.get('bw_ceiling_source')
    if not bw_eff:
        bw_eff, bw_src = bw_eff_from_ceilings(a.ceilings)
    cap = R.get('vram_capacity_mb'); cap_src = R.get('vram_capacity_source')
    if not cap:
        cap, cap_src = vram_capacity_mb(a.device)
    hz_grid = next((r.get('hz_grid') for r in R.get('rows', []) if r.get('hz_grid')), [1, 2, 5, 10, 20, 30])
    rows = []
    for x in mix['rows']:
        if x['role'] == 'frame':
            m = frame_row(x, bw_eff, cap)
        elif x['kind'] == 'e2e':
            m = asr_row(x, bw_eff, cap, rr)
        else:
            m = gen_row(x, bw_eff, cap, hz_grid)
        rows.append({'row': x['name'], 'role': x['role'], 'kind': x['kind'], 'precision': x.get('precision'), 'metrics': m})
    comp = compose(rows)
    out = {'mix': mix['mix'], 'platform': mix['platform'], 'stage4_results': mix['stage4_results'],
           'bw_eff_gbps': bw_eff, 'bw_eff_source': bw_src, 'vram_capacity_mb': cap, 'vram_capacity_source': cap_src,
           'ceilings': R.get('ceilings'), 'headroom': HEADROOM, 'hz_grid': hz_grid, 'rows': rows, 'composed': comp}
    json.dump(out, open(a.out, 'w'), indent=1)
    c = comp
    print(f'{mix["mix"]}: U_time {c["U_time"]:.3f}  U_bw {c["U_bw"]:.3f}  U_vram {c["U_vram"]:.4f}  -> U_max {c["U_max"]:.3f} ({c["binding_budget"]})  '
          f'C {c["C"]:.2f}  L {c["L"]:.2f} (frame rows {c["L_frame"] or 0:.2f})  N {c["N"]:.2f}{"  " + c["cause"] if c["cause"] else ""}  '
          f'units@0.65 {c["units_at_0.65"]}  fits@0.65 {c["fits_one_device_at_0.65"]}' + ('' if c['budget_complete'] else f'  INCOMPLETE: {c["budgets_not_measured"]}'))
    for r in rows:
        m = r['metrics']; p = m.get('p99_contended_pred')
        print(f'  {r["row"]:28} {r["role"]:5} time {m["time_share"] or 0:.3f}  bw {m["bw_share"] if m["bw_share"] is not None else float("nan"):.3f}  '
              f'vram {m["vram_share"] or 0:.4f}  L {m["L"] or 0:.2f}  contended p99 pred {p if p is None else round(p, 2)} (deadline {m["deadline_ms"]:g})')


if __name__ == '__main__':
    main()
