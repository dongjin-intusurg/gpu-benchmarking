#!/usr/bin/env bash
# Side load: the Edge-LLM llm_inference battery (stage-4 generative row, Jetson)
# back-to-back for the run: the registered battery's requests are replicated
# K = ceil(1.3 x seconds x 1000 / (n_requests x request_ms)) times so the
# battery outlasts the frame rows by a margin (the harness waits for it after
# the rows so --dumpProfile lands). request_ms is the wall time of one battery
# request: measured from this row's paced solo when PACED_SOLO_SIDE points at
# it (the arms), otherwise estimated from the stage-4 profile (vision + prefill
# + generated tokens x decode step per request) - NOT the chunked step latency,
# which is several times shorter than a whole request. Same invocation as
# stage 4 (context reuse, one-slot recurrent snapshot pool).
#   side_edgellm.sh <resolved mix.json> <row> <out_dir> <seconds>
# Readiness for the harness: "Runtime tensors successfully allocated" in e2e.log.
# Leaves <out_dir>/side_result.json (stage medians / p99, tok/s, peak memory,
# overlap_frac = share of the battery's wall time inside the frame-row window)
# - also when the harness terminates it after its bounded wait (status FAILED,
# no profile: llm_inference dumps its profile only at the end).
set -uo pipefail
RES="$1"; ROW="$2"; OUT="$3"; SEC="$4"; mkdir -p "$OUT"
eval "$(python3 - "$RES" "$ROW" <<'PY'
import json, sys, shlex
m = json.load(open(sys.argv[1])); r = next(x for x in m['rows'] if x['name'] == sys.argv[2]); g = r['registry']
for k in ('llm_dir', 'visual_dir', 'battery', 'image'):
    print(f'{k.upper()}={shlex.quote(str(g.get(k) or ""))}')
import os
e2e = r.get('solo_e2e') or {}; d = e2e.get('detail') or {}
ms = None; src = 'stage-4 step latency'
ps = os.path.join(os.environ.get('PACED_SOLO_SIDE') or '', 'side_result.json')
if os.path.isfile(ps):
    try:
        q = json.load(open(ps))
        if q.get('status') == 'OK' and q.get('wall_s') and q.get('battery_requests'):
            ms = 1000.0 * q['wall_s'] / q['battery_requests']; src = 'paced solo wall per request'
    except Exception: pass
if ms is None and d.get('generated_tokens') and d.get('decode_ms_median'):
    n_req = (d.get('stages') or {}).get('llm_prefill', {}).get('count') or 1
    ms = (d.get('visual_ms_median') or 0) + (d.get('prefill_ms_median') or 0) + d['decode_ms_median'] * d['generated_tokens'] / n_req
    src = 'stage-4 profile: vision + prefill + tokens x decode per request'
if ms is None: ms = e2e.get('latency_ms') or r['solo']['latency_ms']
print(f'REQ_MS={ms:.1f}'); print(f'REQ_SRC={shlex.quote(src)}')
PY
)"
EDGELLM_ROOT="${EDGELLM_ROOT:-$HOME/tools/TensorRT-Edge-LLM}"
INFER="$EDGELLM_ROOT/build/examples/llm/llm_inference"
export EDGELLM_PLUGIN_PATH="${EDGELLM_PLUGIN_PATH:-$EDGELLM_ROOT/build/libNvInfer_edgellm_plugin.so}"
[ -x "$INFER" ] || { echo "llm_inference not found at $INFER (EDGELLM_ROOT)"; exit 2; }
[ -f "$BATTERY" ] || { echo "battery missing: $BATTERY"; exit 2; }
K=$(python3 - "$BATTERY" "$SEC" "$REQ_MS" "$OUT/battery.json" <<'PY'
import json, sys, math
b = json.load(open(sys.argv[1])); sec = float(sys.argv[2]); ms = float(sys.argv[3])
n = len(b['requests']); k = max(1, math.ceil(1.3 * sec * 1000 / (n * ms)))
b['requests'] = b['requests'] * k; json.dump(b, open(sys.argv[4], 'w')); print(k)
PY
)
N=$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))['requests']))" "$OUT/battery.json")
SNAP=$(python3 - "$LLM_DIR/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
n = int(c.get('num_linear_attn_layers') or 0)
rec = (int(c.get('recurrent_state_num_heads') or 0) * int(c.get('recurrent_state_head_dim') or 0)
       * int(c.get('recurrent_state_size') or 0) * (4 if c.get('recurrent_state_dtype', 'fp32') == 'fp32' else 2))
conv = int(c.get('conv_dim') or 0) * int(c.get('conv_kernel') or 0) * (4 if c.get('conv_state_dtype') == 'fp32' else 2)
slot = n * (rec + conv)
print(max(64 << 20, (-(-slot // (1 << 20))) << 20))
PY
)
echo "side_edgellm: $ROW  battery x$K = $N requests (~$(python3 -c "print(round($N*$REQ_MS/1000))") s at ${REQ_MS} ms per request [$REQ_SRC]; run ${SEC}s)"
T0=$(date +%s.%N)
CHILD=""; TERMED=0
trap 'TERMED=1; [ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null' TERM INT
( cd "$EDGELLM_ROOT" && exec "$INFER" --engineDir "$LLM_DIR" ${VISUAL_DIR:+--multimodalEngineDir "$(dirname "$VISUAL_DIR")"} \
    --inputFile "$OUT/battery.json" --outputFile "$OUT/e2e_out.json" \
    --dumpProfile --profileOutputFile "$OUT/e2e_profile.json" \
    --enableContextReuse --contextCacheRecurrentSnapshotPoolBytes "$SNAP" \
    --contextCachePartialKVSnapshotPoolBytes 67108864 ) > "$OUT/e2e.log" 2>&1 &
CHILD=$!
wait "$CHILD"; RC=$?
[ "$TERMED" = 1 ] && { wait "$CHILD" 2>/dev/null; RC=143; }
python3 - "$OUT" "$ROW" "$RC" "$K" "$N" "$REQ_MS" "$SEC" "$T0" "$REQ_SRC" "${ROW_START_FILE:-}" <<'PY'
import json, sys, os, time, re, datetime
out, row, rc, k, n, req_ms, sec, t0, src = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8]), sys.argv[9]
row_start_file = sys.argv[10] if len(sys.argv) > 10 else ''

def request_starts(log, t0):
    # one 'Processing vision inputs' line per request, stamped [HH:MM:SS.mmm] wall clock; the 'Processing complete' line closes the last one
    day = datetime.datetime.fromtimestamp(t0).replace(hour=0, minute=0, second=0, microsecond=0).timestamp(); ts = []; end = None
    for line in open(log, errors='replace'):
        m = re.match(r'\[(\d\d):(\d\d):(\d\d)\.(\d\d\d)\]', line)
        if not m: continue
        t = day + int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + int(m[4]) / 1e3
        if t < t0 - 3600: t += 86400   # the run crossed midnight
        if 'Processing vision inputs' in line: ts.append(t)
        elif 'Processing complete' in line: end = t
    return ts, end

def pct(v, q):
    v = sorted(v); return v[min(len(v) - 1, int(round(q * (len(v) - 1))))] if v else None
wall = round(time.time() - t0, 1)
r = {'name': row, 'role': 'side', 'runtime': 'edgellm', 'driver_rc': rc, 'battery_replicas': k, 'battery_requests': n, 'request_ms_planned': round(req_ms, 1), 'request_ms_source': src,
     'planned_s': round(n * req_ms / 1000, 1), 'run_window_s': sec, 'wall_s': wall, 'overlap_frac': round(min(1.0, sec / wall), 3) if wall else None,
     'status': 'FAILED', 'why': f'llm_inference rc={rc}'}
pf = f'{out}/e2e_profile.json'
if os.path.exists(pf):
    try:
        p = json.load(open(pf)); stages = {s['stage_id']: s['gpu_time_stats'] for s in p.get('stages', [])}; gen = p.get('generation', {})
        r.update({'status': 'OK', 'why': None, 'stages': stages, 'tokens_per_second': gen.get('tokens_per_second'), 'generated_tokens': gen.get('generated_tokens'),
                  'decode_ms_median': stages.get('llm_generation', {}).get('median_ms'), 'decode_ms_p99': stages.get('llm_generation', {}).get('p99_ms'),
                  'prefill_ms_median': stages.get('llm_prefill', {}).get('median_ms'), 'visual_ms_median': stages.get('vision_encoder', {}).get('median_ms'),
                  'ttft_ms': sum(stages[x]['median_ms'] for x in ('vision_encoder', 'llm_prefill') if x in stages) + (stages['llm_generation']['median_ms'] if 'llm_generation' in stages else 0),
                  'peak_unified_memory_mb': p.get('peak_unified_memory_mb'), 'profile': pf})
        r['summary'] = f"{n} requests: ttft {r['ttft_ms']:.0f} ms, decode med {r['decode_ms_median']:.1f} / p99 {r['decode_ms_p99']:.1f} ms, {r['tokens_per_second']:.1f} tok/s, wall {r['wall_s']} s"
        r['summary'] += f", {n} requests in {wall} s ({r['overlap_frac']:.0%} of that inside the {sec:g} s frame window)"
        # per-request wall latency from the driver log; windowed to the frame run when the runner published the shared trigger
        starts, end = request_starts(f'{out}/e2e.log', t0)
        if len(starts) >= 2:
            ends = starts[1:] + [end or starts[-1]]; dur = [1e3 * (b - a) for a, b in zip(starts, ends)]
            tok_per_req = (gen.get('generated_tokens') or 0) / max(1, len(starts))
            r.update({'requests_timed': len(starts), 'request_ms_mean': round(sum(dur) / len(dur), 1), 'request_ms_median': round(pct(dur, 0.5), 1), 'request_ms_p99': round(pct(dur, 0.99), 1)})
            if row_start_file and os.path.isfile(row_start_file):
                try:
                    rs_ns = int(open(row_start_file).read().split()[0])
                    w0 = time.time() - (time.monotonic_ns() - rs_ns) / 1e9   # wall-clock instant of the shared trigger (CLOCK_MONOTONIC == steady_clock)
                    win = [d for s, d in zip(starts, dur) if w0 <= s < w0 + sec]
                    r.update({'window_s': sec, 'window_requests': len(win)})
                    if win:
                        r.update({'window_request_ms_mean': round(sum(win) / len(win), 1), 'window_request_ms_median': round(pct(win, 0.5), 1), 'window_request_ms_p99': round(pct(win, 0.99), 1),
                                  'window_tokens_per_second_est': round(1e3 * tok_per_req * len(win) / sum(win), 2),
                                  'window_basis': f'{len(win)} requests started inside the frame window; tok/s estimated from the mean tokens per request'})
                        r['summary'] += f" | in the {sec:g} s frame window ({len(win)} requests): request mean {r['window_request_ms_mean']:.0f} / p99 {r['window_request_ms_p99']:.0f} ms, ~{r['window_tokens_per_second_est']:.1f} tok/s"
                except Exception as ex:
                    r['window_error'] = str(ex)
        if rc != 0: r['status'] = 'FAILED'; r['why'] = f'llm_inference rc={rc} (profile present: killed after the rows?)'
    except Exception as ex:
        r['why'] = f'profile unreadable: {ex}'
else:
    r['why'] += (' - terminated by the harness after its bounded wait, no profile dumped: the battery was sized for '
                 f'{n * req_ms / 1000:.0f} s at {req_ms:.0f} ms/request ({src}) and was still running after {wall} s - '
                 'the row slowed more under contention than the wait allows (raise --side-wait)') if rc == 143 else ' - no profile dumped'
json.dump(r, open(f'{out}/side_result.json', 'w'), indent=1); print(r.get('summary') or r['why'])
PY
exit $RC
