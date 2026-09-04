#!/usr/bin/env bash
# Side load: the TensorRT-LLM VLM step tool (stage-4 generative row, discrete
# platform) in its `loop` mode - back-to-back control steps until the harness
# stops it (SIGTERM). Same tool and serving directory as stage 4's sweep;
# prints READY to side.log once the engine is loaded (the harness's readiness
# marker) and leaves <out_dir>/side_result.json from vlm_e2e_steps.jsonl.
#   side_trtllm.sh <resolved mix.json> <row> <out_dir> <seconds>
# Written on the Jetson side without a discrete card to test on: validate on
# the discrete box (BENCH_PY / TRTLLM_PY select the TensorRT-LLM interpreter).
set -uo pipefail
RES="$1"; ROW="$2"; OUT="$3"; SEC="$4"; mkdir -p "$OUT"
eval "$(python3 - "$RES" "$ROW" <<'PY'
import json, sys, shlex
m = json.load(open(sys.argv[1])); r = next(x for x in m['rows'] if x['name'] == sys.argv[2]); g = r['registry']
for k in ('serving_dir', 'image', 'chunk'):
    print(f'{k.upper()}={shlex.quote(str(g.get(k) or ""))}')
print(f"STEP_MS={r['solo']['latency_ms']}")
PY
)"
HERE="$(cd "$(dirname "$0")" && pwd)"
STEP="$HERE/../model_bench/trtllm/trtllm_vlm_step.py"; [ -f "$STEP" ] || STEP="${HANDOFF_ROOT:-$HERE/../model_bench/trtllm}/trtllm_vlm_step.py"
[ -f "$STEP" ] || { echo "trtllm_vlm_step.py not found"; exit 2; }
for _e in "${BENCH_ENV_SH:-}" "${HANDOFF_ROOT:-$HERE/../model_bench/trtllm}/trtllm_env.sh" "$HERE/../model_bench/trtllm/trtllm_env.sh"; do
  [ -n "$_e" ] && [ -r "$_e" ] && { . "$_e"; break; }
done
TRT_PY="${BENCH_PY:-${TRTLLM_PY:-python3}}"
[ -d "$SERVING_DIR" ] || { echo "serving dir missing: $SERVING_DIR"; exit 2; }
echo "side_trtllm: $ROW  loop mode, chunk ${CHUNK:-8}, solo step ${STEP_MS} ms (run ${SEC}s, stopped by the harness)"
T0=$(date +%s.%N)
"$TRT_PY" "$STEP" loop --model "$SERVING_DIR" ${IMAGE:+--image "$IMAGE"} --chunk "${CHUNK:-8}" --out "$OUT" --tag "$ROW" >> "$OUT/side.log" 2>&1 &
PID=$!
trap 'kill -TERM $PID 2>/dev/null' TERM INT
# self-bound like side_edgellm.sh's finite battery: the loop only stops on SIGTERM and in
# the paced-solo pass nobody sends one - stop it seconds + 120 s after READY (the harness's
# own side_wait margin), so it outlasts the frame rows in an arm yet ends on its own solo
( while ! grep -q '^READY' "$OUT/side.log" 2>/dev/null; do sleep 1; kill -0 $PID 2>/dev/null || exit 0; done
  sleep "$(python3 -c "print($SEC + 120)")"; kill -TERM $PID 2>/dev/null ) &
WD=$!
wait $PID; RC=$?
kill $WD 2>/dev/null
python3 - "$OUT" "$ROW" "$RC" "$STEP_MS" "$SEC" "$T0" "${ROW_START_FILE:-}" <<'PY'
import json, sys, os, time, statistics as st
out, row, rc, step_ms, sec, t0 = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
row_start_file = sys.argv[7] if len(sys.argv) > 7 else ''
r = {'name': row, 'role': 'side', 'runtime': 'trtllm', 'driver_rc': rc, 'planned_s': sec, 'wall_s': round(time.time() - t0, 1), 'status': 'FAILED', 'why': f'step tool rc={rc}'}
f = f'{out}/vlm_e2e_steps.jsonl'
recs = [json.loads(l) for l in open(f) if l.strip()] if os.path.exists(f) else []
v = sorted(x['total_ms'] for x in recs if x.get('total_ms'))
if v:
    p99 = v[min(len(v) - 1, int(round(0.99 * (len(v) - 1))))]
    # rc 143 / -15: stopped by the harness's TERM as designed
    ok = rc in (0, 143, -15)
    r.update({'status': 'OK' if ok else 'FAILED', 'why': None if ok else r['why'], 'steps': len(v), 'step_ms_median': st.median(v), 'step_ms_p99': p99,
              'step_ms_solo': step_ms, 'contention_factor': round(p99 / step_ms, 3) if step_ms else None, 'steps_file': f})
    r['summary'] = f"{len(v)} steps: median {r['step_ms_median']:.1f} / p99 {p99:.1f} ms (solo p99 {step_ms:.1f})"
    # windowed to the frame run when the runner published the shared trigger (same monotonic
    # clock as the tool's t_start_ns): the contended numbers verdict.py ratios against the solo pass
    if row_start_file and os.path.isfile(row_start_file):
        start = int(open(row_start_file).read().split()[0])
        win = sorted(x['total_ms'] for x in recs if x.get('total_ms') and start <= x.get('t_start_ns', -1) < start + int(sec * 1e9))
        r.update({'window_s': sec, 'window_start_ns': start, 'window_steps': len(win), 'window_requests': len(win)})
        if win:
            wp99 = win[min(len(win) - 1, int(round(0.99 * (len(win) - 1))))]
            r.update({'window_step_ms_median': st.median(win), 'window_step_ms_p99': wp99,
                      'window_contention_factor': round(wp99 / step_ms, 3) if step_ms else None})
            r['summary'] += f" | in the {sec:g} s frame window ({len(win)} steps): median {r['window_step_ms_median']:.1f} / p99 {wp99:.1f} ms"
else:
    r['why'] += ' - no steps recorded (engine never became READY?)'
json.dump(r, open(f'{out}/side_result.json', 'w'), indent=1); print(r.get('summary') or r['why'])
PY
exit 0
