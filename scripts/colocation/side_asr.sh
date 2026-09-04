#!/usr/bin/env bash
# Side load: the ASR C++ driver (stage-4 kind e2e) in REAL TIME - each clip is
# followed by a sleep to its own duration, so the GPU sees speech-rate work.
# Repeats are sized to cover the run (+15 s) from the clip list's durations.
#   side_asr.sh <resolved mix.json> <row> <out_dir> <seconds>
# Readiness for the harness: the first "clip ..." line in <out_dir>/driver.log.
# Leaves <out_dir>/side_result.json (ttft / decode / rtf / gpu-total stats over
# every clip pass, plus the plan). When the harness hands over ROW_START_FILE
# (the frame rows' shared start, steady-clock ns) the same statistics are also
# reported over the clips that started inside the frame window
# (window_* keys, window_clips) - the pass outlasts the rows by design, so the
# whole-pass numbers mix contended and uncontended clips. CUDA_MPS_* variables
# are inherited as set by the harness (mps arm).
set -uo pipefail
RES="$1"; ROW="$2"; OUT="$3"; SEC="$4"; mkdir -p "$OUT"
eval "$(python3 - "$RES" "$ROW" <<'PY'
import json, sys, shlex
m = json.load(open(sys.argv[1])); r = next(x for x in m['rows'] if x['name'] == sys.argv[2]); g = r['registry']
for k in ('engine_dir', 'precision', 'src', 'inputs', 'args', 'model'):
    print(f'{k.upper()}={shlex.quote(str(g.get(k) or ""))}')
PY
)"
BIN_DIR="${COLOC_BIN_DIR:-$OUT/bin}"; BIN="$BIN_DIR/${MODEL}_e2e"; mkdir -p "$BIN_DIR"
if [ ! -x "$BIN" ] || [ "$SRC" -nt "$BIN" ]; then
  g++ -O2 -std=c++17 "$SRC" -I"/usr/include/$(gcc -dumpmachine)" -I/usr/local/cuda/include \
      -L/usr/local/cuda/lib64 -lnvinfer -lnvinfer_plugin -lcudart -ldl -lpthread -o "$BIN" > "$OUT/compile.log" 2>&1 \
    || { echo "driver failed to compile - $OUT/compile.log"; exit 2; }
fi
PASS_S=$(awk '{s+=$2} END{printf "%.3f", s}' "$INPUTS")
REPEATS=$(python3 -c "import math,sys; print(max(1, math.ceil((float(sys.argv[1])+15)/float(sys.argv[2]))))" "$SEC" "$PASS_S")
echo "side_asr: $ROW  pass ${PASS_S}s x $REPEATS repeats (run ${SEC}s + 15 s) realtime=1"
T0=$(date +%s.%N)
# stdbuf: the driver's per-clip lines are the readiness signal and must reach driver.log as they happen (block-buffered when redirected otherwise)
( cd "$(dirname "$INPUTS")" && stdbuf -oL -eL "$BIN" "$ENGINE_DIR" "$PRECISION" "$INPUTS" "$OUT/out.jsonl" "$REPEATS" 1 $ARGS ) > "$OUT/driver.log" 2>&1
RC=$?
python3 - "$OUT" "$ROW" "$RC" "$PASS_S" "$REPEATS" "$SEC" "$T0" "${ROW_START_FILE:-}" <<'PY'
import json, sys, statistics as st, time, os
out, row, rc, pass_s, rep, sec, t0, sf = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]), sys.argv[8]
recs = []
try: recs = [json.loads(l) for l in open(f'{out}/out.jsonl') if l.strip()]
except Exception: pass
def pct(v, p):
    v = sorted(v); return v[min(len(v) - 1, max(0, int(round(p * len(v) + 0.5)) - 1))] if v else None
r = {'name': row, 'role': 'side', 'runtime': 'asr_driver', 'driver_rc': rc, 'clips': len(recs), 'planned_s': pass_s * rep, 'pass_s': pass_s, 'repeats': rep,
     'wall_s': round(time.time() - t0, 1), 'realtime': True, 'status': 'OK' if (rc == 0 and recs) else 'FAILED', 'why': None if (rc == 0 and recs) else f'driver rc={rc}, {len(recs)} clips'}
if r['status'] != 'OK':
    try:   # first error line of the driver log, so the verdict names the cause (e.g. a kernel the MPS thread cap cannot load)
        err = next((l.strip() for l in open(f'{out}/driver.log', errors='ignore') if 'Error' in l or 'error' in l or 'fault' in l), None)
        if err: r['why'] += ' - ' + err[:300]
    except Exception: pass
def stats(rs):
    tot = [x['gpu_total_ms'] for x in rs]
    return {'ttft_ms_median': st.median([x['ttft_ms'] for x in rs]), 'ttft_ms_p99': pct([x['ttft_ms'] for x in rs], .99),
            'decode_ms_median': st.median([x['decode_ms_median'] for x in rs]), 'decode_ms_p99_max': max(x['decode_ms_p99'] for x in rs),
            'decode_ms_p99_median': st.median([x['decode_ms_p99'] for x in rs]),
            'gpu_total_ms_median': st.median(tot), 'gpu_total_ms_p99': pct(tot, .99), 'rtf_wall_median': st.median([x['rtf_wall'] for x in rs]),
            'rtf_wall_max': max(x['rtf_wall'] for x in rs), 'speech_s_total': sum(x['seconds'] for x in rs)}
if recs:
    r.update(stats(recs))
    r['summary'] = f"{len(recs)} clips: ttft med {r['ttft_ms_median']:.1f} ms, decode p99 max {r['decode_ms_p99_max']:.2f} ms, rtf med {r['rtf_wall_median']:.3f} max {r['rtf_wall_max']:.3f}"
    start = None
    try: start = int(open(sf).read().split()[0]) if sf and os.path.exists(sf) else None
    except Exception: start = None
    if start is not None:
        win = [x for x in recs if start <= x['t_start_ns'] < start + int(sec * 1e9)]
        r['window_clips'] = len(win); r['window_s'] = sec; r['window_start_ns'] = start
        if win:
            r.update({'window_' + k: v for k, v in stats(win).items()})
            r['summary'] += f" | in the {sec:g} s frame window ({len(win)} clips): ttft med {r['window_ttft_ms_median']:.1f} ms, decode p99 max {r['window_decode_ms_p99_max']:.2f} ms, rtf med {r['window_rtf_wall_median']:.3f} max {r['window_rtf_wall_max']:.3f}"
json.dump(r, open(f'{out}/side_result.json', 'w'), indent=1); print(r.get('summary') or r['why'])
PY
exit $RC
