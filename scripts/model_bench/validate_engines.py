#!/usr/bin/env python3
"""Validate BUILT engines before anything is measured with them.

Builds run unlocked and are not timed, so this is a build-artifact gate, not a
performance one. It answers: is this engine the artifact we think it is, and
does it actually carry the precision we asked for?

Gates
  1. artifacts   - the engine file exists; its sha256 matches engines.sha256 when
                   that record is present; build_stamp exists (the record tying
                   the engine to one ONNX + precision + TensorRT version)
  2. precision   - the realized per-layer datatype census from *_profile.json.
                   A build silently falling back (an "int8" engine with no int8
                   layers) is the failure this catches; TensorRT mixes
                   precisions by design, so the gate is "the requested precision
                   is materially present", not "every layer is that precision".
  3. loadable    - the engine deserializes and runs a few iterations

Serving-runtime builds (Edge-LLM on Jetson, TensorRT-LLM on discrete) are
directories rather than single files: --serving <dir> checks the expected
artifacts are present and non-empty instead of gates 2-3.

Usage:
  validate_engines.py <engine.engine> [more.engine ...] [--precision int8]
  validate_engines.py --serving <engine_dir> [--serving <dir> ...]
Exit 0 = PASS, 2 = INVESTIGATE.
"""
import argparse, collections, hashlib, json, os, subprocess, sys

MAP = {'int8': 'Int8', 'fp8': 'Fp8', 'fp16': 'Half', 'fp32': 'Float'}


def sha256(p, buf=1 << 20):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(buf), b''):
            h.update(b)
    return h.hexdigest()


def census(engine):
    prof = engine[:-len('.engine')] + '_profile.json' if engine.endswith('.engine') else engine + '_profile.json'
    if not os.path.exists(prof):
        # the builder may keep censuses in a sibling directory
        alt = os.path.join(os.path.dirname(engine), '..', 'engine_profiles',
                           os.path.basename(prof))
        prof = alt if os.path.exists(alt) else None
    if not prof:
        return None, None
    try:
        d = json.load(open(prof))
    except Exception:
        return prof, None
    layers = d.get('Layers', d) if isinstance(d, dict) else d
    c = collections.Counter()
    for l in layers:
        if not isinstance(l, dict):
            continue
        for io in (l.get('Outputs') or []):
            c[io.get('Format/Datatype', '?')] += 1
    return prof, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('engines', nargs='*')
    ap.add_argument('--precision', help='requested precision: int8|fp8|fp16|fp32')
    ap.add_argument('--serving', action='append', default=[],
                    help='a serving-runtime engine DIRECTORY (Edge-LLM / TensorRT-LLM)')
    ap.add_argument('--min-frac', type=float, default=0.10,
                    help='minimum share of layer outputs at the requested precision (default 0.10)')
    ap.add_argument('--skip-load', action='store_true', help='skip the deserialize+run check')
    ap.add_argument('--extra', default='',
                    help='trtexec flags the engine needs to load (--staticPlugins=..., --shapes=...)')
    a = ap.parse_args()
    fails, warns, lines = [], [], []

    for eng in a.engines:
        name = os.path.basename(eng)
        # --- gate 1: artifacts ---
        if not os.path.exists(eng):
            fails.append(f'{name}: engine file missing ({eng})'); continue
        size_mb = os.path.getsize(eng) / 1e6
        lines.append(f'[1] artifact  {name:38} {size_mb:8.1f} MB')
        d = os.path.dirname(eng)
        rec = os.path.join(d, 'engines.sha256')
        if os.path.exists(rec):
            want = {os.path.basename(p): h for h, p in
                    (l.split() for l in open(rec).read().split('\n') if l.strip())}
            if name in want:
                got = sha256(eng)
                ok = got == want[name]
                (lines if ok else fails).append(
                    f'[1] sha256    {name:38} {"matches engines.sha256" if ok else "MISMATCH - engine changed since build"}')
        else:
            warns.append(f'{name}: no engines.sha256 record beside it - provenance is weaker')
        if not os.path.exists(os.path.join(d, 'build_stamp')):
            warns.append(f'{name}: no build_stamp - cannot tie this engine to an ONNX + precision + TRT version')

        # --- gate 2: realized precision ---
        prof, c = census(eng)
        if c:
            tot = sum(c.values()) or 1
            hist = '  '.join(f'{k} {v} ({v/tot*100:.0f}%)' for k, v in c.most_common())
            lines.append(f'[2] precision {name:38} {hist}')
            want_prec = a.precision
            if not want_prec:
                for p in MAP:
                    if p in name.lower():
                        want_prec = p; break
            if want_prec:
                key = MAP.get(want_prec)
                frac = c.get(key, 0) / tot
                if frac < a.min_frac:
                    fails.append(f'{name}: requested {want_prec} but only {frac*100:.0f}% of layer '
                                 f'outputs are {key} - the builder fell back to another precision')
                else:
                    lines.append(f'[2] requested {name:38} {want_prec}: {frac*100:.0f}% of outputs (>= {a.min_frac*100:.0f}%)  PASS')
        else:
            warns.append(f'{name}: no layer census (*_profile.json) - realized precision unverifiable')

        # --- gate 3: loadable ---
        if not a.skip_load:
            p = subprocess.run(['trtexec', f'--loadEngine={eng}', '--iterations=10',
                                '--warmUp=200', '--noDataTransfers'] + a.extra.split(),
                               capture_output=True, text=True)
            ok = p.returncode == 0 and 'PASSED' in p.stdout
            (lines if ok else fails).append(
                f'[3] loadable  {name:38} {"deserializes and runs" if ok else "FAILED to load/run (see trtexec output)"}')

    for d in a.serving:
        nm = os.path.basename(d.rstrip('/'))
        if not os.path.isdir(d):
            fails.append(f'{nm}: serving engine directory missing ({d})'); continue
        # a serving export nests its artifacts (engines_<prec>/llm/...), so walk
        # the tree - a one-level listing reports a populated export as empty
        files, big = [], []
        for root, _dirs, fs in os.walk(d):
            for f in fs:
                if f.startswith('.'):
                    continue
                fp = os.path.join(root, f)
                files.append(os.path.relpath(fp, d))
                try:
                    if os.path.getsize(fp) > 1e6:
                        big.append(os.path.relpath(fp, d))
                except OSError:
                    pass
        lines.append(f'[1] serving   {nm:38} {len(files)} files, {len(big)} >1MB')
        if not big:
            fails.append(f'{nm}: serving engine directory holds no substantial artifact - export did not produce engines')
        else:
            eng = [f for f in big if f.endswith(('.engine', '.plan', '.safetensors'))]
            lines.append(f'[2] serving   {nm:38} engines/weights: {", ".join(sorted(eng)[:3]) or "(none named .engine/.safetensors)"}')
            if not eng:
                warns.append(f'{nm}: large files present but none look like an engine or weight file')
        cfg = [f for f in files if f.endswith('.json')]
        if cfg:
            lines.append(f'[2] serving   {nm:38} config: {", ".join(sorted(cfg)[:3])}')
        else:
            warns.append(f'{nm}: no json config beside the serving engines - runtime version/quantization unrecorded')

    print('\n'.join(lines))
    for w in warns: print('  WARN  ' + w)
    for f in fails: print('  FAIL  ' + f)
    if fails: print('\nVERDICT: INVESTIGATE'); return 2
    if warns: print('\nVERDICT: PASS (with warnings)'); return 0
    print('\nVERDICT: PASS'); return 0


if __name__ == '__main__':
    sys.exit(main())
