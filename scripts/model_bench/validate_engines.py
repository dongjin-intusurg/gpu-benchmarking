#!/usr/bin/env python3
"""Gate BUILT engines before anything is measured with them (a build-artifact gate, not a timing one).

  validate_engines.py <engine.engine> [more ...] [--precision int8]   gates: [1] artifacts (file, sha256
      record, build_stamp)  [2] realized precision from the *_profile.json layer census  [3] loads and runs
  validate_engines.py --serving <engine_dir> [--serving <dir> ...]   a serving-runtime export: expected
      artifacts present and non-empty, instead of gates 2-3
Prints the gate lines, then '  WARN  ' / '  FAIL  ' lines, then a VERDICT line. Exit 0 PASS, 2 INVESTIGATE.
"""
import argparse
import collections
import hashlib
import json
import os
import subprocess
import sys

# TensorRT names an output format either as a bare datatype ("Float") or as a descriptive sentence
# ("Row major linear FP32", "Thirty-two wide channel vectorized row major Int8 format"). An exact
# lookup sees only the bare form, so a correctly built engine whose layers use a vectorized layout -
# which is what the int8 and fp8 fast paths actually emit - counted as 0% and was failed as a silent
# fallback. Match any alias as a case-insensitive substring.
PRECISION_ALIASES = {'int8': ('int8',), 'fp8': ('fp8',), 'fp16': ('fp16', 'half'), 'fp32': ('fp32', 'float')}

# TensorRT mixes precisions by design, so the gate is "the requested precision is materially
# present" (this share of layer outputs), not "every layer is that precision".
DEFAULT_MIN_FRAC = 0.10

SUBSTANTIAL_BYTES = 1e6


class Findings:
    def __init__(self):
        self.lines, self.warns, self.fails = [], [], []

    def gate(self, ok, line_if_ok, fail_if_not):
        (self.lines if ok else self.fails).append(line_if_ok if ok else fail_if_not)


def fraction_of(counter, aliases):
    """Share of layer outputs whose format names any of these aliases."""
    total = sum(counter.values()) or 1
    hits = sum(count for fmt, count in counter.items() if any(alias in str(fmt).lower() for alias in aliases))
    return hits / total


def sha256(path, chunk_bytes=1 << 20):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b''):
            digest.update(chunk)
    return digest.hexdigest()


def census(engine):
    """(profile path, Counter of output Format/Datatype) for the engine's layer census, or Nones."""
    stem = engine[:-len('.engine')] if engine.endswith('.engine') else engine
    profile = stem + '_profile.json'
    if not os.path.exists(profile):
        # the builder may keep censuses in a sibling directory
        sibling = os.path.join(os.path.dirname(engine), '..', 'engine_profiles', os.path.basename(profile))
        profile = sibling if os.path.exists(sibling) else None
    if not profile:
        return None, None
    try:
        doc = json.load(open(profile))
    except Exception:
        return profile, None
    layers = doc.get('Layers', doc) if isinstance(doc, dict) else doc
    formats = collections.Counter()
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        for output in (layer.get('Outputs') or []):
            formats[output.get('Format/Datatype', '?')] += 1
    return profile, formats


def recorded_hashes(record_path):
    return {os.path.basename(path): digest
            for digest, path in (line.split() for line in open(record_path).read().split('\n')
                                 if line.strip())}


def gate_artifacts(engine, name, findings):
    size_mb = os.path.getsize(engine) / 1e6
    findings.lines.append(f'[1] artifact  {name:38} {size_mb:8.1f} MB')
    engine_dir = os.path.dirname(engine)
    record = os.path.join(engine_dir, 'engines.sha256')
    if os.path.exists(record):
        wanted = recorded_hashes(record)
        if name in wanted:
            findings.gate(sha256(engine) == wanted[name],
                          f'[1] sha256    {name:38} matches engines.sha256',
                          f'[1] sha256    {name:38} MISMATCH - engine changed since build')
    else:
        findings.warns.append(f'{name}: no engines.sha256 record beside it - provenance is weaker')
    if not os.path.exists(os.path.join(engine_dir, 'build_stamp')):
        findings.warns.append(f'{name}: no build_stamp - cannot tie this engine to an ONNX + precision + '
                              'TRT version')


def requested_precision(args, name):
    if args.precision:
        return args.precision
    for precision in PRECISION_ALIASES:
        if precision in name.lower():
            return precision
    return None


def gate_precision(engine, name, args, findings):
    _profile, formats = census(engine)
    if not formats:
        findings.warns.append(f'{name}: no layer census (*_profile.json) - realized precision unverifiable')
        return
    total = sum(formats.values()) or 1
    histogram = '  '.join(f'{fmt} {count} ({count / total * 100:.0f}%)'
                          for fmt, count in formats.most_common())
    findings.lines.append(f'[2] precision {name:38} {histogram}')
    wanted = requested_precision(args, name)
    if not wanted:
        return
    aliases = PRECISION_ALIASES.get(wanted)
    fraction = fraction_of(formats, aliases) if aliases else 0.0
    findings.gate(fraction >= args.min_frac,
                  f'[2] requested {name:38} {wanted}: {fraction * 100:.0f}% of outputs '
                  f'(>= {args.min_frac * 100:.0f}%)  PASS',
                  f'{name}: requested {wanted} but only {fraction * 100:.0f}% of layer '
                  f'outputs are {wanted} - the builder fell back to another precision')


def gate_loadable(engine, name, args, findings):
    proc = subprocess.run(['trtexec', f'--loadEngine={engine}', '--iterations=10', '--warmUp=200',
                           '--noDataTransfers'] + args.extra.split(),
                          capture_output=True, text=True)
    findings.gate(proc.returncode == 0 and 'PASSED' in proc.stdout,
                  f'[3] loadable  {name:38} deserializes and runs',
                  f'[3] loadable  {name:38} FAILED to load/run (see trtexec output)')


def check_engine(engine, args, findings):
    name = os.path.basename(engine)
    if not os.path.exists(engine):
        findings.fails.append(f'{name}: engine file missing ({engine})')
        return
    gate_artifacts(engine, name, findings)
    gate_precision(engine, name, args, findings)
    if not args.skip_load:
        gate_loadable(engine, name, args, findings)


def serving_files(directory):
    """(all files, the substantial ones) relative to the export root. A serving export nests its
    artifacts (engines_<prec>/llm/...), so walk the tree - a one-level listing reports a populated
    export as empty."""
    files, substantial = [], []
    for root, _dirs, names in os.walk(directory):
        for filename in names:
            if filename.startswith('.'):
                continue
            path = os.path.join(root, filename)
            files.append(os.path.relpath(path, directory))
            try:
                if os.path.getsize(path) > SUBSTANTIAL_BYTES:
                    substantial.append(os.path.relpath(path, directory))
            except OSError:
                pass
    return files, substantial


def check_serving_dir(directory, findings):
    name = os.path.basename(directory.rstrip('/'))
    if not os.path.isdir(directory):
        findings.fails.append(f'{name}: serving engine directory missing ({directory})')
        return
    files, substantial = serving_files(directory)
    findings.lines.append(f'[1] serving   {name:38} {len(files)} files, {len(substantial)} >1MB')
    if not substantial:
        findings.fails.append(f'{name}: serving engine directory holds no substantial artifact - '
                              'export did not produce engines')
    else:
        weights = [path for path in substantial if path.endswith(('.engine', '.plan', '.safetensors'))]
        listed = ', '.join(sorted(weights)[:3]) or '(none named .engine/.safetensors)'
        findings.lines.append(f'[2] serving   {name:38} engines/weights: {listed}')
        if not weights:
            findings.warns.append(f'{name}: large files present but none look like an engine or weight file')
    configs = [path for path in files if path.endswith('.json')]
    if configs:
        findings.lines.append(f'[2] serving   {name:38} config: {", ".join(sorted(configs)[:3])}')
    else:
        findings.warns.append(f'{name}: no json config beside the serving engines - '
                              'runtime version/quantization unrecorded')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('engines', nargs='*')
    parser.add_argument('--precision', help='requested precision: int8|fp8|fp16|fp32')
    parser.add_argument('--serving', action='append', default=[],
                        help='a serving-runtime engine DIRECTORY (Edge-LLM / TensorRT-LLM)')
    parser.add_argument('--min-frac', type=float, default=DEFAULT_MIN_FRAC,
                        help='minimum share of layer outputs at the requested precision (default 0.10)')
    parser.add_argument('--skip-load', action='store_true', help='skip the deserialize+run check')
    parser.add_argument('--extra', default='',
                        help='trtexec flags the engine needs to load (--staticPlugins=..., --shapes=...)')
    args = parser.parse_args()

    findings = Findings()
    for engine in args.engines:
        check_engine(engine, args, findings)
    for directory in args.serving:
        check_serving_dir(directory, findings)

    print('\n'.join(findings.lines))
    for warning in findings.warns:
        print('  WARN  ' + warning)
    for failure in findings.fails:
        print('  FAIL  ' + failure)
    if findings.fails:
        print('\nVERDICT: INVESTIGATE')
        return 2
    if findings.warns:
        print('\nVERDICT: PASS (with warnings)')
        return 0
    print('\nVERDICT: PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
