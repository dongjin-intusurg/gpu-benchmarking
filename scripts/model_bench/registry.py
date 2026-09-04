#!/usr/bin/env python3
"""The model registry: every configured model_manifest_*.env, parsed and checked.

A model is registered ONCE, in configs/manifests/model_manifest_<name>.env
(configure.sh expands it into scripts/manifests.local/). The stage-4 scripts
never read the manifests themselves - they ask this tool, so the defaults,
the platform x builder rule and the source checks live in one place.

  registry.py list      [--platform P]           human table
  registry.py validate  [--platform P]           every error at once; exit 1 on any
  registry.py json      [--platform P] [--out F] the resolved registry as JSON
  registry.py export <name> [--platform P]       R_* shell assignments for one model

Common options: --manifest-dir (default: <kit>/manifests.local), --engine-root,
--device-tag, --only <name> (repeatable).

Resolution rules (documented in the manifest template):
  builders    MODEL_BUILDERS_<platform> if set, else MODEL_BUILDERS, else "trt"
  precisions  MODEL_PRECISIONS if set, else MODEL_PRECISION
  engine set  MODEL_ENGINE_SET members; per-member fields MODEL_<FIELD>_<member>
  engine dir  MODEL_ENGINE_DIR if set, else <engine-root>/<device-tag>/<name>
  e2e         present iff MODEL_E2E_SRC is set

Platform x builder (a mismatch is an ERROR, never a warning):
                 trt   adopt   edgellm   trtllm
  jetson          ok    ok      ok        NO   (TensorRT-LLM does not target Jetson)
  discrete        ok    ok      NO        ok   (the discrete runtime of record is TensorRT-LLM)
"""
import argparse, glob, json, os, re, shlex, subprocess, sys

BUILDERS = ('trt', 'adopt', 'edgellm', 'trtllm')
PLATFORM_OK = {'jetson': {'trt', 'adopt', 'edgellm'},
               'discrete': {'trt', 'adopt', 'trtllm'}}
PRECISIONS = ('fp32', 'fp16', 'int8', 'fp8', 'nvfp4', 'nvfp4_nvflags')
GENERATIVE = {'edgellm', 'trtllm'}
MEMBER_RE = re.compile(r'^[A-Za-z0-9_]+$')


def source_env(path):
    """Source a manifest with bash and return the MODEL_*/ACC_* it defined."""
    # set -e: a line bash misreads (an unquoted value with spaces becomes a
    # command) must fail the registration, not silently drop the field
    script = 'set -ae; . "$1"; env -0'
    p = subprocess.run(['bash', '-c', script, '_', path], capture_output=True)
    if p.returncode != 0:
        err = p.stderr.decode('utf-8', 'replace').strip().splitlines()
        raise RuntimeError(f'{path}: bash could not source it - {err[-1] if err else "exit " + str(p.returncode)} '
                           '(quote values that contain spaces; plain KEY=value lines only)')
    out = {}
    for kv in p.stdout.split(b'\0'):
        if not kv or b'=' not in kv:
            continue
        k, v = kv.decode('utf-8', 'replace').split('=', 1)
        if k.startswith('MODEL_') or k.startswith('ACC_'):
            out[k] = v
    return out


def split_list(s):
    return [x.strip() for x in re.split(r'[,\s]+', s or '') if x.strip()]


def flags_value(v, errors, label):
    """A flag string, or '@file' holding one (long shape profiles). Comment
    lines (#) and line breaks in the file are folded to single spaces."""
    v = (v or '').strip()
    if not v.startswith('@'):
        return v
    path = v[1:]
    if not os.path.isfile(path):
        errors.append(f'{label}: flags file not found: {path}')
        return ''
    lines = [ln.strip() for ln in open(path) if ln.strip() and not ln.strip().startswith('#')]
    return ' '.join(lines)


def resolve(env, path, platform, engine_root, device_tag):
    """One manifest -> one registry record (no validation yet)."""
    name = env.get('MODEL_NAME', '').strip()
    rec = {'name': name, 'manifest': path, 'fields': env}
    b = env.get(f'MODEL_BUILDERS_{platform}', '').strip() or env.get('MODEL_BUILDERS', '').strip() or 'trt'
    rec['builders'] = split_list(b)
    rec['builders_source'] = (f'MODEL_BUILDERS_{platform}' if env.get(f'MODEL_BUILDERS_{platform}', '').strip()
                              else 'MODEL_BUILDERS' if env.get('MODEL_BUILDERS', '').strip() else 'default')
    prim = env.get('MODEL_PRECISION', '').strip()
    precs = split_list(env.get('MODEL_PRECISIONS', '')) or ([prim] if prim else [])
    rec['precisions'] = precs
    rec['primary_precision'] = prim or (precs[0] if precs else '')
    rec['members'] = split_list(env.get('MODEL_ENGINE_SET', ''))
    rec['engine_dir'] = env.get('MODEL_ENGINE_DIR', '').strip() or os.path.join(engine_root, device_tag, name)
    rec['hz'] = env.get('MODEL_HZ', '').strip()
    rec['deadline_ms'] = env.get('MODEL_DEADLINE_MS', '').strip()
    rec['arch_gflops'] = env.get('MODEL_ARCH_GFLOPS', '').strip()
    rec['onnx'] = env.get('MODEL_ONNX', '').strip()
    rec['checkpoint'] = env.get('MODEL_CHECKPOINT', '').strip()
    rec['calib_cache'] = env.get('MODEL_CALIB_CACHE', '').strip()
    rec['plugins'] = env.get('MODEL_PLUGINS', '').strip()
    rec['shapes'] = env.get('MODEL_SHAPES', '').strip()
    rec['extra_args'] = env.get('MODEL_EXTRA_ARGS', '').strip()
    rec['engine'] = env.get('MODEL_ENGINE', '').strip()          # adopt, single
    rec['acc_manifest'] = env.get('ACC_MANIFEST', '').strip()
    rec['file_errors'] = []
    rec['build_args'] = flags_value(env.get('MODEL_BUILD_ARGS', ''), rec['file_errors'], f'{name}: MODEL_BUILD_ARGS')
    rec['per_member'] = {}
    for m in rec['members']:
        rec['per_member'][m] = {
            'onnx': env.get(f'MODEL_ONNX_{m}', '').strip(),
            'engine': env.get(f'MODEL_ENGINE_{m}', '').strip(),
            'shapes': env.get(f'MODEL_SHAPES_{m}', '').strip(),
            'build_flags': flags_value(env.get(f'MODEL_BUILD_FLAGS_{m}', ''), rec['file_errors'],
                                       f'{name}: MODEL_BUILD_FLAGS_{m}'),
            'run_flags': env.get(f'MODEL_RUN_FLAGS_{m}', '').strip(),
            'arch_gflops': env.get(f'MODEL_ARCH_GFLOPS_{m}', '').strip(),
        }
    e2e_src = env.get('MODEL_E2E_SRC', '').strip()
    rec['e2e'] = None
    if e2e_src:
        rec['e2e'] = {'src': e2e_src,
                      'inputs': env.get('MODEL_E2E_INPUTS', '').strip(),
                      'args': env.get('MODEL_E2E_ARGS', '').strip(),
                      'repeats': env.get('MODEL_E2E_REPEATS', '').strip() or '3',
                      'hz': env.get('MODEL_E2E_HZ', '').strip(),
                      'deadline_ms': env.get('MODEL_E2E_DEADLINE_MS', '').strip()}
    rec['generative'] = None
    if any(x in GENERATIVE for x in rec['builders']):
        rec['generative'] = {'battery': env.get('MODEL_LLM_BATTERY', '').strip(),
                             'image': env.get('MODEL_LLM_IMAGE', '').strip(),
                             'chunk': env.get('MODEL_LLM_CHUNK', '').strip() or '8',
                             'context_len': env.get('MODEL_LLM_CONTEXT_LEN', '').strip() or '1024',
                             'reuse_len': env.get('MODEL_LLM_REUSE_LEN', '').strip() or '896',
                             'max_input_len': env.get('MODEL_LLM_MAX_INPUT_LEN', '').strip() or '4096',
                             'has_visual': (env.get('MODEL_LLM_VISUAL', '').strip() or '1') != '0',
                             'build_flags': env.get('MODEL_LLM_BUILD_FLAGS', '').strip()}
        for p in precs:
            rec['generative'][f'build_flags_{p}'] = env.get(f'MODEL_LLM_BUILD_FLAGS_{p}', '').strip()
    kind = 'engine'
    if rec['generative']:
        kind = 'generative'
    elif rec['members']:
        kind = 'multi_engine'
    rec['kind'] = kind
    return rec


def validate(rec, platform):
    """Return the list of errors for one record. Empty list = registered OK."""
    e = list(rec.get('file_errors', []))
    f = rec['fields']
    n = rec['name'] or os.path.basename(rec['manifest'])
    if not rec['name']:
        e.append(f'{n}: MODEL_NAME is empty - it names every engine, row and results directory')
    elif not MEMBER_RE.match(rec['name']):
        e.append(f'{n}: MODEL_NAME must be snake_case [A-Za-z0-9_] - it is embedded in file names')
    if platform not in PLATFORM_OK:
        e.append(f'{n}: unknown platform "{platform}" - device config must say jetson|discrete')
        return e
    for b in rec['builders']:
        if b not in BUILDERS:
            e.append(f'{n}: MODEL_BUILDERS entry "{b}" is not one of {"|".join(BUILDERS)}')
        elif b not in PLATFORM_OK[platform]:
            why = ('TensorRT-LLM does not target Jetson - register edgellm for this platform '
                   '(MODEL_BUILDERS_jetson=edgellm)' if b == 'trtllm' else
                   'the discrete runtime of record is TensorRT-LLM - register trtllm for this platform '
                   '(MODEL_BUILDERS_discrete=trtllm)' if b == 'edgellm' else 'not supported here')
            e.append(f'{n}: builder "{b}" is INVALID on platform "{platform}" ({rec["builders_source"]}): {why}')
    if not rec['precisions']:
        e.append(f'{n}: MODEL_PRECISION (or MODEL_PRECISIONS) is empty - it selects the builder flags and the ceiling the budgets use')
    for p in rec['precisions']:
        if p not in PRECISIONS:
            e.append(f'{n}: precision "{p}" is not one of {"|".join(PRECISIONS)}')
    for k, why in (('hz', 'MODEL_HZ is the mix-defined rate: the demand side of every budget'),
                   ('deadline_ms', 'MODEL_DEADLINE_MS is the L term of N = min(L, C)')):
        v = rec[k]
        try:
            float(v)
        except ValueError:
            e.append(f'{n}: {why} - got "{v}"')
    # sources per builder
    gen = rec['generative'] is not None
    for b in rec['builders']:
        if b == 'trt':
            if rec['members']:
                for m, pm in rec['per_member'].items():
                    if not pm['onnx']:
                        e.append(f'{n}: MODEL_ONNX_{m} is empty - the trt builder needs one ONNX per MODEL_ENGINE_SET member')
                    elif not os.path.isfile(pm['onnx']):
                        e.append(f'{n}: MODEL_ONNX_{m} not found: {pm["onnx"]}')
            else:
                if not rec['onnx']:
                    e.append(f'{n}: MODEL_ONNX is empty - the trt builder builds the candidate AND the fp32 reference from it')
                elif not os.path.isfile(rec['onnx']):
                    e.append(f'{n}: MODEL_ONNX not found: {rec["onnx"]}')
        elif b == 'adopt':
            if rec['members']:
                for m, pm in rec['per_member'].items():
                    if not pm['engine']:
                        e.append(f'{n}: MODEL_ENGINE_{m} is empty - "adopt" measures an engine the model repo built; name it')
                    elif not os.path.isfile(pm['engine']):
                        e.append(f'{n}: MODEL_ENGINE_{m} not found: {pm["engine"]}')
            else:
                if not rec['engine']:
                    e.append(f'{n}: MODEL_ENGINE is empty - "adopt" measures an engine the model repo built; name it')
                elif not os.path.isfile(rec['engine']):
                    e.append(f'{n}: MODEL_ENGINE not found: {rec["engine"]}')
            if len(rec['precisions']) != 1:
                e.append(f'{n}: "adopt" takes exactly one precision (the adopted engine has one) - got {rec["precisions"]}')
        elif b in GENERATIVE:
            if not rec['checkpoint']:
                e.append(f'{n}: MODEL_CHECKPOINT is empty - {b} quantizes/exports from the HF checkpoint directory')
            elif not os.path.isdir(rec['checkpoint']) and not os.path.isfile(os.path.join(rec['checkpoint'], 'config.json')):
                e.append(f'{n}: MODEL_CHECKPOINT is not a directory: {rec["checkpoint"]}')
            if b == 'edgellm':
                bat = rec['generative']['battery']
                if not bat:
                    e.append(f'{n}: MODEL_LLM_BATTERY is empty - the end-to-end llm_inference run needs a request battery json')
                elif not os.path.isfile(bat):
                    e.append(f'{n}: MODEL_LLM_BATTERY not found: {bat}')
            if b == 'trtllm':
                img = rec['generative']['image']
                if rec['generative']['has_visual'] and not img:
                    e.append(f'{n}: MODEL_LLM_IMAGE is empty - the TensorRT-LLM step sweep needs one image')
                elif img and not os.path.isfile(img):
                    e.append(f'{n}: MODEL_LLM_IMAGE not found: {img}')
    # the fallback rule: an entry AFTER a non-trt builder is a fallback target,
    # and the only fallback the kit can perform is trt from MODEL_ONNX
    for i, b in enumerate(rec['builders'][1:], 1):
        prev = rec['builders'][i - 1]
        if b == 'trt' and prev != 'trt' and not rec['onnx'] and not rec['members']:
            e.append(f'{n}: MODEL_BUILDERS lists trt as a fallback after {prev}, but MODEL_ONNX is empty - '
                     f'the fallback could never fire; set MODEL_ONNX or drop the entry')
        if b in GENERATIVE or b == 'adopt':
            e.append(f'{n}: "{b}" cannot be a fallback target (only trt from MODEL_ONNX is) - list it first or on its own')
    # optional sources that must exist when named
    for k, v in (('MODEL_CALIB_CACHE', rec['calib_cache']), ('MODEL_PLUGINS', rec['plugins']),
                 ('ACC_MANIFEST', rec['acc_manifest'])):
        if v and not os.path.isfile(v):
            e.append(f'{n}: {k} not found: {v}')
    for m in rec['members']:
        if not MEMBER_RE.match(m):
            e.append(f'{n}: MODEL_ENGINE_SET member "{m}" must be [A-Za-z0-9_] (it names files and fields)')
    if rec['members'] and gen:
        e.append(f'{n}: MODEL_ENGINE_SET is for TensorRT multi-graph models; a generative builder emits its own llm/visual set')
    if rec['e2e']:
        s = rec['e2e']['src']
        if not os.path.isfile(s):
            e.append(f'{n}: MODEL_E2E_SRC not found: {s}')
        elif not s.endswith(('.cpp', '.cc', '.cu')):
            e.append(f'{n}: MODEL_E2E_SRC must be a C++ source the kit compiles (.cpp/.cc/.cu): {s}')
        inp = rec['e2e']['inputs']
        if not inp:
            e.append(f'{n}: MODEL_E2E_INPUTS is empty - the driver needs an input list (mel list, battery json, image)')
        elif not os.path.exists(inp):
            e.append(f'{n}: MODEL_E2E_INPUTS not found: {inp}')
        if rec['members'] and not rec['e2e']['args'] and rec['kind'] == 'multi_engine':
            pass  # fixed argv contract needs nothing more
        for k in ('hz', 'deadline_ms'):
            v = rec['e2e'][k]
            if v:
                try:
                    float(v)
                except ValueError:
                    e.append(f'{n}: MODEL_E2E_{k.upper()} must be numeric - got "{v}"')
    return e


def discover(manifest_dir, only):
    files = sorted(glob.glob(os.path.join(manifest_dir, 'model_manifest_*.env')))
    if not files:
        # the single-model slot, honoured only when nothing else is registered
        single = os.path.join(manifest_dir, 'model_manifest.env')
        if os.path.isfile(single):
            files = [single]
    return files


def load(args):
    recs, errs = [], []
    if not os.path.isdir(args.manifest_dir):
        errs.append(f'manifest dir missing: {args.manifest_dir} - run ". ./env.sh && ./configure.sh" first')
        return recs, errs
    files = discover(args.manifest_dir, args.only)
    if not files:
        errs.append(f'no model registered: no model_manifest_*.env under {args.manifest_dir} '
                    f'(write configs/manifests/model_manifest_<name>.env, then ./configure.sh)')
    seen, all_names = {}, []
    for f in files:
        try:
            env = source_env(f)
        except RuntimeError as ex:
            errs.append(str(ex)); continue
        rec = resolve(env, f, args.platform, args.engine_root, args.device_tag)
        all_names.append(rec['name'])
        if args.only and rec['name'] not in args.only:
            continue
        if rec['name'] in seen:
            errs.append(f'{rec["name"]}: registered twice ({seen[rec["name"]]} and {f}) - one manifest per model')
        seen[rec['name']] = f
        recs.append(rec)
    if args.only:
        for o in args.only:
            if o not in seen:
                errs.append(f'--only {o}: no registered model of that name (registered: {", ".join(sorted(all_names)) or "none"})')
    return recs, errs


def rows_for(rec):
    """The measured rows a registration will produce - shown by `list`."""
    out = []
    for b in rec['builders']:
        for p in rec['precisions']:
            if rec['kind'] == 'generative':
                out.append(f'{rec["name"]}_{p} [{b}: step/TTFT/decode]')
            elif rec['members']:
                for m in rec['members']:
                    out.append(f'{rec["name"]}_{m}' + ('' if p == rec['primary_precision'] else f'_{p}') + f' [{b}: engine p99]')
            else:
                out.append(rec['name'] + ('' if p == rec['primary_precision'] else f'_{p}') + f' [{b}: engine p99]')
    if rec['e2e']:
        out.append(f'{rec["name"]}_e2e [driver: {os.path.basename(rec["e2e"]["src"])}]')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['list', 'validate', 'json', 'export'])
    ap.add_argument('name', nargs='?')
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument('--manifest-dir', default=os.path.join(here, '..', 'manifests.local'))
    ap.add_argument('--platform', default=os.environ.get('BENCH_PLATFORM', 'jetson'))
    ap.add_argument('--engine-root', default=os.environ.get('ENGINE_ROOT', os.path.join(here, '..', '..', 'engines')))
    ap.add_argument('--device-tag', default=os.environ.get('DEVICE_TAG', 'device'))
    ap.add_argument('--only', action='append', default=[])
    ap.add_argument('--out')
    a = ap.parse_args()
    a.manifest_dir = os.path.abspath(a.manifest_dir)
    a.engine_root = os.path.abspath(a.engine_root)
    if a.cmd == 'export' and a.name:
        a.only = [a.name]

    recs, errs = load(a)
    for r in recs:
        r['errors'] = validate(r, a.platform)
        r['rows'] = rows_for(r)
    n_err = len(errs) + sum(len(r['errors']) for r in recs)

    if a.cmd == 'json':
        doc = {'platform': a.platform, 'device_tag': a.device_tag, 'manifest_dir': a.manifest_dir,
               'errors': errs, 'models': recs}
        s = json.dumps(doc, indent=1)
        if a.out:
            open(a.out, 'w').write(s)
        else:
            print(s)
        return 1 if n_err else 0

    if a.cmd == 'export':
        if not a.name:
            ap.error('export needs a model name')
        if errs:
            for x in errs: print('# ERROR ' + x)
            return 1
        r = recs[0]
        q = shlex.quote

        def put(k, v): print(f'R_{k}={q(str(v))}')
        put('NAME', r['name']); put('KIND', r['kind']); put('MANIFEST', r['manifest'])
        put('BUILDERS', ' '.join(r['builders'])); put('PRECISIONS', ' '.join(r['precisions']))
        put('PRIMARY_PRECISION', r['primary_precision']); put('MEMBERS', ' '.join(r['members']))
        put('ENGINE_DIR', r['engine_dir']); put('HZ', r['hz']); put('DEADLINE_MS', r['deadline_ms'])
        put('ARCH_GFLOPS', r['arch_gflops']); put('ONNX', r['onnx']); put('CHECKPOINT', r['checkpoint'])
        put('CALIB_CACHE', r['calib_cache']); put('PLUGINS', r['plugins']); put('SHAPES', r['shapes'])
        put('EXTRA_ARGS', r['extra_args']); put('ENGINE', r['engine']); put('ACC_MANIFEST', r['acc_manifest'])
        put('BUILD_ARGS', r['build_args'])
        for m, pm in r['per_member'].items():
            for k, v in pm.items():
                put(f'{k.upper()}_{m}', v)
        if r['e2e']:
            for k, v in r['e2e'].items():
                put(f'E2E_{k.upper()}', v)
        else:
            put('E2E_SRC', '')
        if r['generative']:
            for k, v in r['generative'].items():
                put(f'LLM_{k.upper()}', (1 if v is True else 0 if v is False else v))
        put('ERRORS', '\n'.join(r['errors']))
        return 1 if r['errors'] else 0

    # list / validate
    print(f'registry: {a.manifest_dir}  platform={a.platform}  device_tag={a.device_tag}')
    for x in errs:
        print(f'  ERROR  {x}')
    for r in recs:
        tag = 'ERROR' if r['errors'] else 'ok'
        print(f'  [{tag:5}] {r["name"]:22} kind={r["kind"]:12} builders={",".join(r["builders"]):14} '
              f'precisions={",".join(r["precisions"])}')
        if a.cmd == 'list':
            print(f'          engine_dir: {r["engine_dir"]}')
            for row in r['rows']:
                print(f'          row  {row}')
        for x in r['errors']:
            print(f'          ERROR  {x}')
    ok = sum(1 for r in recs if not r['errors'])
    print(f'  {ok}/{len(recs)} registrations valid, {n_err} error(s)')
    return 1 if n_err else 0


if __name__ == '__main__':
    sys.exit(main())
