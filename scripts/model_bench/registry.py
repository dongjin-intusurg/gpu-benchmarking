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
import argparse
import glob
import json
import os
import re
import shlex
import subprocess
import sys

BUILDERS = ('trt', 'adopt', 'edgellm', 'trtllm')
PLATFORM_OK = {'jetson': {'trt', 'adopt', 'edgellm'},
               'discrete': {'trt', 'adopt', 'trtllm'}}
PRECISIONS = ('fp32', 'fp16', 'int8', 'fp8', 'nvfp4', 'nvfp4_nvflags')
GENERATIVE = {'edgellm', 'trtllm'}
MEMBER_RE = re.compile(r'^[A-Za-z0-9_]+$')

PLATFORM_MISMATCH_WHY = {
    'trtllm': ('TensorRT-LLM does not target Jetson - register edgellm for this platform '
               '(MODEL_BUILDERS_jetson=edgellm)'),
    'edgellm': ('the discrete runtime of record is TensorRT-LLM - register trtllm for this platform '
                '(MODEL_BUILDERS_discrete=trtllm)'),
}


def source_env(path):
    """Source a manifest with bash and return the MODEL_*/ACC_* variables it defined."""
    # set -e: a line bash misreads (an unquoted value with spaces becomes a command) must fail
    # the registration, not silently drop the field
    script = 'set -ae; . "$1"; env -0'
    proc = subprocess.run(['bash', '-c', script, '_', path], capture_output=True)
    if proc.returncode != 0:
        stderr_lines = proc.stderr.decode('utf-8', 'replace').strip().splitlines()
        reason = stderr_lines[-1] if stderr_lines else 'exit ' + str(proc.returncode)
        raise RuntimeError(f'{path}: bash could not source it - {reason} '
                           '(quote values that contain spaces; plain KEY=value lines only)')
    fields = {}
    for pair in proc.stdout.split(b'\0'):
        if not pair or b'=' not in pair:
            continue
        key, value = pair.decode('utf-8', 'replace').split('=', 1)
        if key.startswith('MODEL_') or key.startswith('ACC_'):
            fields[key] = value
    return fields


def split_list(text):
    return [item.strip() for item in re.split(r'[,\s]+', text or '') if item.strip()]


def flags_value(value, errors, label):
    """A flag string, or '@file' holding one (long shape profiles); comment lines and
    line breaks in the file fold to single spaces."""
    value = (value or '').strip()
    if not value.startswith('@'):
        return value
    path = value[1:]
    if not os.path.isfile(path):
        errors.append(f'{label}: flags file not found: {path}')
        return ''
    lines = [line.strip() for line in open(path) if line.strip() and not line.strip().startswith('#')]
    return ' '.join(lines)


def resolve_builders(fields, platform):
    """MODEL_BUILDERS_<platform> if set, else MODEL_BUILDERS, else 'trt' - with the source used."""
    platform_specific = fields.get(f'MODEL_BUILDERS_{platform}', '').strip()
    generic = fields.get('MODEL_BUILDERS', '').strip()
    if platform_specific:
        return split_list(platform_specific), f'MODEL_BUILDERS_{platform}'
    if generic:
        return split_list(generic), 'MODEL_BUILDERS'
    return ['trt'], 'default'


def resolve_generative(fields, precisions):
    generative = {'battery': fields.get('MODEL_LLM_BATTERY', '').strip(),
                  'image': fields.get('MODEL_LLM_IMAGE', '').strip(),
                  'chunk': fields.get('MODEL_LLM_CHUNK', '').strip() or '8',
                  'context_len': fields.get('MODEL_LLM_CONTEXT_LEN', '').strip() or '1024',
                  'reuse_len': fields.get('MODEL_LLM_REUSE_LEN', '').strip() or '896',
                  'max_input_len': fields.get('MODEL_LLM_MAX_INPUT_LEN', '').strip() or '4096',
                  'has_visual': (fields.get('MODEL_LLM_VISUAL', '').strip() or '1') != '0',
                  'build_flags': fields.get('MODEL_LLM_BUILD_FLAGS', '').strip()}
    for precision in precisions:
        generative[f'build_flags_{precision}'] = fields.get(f'MODEL_LLM_BUILD_FLAGS_{precision}', '').strip()
    return generative


def resolve(fields, path, platform, engine_root, device_tag):
    """One manifest -> one registry record (no validation yet)."""
    name = fields.get('MODEL_NAME', '').strip()
    record = {'name': name, 'manifest': path, 'fields': fields}
    record['builders'], record['builders_source'] = resolve_builders(fields, platform)
    primary = fields.get('MODEL_PRECISION', '').strip()
    precisions = split_list(fields.get('MODEL_PRECISIONS', '')) or ([primary] if primary else [])
    record['precisions'] = precisions
    record['primary_precision'] = primary or (precisions[0] if precisions else '')
    record['members'] = split_list(fields.get('MODEL_ENGINE_SET', ''))
    record['engine_dir'] = (fields.get('MODEL_ENGINE_DIR', '').strip()
                            or os.path.join(engine_root, device_tag, name))
    for key, field in (('hz', 'MODEL_HZ'), ('deadline_ms', 'MODEL_DEADLINE_MS'),
                       ('arch_gflops', 'MODEL_ARCH_GFLOPS'), ('onnx', 'MODEL_ONNX'),
                       ('checkpoint', 'MODEL_CHECKPOINT'), ('calib_cache', 'MODEL_CALIB_CACHE'),
                       ('plugins', 'MODEL_PLUGINS'), ('shapes', 'MODEL_SHAPES'),
                       ('extra_args', 'MODEL_EXTRA_ARGS'), ('engine', 'MODEL_ENGINE'),
                       ('acc_manifest', 'ACC_MANIFEST')):
        record[key] = fields.get(field, '').strip()
    record['file_errors'] = []
    record['build_args'] = flags_value(fields.get('MODEL_BUILD_ARGS', ''), record['file_errors'],
                                       f'{name}: MODEL_BUILD_ARGS')
    record['per_member'] = {}
    for member in record['members']:
        record['per_member'][member] = {
            'onnx': fields.get(f'MODEL_ONNX_{member}', '').strip(),
            'engine': fields.get(f'MODEL_ENGINE_{member}', '').strip(),
            'shapes': fields.get(f'MODEL_SHAPES_{member}', '').strip(),
            'build_flags': flags_value(fields.get(f'MODEL_BUILD_FLAGS_{member}', ''), record['file_errors'],
                                       f'{name}: MODEL_BUILD_FLAGS_{member}'),
            'run_flags': fields.get(f'MODEL_RUN_FLAGS_{member}', '').strip(),
            'arch_gflops': fields.get(f'MODEL_ARCH_GFLOPS_{member}', '').strip(),
        }
    e2e_src = fields.get('MODEL_E2E_SRC', '').strip()
    record['e2e'] = None
    if e2e_src:
        record['e2e'] = {'src': e2e_src,
                         'inputs': fields.get('MODEL_E2E_INPUTS', '').strip(),
                         'args': fields.get('MODEL_E2E_ARGS', '').strip(),
                         'repeats': fields.get('MODEL_E2E_REPEATS', '').strip() or '3',
                         'hz': fields.get('MODEL_E2E_HZ', '').strip(),
                         'deadline_ms': fields.get('MODEL_E2E_DEADLINE_MS', '').strip()}
    record['generative'] = None
    if any(builder in GENERATIVE for builder in record['builders']):
        record['generative'] = resolve_generative(fields, precisions)
    if record['generative']:
        record['kind'] = 'generative'
    elif record['members']:
        record['kind'] = 'multi_engine'
    else:
        record['kind'] = 'engine'
    return record


def check_file(errors, label, field, path):
    if path and not os.path.isfile(path):
        errors.append(f'{label}: {field} not found: {path}')


def validate_builders(record, platform, label, errors):
    for builder in record['builders']:
        if builder not in BUILDERS:
            errors.append(f'{label}: MODEL_BUILDERS entry "{builder}" is not one of {"|".join(BUILDERS)}')
        elif builder not in PLATFORM_OK[platform]:
            why = PLATFORM_MISMATCH_WHY.get(builder, 'not supported here')
            errors.append(f'{label}: builder "{builder}" is INVALID on platform "{platform}" '
                          f'({record["builders_source"]}): {why}')


def validate_precisions_and_rates(record, label, errors):
    if not record['precisions']:
        errors.append(f'{label}: MODEL_PRECISION (or MODEL_PRECISIONS) is empty - it selects the builder '
                      'flags and the ceiling the budgets use')
    for precision in record['precisions']:
        if precision not in PRECISIONS:
            errors.append(f'{label}: precision "{precision}" is not one of {"|".join(PRECISIONS)}')
    for key, why in (('hz', 'MODEL_HZ is the mix-defined rate: the demand side of every budget'),
                     ('deadline_ms', 'MODEL_DEADLINE_MS is the L term of N = min(L, C)')):
        value = record[key]
        try:
            float(value)
        except ValueError:
            errors.append(f'{label}: {why} - got "{value}"')


def validate_trt_sources(record, label, errors):
    if record['members']:
        for member, member_fields in record['per_member'].items():
            if not member_fields['onnx']:
                errors.append(f'{label}: MODEL_ONNX_{member} is empty - the trt builder needs one ONNX per '
                              'MODEL_ENGINE_SET member')
            else:
                check_file(errors, label, f'MODEL_ONNX_{member}', member_fields['onnx'])
        return
    if not record['onnx']:
        errors.append(f'{label}: MODEL_ONNX is empty - the trt builder builds the candidate AND the fp32 '
                      'reference from it')
    else:
        check_file(errors, label, 'MODEL_ONNX', record['onnx'])


def validate_adopt_sources(record, label, errors):
    adopt_why = '"adopt" measures an engine the model repo built; name it'
    if record['members']:
        for member, member_fields in record['per_member'].items():
            if not member_fields['engine']:
                errors.append(f'{label}: MODEL_ENGINE_{member} is empty - {adopt_why}')
            else:
                check_file(errors, label, f'MODEL_ENGINE_{member}', member_fields['engine'])
    elif not record['engine']:
        errors.append(f'{label}: MODEL_ENGINE is empty - {adopt_why}')
    else:
        check_file(errors, label, 'MODEL_ENGINE', record['engine'])
    if len(record['precisions']) != 1:
        errors.append(f'{label}: "adopt" takes exactly one precision (the adopted engine has one) - '
                      f'got {record["precisions"]}')


def validate_generative_sources(record, builder, label, errors):
    checkpoint = record['checkpoint']
    if not checkpoint:
        errors.append(f'{label}: MODEL_CHECKPOINT is empty - {builder} quantizes/exports from the HF '
                      'checkpoint directory')
    elif not os.path.isdir(checkpoint) and not os.path.isfile(os.path.join(checkpoint, 'config.json')):
        errors.append(f'{label}: MODEL_CHECKPOINT is not a directory: {checkpoint}')
    generative = record['generative']
    if builder == 'edgellm':
        if not generative['battery']:
            errors.append(f'{label}: MODEL_LLM_BATTERY is empty - the end-to-end llm_inference run needs a '
                          'request battery json')
        else:
            check_file(errors, label, 'MODEL_LLM_BATTERY', generative['battery'])
    if builder == 'trtllm':
        image = generative['image']
        if generative['has_visual'] and not image:
            errors.append(f'{label}: MODEL_LLM_IMAGE is empty - the TensorRT-LLM step sweep needs one image')
        else:
            check_file(errors, label, 'MODEL_LLM_IMAGE', image)


def validate_fallback_rule(record, label, errors):
    """An entry AFTER a non-trt builder is a fallback target, and the only fallback the kit can
    perform is trt from MODEL_ONNX."""
    builders = record['builders']
    for i, builder in enumerate(builders[1:], 1):
        previous = builders[i - 1]
        if builder == 'trt' and previous != 'trt' and not record['onnx'] and not record['members']:
            errors.append(f'{label}: MODEL_BUILDERS lists trt as a fallback after {previous}, but MODEL_ONNX '
                          'is empty - the fallback could never fire; set MODEL_ONNX or drop the entry')
        if builder in GENERATIVE or builder == 'adopt':
            errors.append(f'{label}: "{builder}" cannot be a fallback target (only trt from MODEL_ONNX is) - '
                          'list it first or on its own')


def validate_e2e(record, label, errors):
    e2e = record['e2e']
    src = e2e['src']
    if not os.path.isfile(src):
        errors.append(f'{label}: MODEL_E2E_SRC not found: {src}')
    elif not src.endswith(('.cpp', '.cc', '.cu')):
        errors.append(f'{label}: MODEL_E2E_SRC must be a C++ source the kit compiles (.cpp/.cc/.cu): {src}')
    inputs = e2e['inputs']
    if not inputs:
        errors.append(f'{label}: MODEL_E2E_INPUTS is empty - the driver needs an input list (mel list, '
                      'battery json, image)')
    elif not os.path.exists(inputs):
        errors.append(f'{label}: MODEL_E2E_INPUTS not found: {inputs}')
    for key in ('hz', 'deadline_ms'):
        value = e2e[key]
        if not value:
            continue
        try:
            float(value)
        except ValueError:
            errors.append(f'{label}: MODEL_E2E_{key.upper()} must be numeric - got "{value}"')


def validate(record, platform):
    """Return the list of errors for one record. Empty list = registered OK."""
    errors = list(record.get('file_errors', []))
    label = record['name'] or os.path.basename(record['manifest'])
    if not record['name']:
        errors.append(f'{label}: MODEL_NAME is empty - it names every engine, row and results directory')
    elif not MEMBER_RE.match(record['name']):
        errors.append(f'{label}: MODEL_NAME must be snake_case [A-Za-z0-9_] - it is embedded in file names')
    if platform not in PLATFORM_OK:
        errors.append(f'{label}: unknown platform "{platform}" - device config must say jetson|discrete')
        return errors
    validate_builders(record, platform, label, errors)
    validate_precisions_and_rates(record, label, errors)
    for builder in record['builders']:
        if builder == 'trt':
            validate_trt_sources(record, label, errors)
        elif builder == 'adopt':
            validate_adopt_sources(record, label, errors)
        elif builder in GENERATIVE:
            validate_generative_sources(record, builder, label, errors)
    validate_fallback_rule(record, label, errors)
    for field, path in (('MODEL_CALIB_CACHE', record['calib_cache']), ('MODEL_PLUGINS', record['plugins']),
                        ('ACC_MANIFEST', record['acc_manifest'])):
        check_file(errors, label, field, path)
    for member in record['members']:
        if not MEMBER_RE.match(member):
            errors.append(f'{label}: MODEL_ENGINE_SET member "{member}" must be [A-Za-z0-9_] (it names files '
                          'and fields)')
    if record['members'] and record['generative'] is not None:
        errors.append(f'{label}: MODEL_ENGINE_SET is for TensorRT multi-graph models; a generative builder '
                      'emits its own llm/visual set')
    if record['e2e']:
        validate_e2e(record, label, errors)
    return errors


def discover(manifest_dir):
    files = sorted(glob.glob(os.path.join(manifest_dir, 'model_manifest_*.env')))
    if files:
        return files
    # the single-model slot, honoured only when nothing else is registered
    single = os.path.join(manifest_dir, 'model_manifest.env')
    return [single] if os.path.isfile(single) else []


def load(args):
    records, errors = [], []
    if not os.path.isdir(args.manifest_dir):
        errors.append(f'manifest dir missing: {args.manifest_dir} - run ". ./env.sh && ./configure.sh" first')
        return records, errors
    files = discover(args.manifest_dir)
    if not files:
        errors.append(f'no model registered: no model_manifest_*.env under {args.manifest_dir} '
                      f'(write configs/manifests/model_manifest_<name>.env, then ./configure.sh)')
    manifest_of_name, all_names = {}, []
    for path in files:
        try:
            fields = source_env(path)
        except RuntimeError as exc:
            errors.append(str(exc))
            continue
        record = resolve(fields, path, args.platform, args.engine_root, args.device_tag)
        name = record['name']
        all_names.append(name)
        if args.only and name not in args.only:
            continue
        if name in manifest_of_name:
            errors.append(f'{name}: registered twice ({manifest_of_name[name]} and {path}) - '
                          'one manifest per model')
        manifest_of_name[name] = path
        # member rows are named <name>_<member>; one colliding with another registration's
        # name would share its results directory and overwrite it
        for member in (record.get('members') or []):
            row_name = f'{name}_{member}'
            if row_name in manifest_of_name and manifest_of_name[row_name] != path:
                errors.append(f'{name}: member row "{row_name}" collides with the model registered in '
                              f'{manifest_of_name[row_name]} - rename one')
            manifest_of_name[row_name] = path
        records.append(record)
    for wanted in args.only:
        if wanted not in manifest_of_name:
            errors.append(f'--only {wanted}: no registered model of that name '
                          f'(registered: {", ".join(sorted(all_names)) or "none"})')
    return records, errors


def rows_for(record):
    """The measured rows a registration will produce - shown by `list`."""
    name = record['name']
    rows = []
    for builder in record['builders']:
        for precision in record['precisions']:
            suffix = '' if precision == record['primary_precision'] else f'_{precision}'
            if record['kind'] == 'generative':
                rows.append(f'{name}_{precision} [{builder}: step/TTFT/decode]')
            elif record['members']:
                for member in record['members']:
                    rows.append(f'{name}_{member}{suffix} [{builder}: engine p99]')
            else:
                rows.append(f'{name}{suffix} [{builder}: engine p99]')
    if record['e2e']:
        rows.append(f'{name}_e2e [driver: {os.path.basename(record["e2e"]["src"])}]')
    return rows


def print_export(record):
    """R_* shell assignments for one model, one per line, shell-quoted."""
    def put(key, value):
        print(f'R_{key}={shlex.quote(str(value))}')

    for key in ('NAME', 'KIND', 'MANIFEST'):
        put(key, record[key.lower()])
    put('BUILDERS', ' '.join(record['builders']))
    put('PRECISIONS', ' '.join(record['precisions']))
    put('PRIMARY_PRECISION', record['primary_precision'])
    put('MEMBERS', ' '.join(record['members']))
    for key in ('ENGINE_DIR', 'HZ', 'DEADLINE_MS', 'ARCH_GFLOPS', 'ONNX', 'CHECKPOINT', 'CALIB_CACHE',
                'PLUGINS', 'SHAPES', 'EXTRA_ARGS', 'ENGINE', 'ACC_MANIFEST', 'BUILD_ARGS'):
        put(key, record[key.lower()])
    for member, member_fields in record['per_member'].items():
        for key, value in member_fields.items():
            put(f'{key.upper()}_{member}', value)
    if record['e2e']:
        for key, value in record['e2e'].items():
            put(f'E2E_{key.upper()}', value)
    else:
        put('E2E_SRC', '')
    if record['generative']:
        for key, value in record['generative'].items():
            put(f'LLM_{key.upper()}', (1 if value is True else 0 if value is False else value))
    put('ERRORS', '\n'.join(record['errors']))


def print_table(args, records, errors, error_count):
    print(f'registry: {args.manifest_dir}  platform={args.platform}  device_tag={args.device_tag}')
    for error in errors:
        print(f'  ERROR  {error}')
    for record in records:
        tag = 'ERROR' if record['errors'] else 'ok'
        print(f'  [{tag:5}] {record["name"]:22} kind={record["kind"]:12} '
              f'builders={",".join(record["builders"]):14} precisions={",".join(record["precisions"])}')
        if args.cmd == 'list':
            print(f'          engine_dir: {record["engine_dir"]}')
            for row in record['rows']:
                print(f'          row  {row}')
        for error in record['errors']:
            print(f'          ERROR  {error}')
    valid = sum(1 for record in records if not record['errors'])
    print(f'  {valid}/{len(records)} registrations valid, {error_count} error(s)')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('cmd', choices=['list', 'validate', 'json', 'export'])
    parser.add_argument('name', nargs='?')
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--manifest-dir', default=os.path.join(here, '..', 'manifests.local'))
    parser.add_argument('--platform', default=os.environ.get('BENCH_PLATFORM', 'jetson'))
    parser.add_argument('--engine-root',
                        default=os.environ.get('ENGINE_ROOT', os.path.join(here, '..', '..', 'engines')))
    parser.add_argument('--device-tag', default=os.environ.get('DEVICE_TAG', 'device'))
    parser.add_argument('--only', action='append', default=[])
    parser.add_argument('--out')
    args = parser.parse_args()
    args.manifest_dir = os.path.abspath(args.manifest_dir)
    args.engine_root = os.path.abspath(args.engine_root)
    if args.cmd == 'export' and args.name:
        args.only = [args.name]

    records, errors = load(args)
    for record in records:
        record['errors'] = validate(record, args.platform)
        record['rows'] = rows_for(record)
    error_count = len(errors) + sum(len(record['errors']) for record in records)

    if args.cmd == 'json':
        doc = {'platform': args.platform, 'device_tag': args.device_tag, 'manifest_dir': args.manifest_dir,
               'errors': errors, 'models': records}
        text = json.dumps(doc, indent=1)
        if args.out:
            open(args.out, 'w').write(text)
        else:
            print(text)
        return 1 if error_count else 0

    if args.cmd == 'export':
        if not args.name:
            parser.error('export needs a model name')
        if errors:
            for error in errors:
                print('# ERROR ' + error)
            return 1
        record = records[0]
        print_export(record)
        return 1 if record['errors'] else 0

    print_table(args, records, errors, error_count)
    return 1 if error_count else 0


if __name__ == '__main__':
    sys.exit(main())
