# Adding a new model

Four files describe a model to the suite, and they live in two places: the
**assets** go under `MODEL_ROOT` on the machine, the **descriptions** go in this
repo under `configs/`. Nothing else needs editing.

## 1. Lay the assets out under `MODEL_ROOT`

`setup.sh` discovers `MODEL_ROOT`; everything a model needs lives beneath it.
The convention the existing models follow:

```
$MODEL_ROOT/<model_name>/
    onnx/        the source ONNX — the candidate AND the fp32 reference build from this one file
    calib/       int8/fp8 calibration cache, if the precision needs one
    samples/     real input frames for the accuracy battery (.npy / .png / .bin)
    plugins/     custom .so, if the graph has plugin layers
```

Engines are **not** placed here — they are built into `ENGINE_ROOT` by the
pipeline and are never committed.

Three rules that come from things that went wrong before:

- **One ONNX, both engines.** The quantized candidate and the fp32 reference must
  build from the *same* file, or the accuracy gate is comparing two models rather
  than two precisions.
- **A calibration cache is deployment provenance, not a build detail.** Use the
  cache the deployed model was blessed with. A regenerated cache with different
  scales is a different model: one such swap moved a detector output by 10%
  of its range. If you have no cache the build self-calibrates on synthetic data
  — timing stays valid, the gate will judge the row and probably fail it, and
  that failure is correct.
- **Real samples beat seeded-random.** Without `MODEL_SAMPLE_DIR` the battery
  falls back to seeded-random inputs and the result is stamped
  `inputs_source: "seeded-random"` — weaker evidence, and labelled as such.

## 2. `configs/manifests/model_manifest_<name>.env`

Copy `configs/manifests/model_manifest.env.template` and fill it. Reference every
asset through `${MODEL_ROOT}` — never an absolute path, or the repo stops being
portable.

| Field | Required for | Meaning |
|---|---|---|
| `MODEL_NAME` | engines | snake_case id; names the engines, the mix row and the results directory |
| `MODEL_ONNX` | engines | the source ONNX |
| `MODEL_PRECISION` | engines | `fp16` \| `int8` \| `fp8` \| `fp32` |
| `MODEL_HZ` | engines | rate from the frozen mix definition |
| `MODEL_DEADLINE_MS` | engines | deadline from the frozen mix definition |
| `ACC_MANIFEST` | gate | path to the accuracy manifest below |
| `MODEL_CALIB_CACHE` | – | calibration cache for int8/fp8 |
| `MODEL_PLUGINS` | – | plugin `.so`; rides every `trtexec` invocation |
| `MODEL_SHAPES` | – | `trtexec --shapes` syntax; **required if the ONNX has dynamic dims** |
| `MODEL_EXTRA_ARGS` | – | extra `trtexec` flags on every invocation |
| `MODEL_SAMPLE_DIR` | – | real inputs for the battery |
| `MODEL_ARCH_GFLOPS` | – | leave empty — computed by `compute_arch_gflops.py` |
| `MODEL_REPO_URL`, `MODEL_CHECKPOINT` | – | provenance only, but fill them |

Worked example — a detector, an int8 model with a calibration cache and no
plugins:

```bash
MODEL_NAME=detector
MODEL_ONNX=${MODEL_ROOT}/detector/onnx/detector_v2.onnx
MODEL_PRECISION=int8
MODEL_CALIB_CACHE=${MODEL_ROOT}/detector/calib/detector_v2.int8_calibration
MODEL_SAMPLE_DIR=${MODEL_ROOT}/detector/samples
ACC_MANIFEST=${KIT_ROOT}/scripts/manifests.local/acc_manifest_detector.json
MODEL_HZ=30
MODEL_DEADLINE_MS=33.3
```

And the plugin case — the depth variant carries a custom `.so`:

```bash
MODEL_PLUGINS=${MODEL_ROOT}/depth_model/plugins/libdepth_plugins.so
```

## 3. `configs/manifests/acc_manifest_<name>.json`

Tensor-level truth for the accuracy gate. This is the part that cannot be
guessed — it needs the model's real I/O spec.

| Field | Meaning |
|---|---|
| `inputs[]` | per input: `name`, `shape` (**static** — a dynamic dim is a hard error), `dtype`, `kind` (`image` \| `opaque`), `layout`, `range`, `samples[]` |
| `paired_inputs[]` | input names that must receive *identical* perturbations — stereo pairs; these also get hflip-swap and 4/8/14 px shift variants |
| `outputs[]` | per output: `abs_tol`, `rel_tol`, `gate` (true = counts toward the verdict). Defaults are 2% relative, 0.1% of range; integer outputs must match exactly |
| `gate.agree_frac_min` | `0.99` for this suite |
| `trtexec_extra` | flags appended to **every** gate invocation, the fp32 reference included |

**`samples[]` paths.** Write them as `${INPUTS_ROOT}/samples_<name>/...`.
`configure.sh` expands the JSON along with the `.env` files into
`scripts/manifests.local/`, and `ACC_MANIFEST` in the model manifest points at
that expanded copy (see the worked example). Relative sample paths resolve
against the manifest file's own directory — and the gate refuses loudly when a
sample is missing rather than silently substituting random data, which is the
behaviour you want but will stop you if the paths are wrong.

If the model needs a task metric rather than raw tensor agreement (pooled
detection recall/precision, valid-pixel disparity, WER), write a metrics plugin
under `scripts/model_bench/` and pass it with
`--metrics-plugin`. That is how a detector can report PASS\* under a pooled
detection rule while its raw agreement gate reads FAIL.

## 4. `configs/mixes/mix_<device>_<name>.csv`

One row per model in the workload. Columns:

```
name,onnx,precision,hz,deadline_ms,arch_gflops,extra
```

`onnx` holds the **engine** path for a measurement mix (the column name is
historical). Use `${ENGINE_ROOT}` and `${MODEL_ROOT}` placeholders — `configure.sh`
expands them into `scripts/mixes.local/` at setup time:

```
name,onnx,precision,hz,deadline_ms,arch_gflops,extra
detector,${ENGINE_ROOT}/engines_detector/detector_int8.engine,int8,30,33.3,135.27,
depth_model,${ENGINE_ROOT}/engines_depth_model/depth_model_int8.engine,int8,30,33.3,17.09,--staticPlugins=${MODEL_ROOT}/depth_model/plugins/libdepth_plugins.so
```

A co-location mix is the same file with several rows. Rate and deadline come from
the frozen mix definition, not from what the model happens to achieve.

## 5. Run it

```bash
./setup.sh                                   # links assets, expands paths, verifies every row resolves
MODEL_MANIFEST=configs/manifests/model_manifest_<name>.env ./scripts/run_model_bench.sh
```

`setup.sh` step 6 tells you before you start whether every path resolves —
use it as the check that you filled the manifest correctly, rather than
discovering a typo forty minutes into an engine build.

The pipeline then runs: engines → accuracy gate → causal dial → measure → package,
writing to `results_<device>_<model_name>/`.

## 6. What to check before believing the numbers

1. `provenance/preflight.json` — verdict proceed, `smoke_only: false`.
2. `lock_verified.json` PASS, drift verdict PASS (or the pre-declared
   power-governed WARN on a discrete card).
3. Accuracy gate PASS, and `inputs_source` says `samples`, not `seeded-random`.
4. Engine sha256 and build stamp match the manifest ONNX.
5. The bytes-per-frame the dial reports is below the physical bound
   `p99 × bw_eff` — the pipeline clamps and flags it otherwise.

See `REPRODUCIBILITY.md` for the error bands to expect on each quantity, and
`METHOD_NOTES.md` for the measurement rules and the gotchas behind them.
