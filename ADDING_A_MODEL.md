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

### 2b. Registration fields — how stage 4 builds and measures it

The same manifest is also the model's **registration** for the per-model stage
(`./scripts/run_model_solo.sh`). Everything below is optional: leaving all of
it empty means "one TensorRT engine from `MODEL_ONNX` at `MODEL_PRECISION`,
timed with `trtexec`". The template documents every field; the ones that
change what gets built are:

| Field | Meaning |
|---|---|
| `MODEL_BUILDERS` | `trt` (engine from ONNX), `adopt` (a repo-built engine, `MODEL_ENGINE`), `edgellm` (Jetson generative stack), `trtllm` (discrete generative stack). Comma-separated; **every entry becomes its own measured row**. `MODEL_BUILDERS_jetson` / `MODEL_BUILDERS_discrete` declare per platform from one manifest |
| `MODEL_PRECISIONS` | every precision to build and measure; `MODEL_PRECISION` stays the primary (row `<name>`), the rest are rows `<name>_<prec>` |
| `MODEL_ENGINE_SET` | a multi-graph model: `encoder,decoder_first,decoder_past` with `MODEL_ONNX_<member>`, `MODEL_BUILD_FLAGS_<member>`, `MODEL_RUN_FLAGS_<member>`, `MODEL_ARCH_GFLOPS_<member>`; every member is also timed on its own |
| `MODEL_BUILD_ARGS` | build-only `trtexec` flags (shape profiles); they enter the build stamp but never a timing run |
| `MODEL_ENGINE_DIR` | reuse a workspace that already holds valid artifacts (default `${ENGINE_ROOT}/<device_tag>/<name>`) |
| `MODEL_E2E_SRC`, `MODEL_E2E_INPUTS`, `MODEL_E2E_ARGS`, `MODEL_E2E_REPEATS`, `MODEL_E2E_HZ`, `MODEL_E2E_DEADLINE_MS` | an end-to-end C++ driver over all the engines → row `<name>_e2e` (plus `<name>_decode` for a decode loop) |
| `MODEL_CHECKPOINT`, `MODEL_LLM_*` | the generative fields: HF checkpoint dir, request battery / image, decode chunk, context and reuse lengths, extra build flags per precision |

Rules that the validator enforces (all errors are reported at once, before any
GPU work):

- **Platform × builder is a hard error, never a warning.** Jetson accepts
  `trt`, `adopt`, `edgellm`; a discrete card accepts `trt`, `adopt`, `trtllm`.
  Register the other stack and the run stops at stage 4.0 naming the platform.
- **Every referenced file must exist** — ONNX, checkpoint, calibration cache,
  plugins, driver source, input list, `@file` flag files.
- **A fallback needs a source.** An entry after a non-`trt` builder fires only
  when the declared builder fails *and* `MODEL_ONNX` is set; the row then
  records `builder_requested` vs `builder_used`.
- **Quote values with spaces.** The manifest is sourced by bash under `set -e`,
  so an unquoted `--a --b` fails registration instead of silently dropping
  `--b`. Long shape profiles go in a file: `MODEL_BUILD_FLAGS_<member>=@${KIT_ROOT}/configs/shapes/<file>.flags`
  (comment lines and line breaks are folded to spaces).
- **A pre-existing engine is never rebuilt.** If `<engine_dir>/<name>_<prec>.engine`
  already exists and is not something this kit built, it is load-gated at the
  declared precision, recorded `adopted: true` with its sha256, and used as is.
  Delete or relink the file to force a build. Generative workspaces are reused
  the same way via their build stamp (a changed checkpoint, precision or build
  flag changes the stamp and triggers a rebuild).

The end-to-end driver contract, if you supply one (`scripts/model_bench/cpp/asr_e2e.cpp`
is the reference implementation):

```
<bin> <engine_dir> <precision> <input_list> <out.jsonl> <repeats> 0 [MODEL_E2E_ARGS]
```

The driver loads `<engine_dir>/<member>_<precision>.engine` and writes one JSON
line per request with `gpu_total_ms` (the latency of record), `ttft_ms`,
`decode_ms_p99`, `wall_ms`, and optionally `encoder_ms`, `n_tokens`, `seconds`,
`rtf_wall`. It is compiled by the stage with the same `g++ … -lnvinfer
-lnvinfer_plugin -lcudart` line `setup.sh` uses.

Two more worked examples. A three-graph ASR model with its end-to-end driver:

```bash
MODEL_NAME=asr
MODEL_PRECISION=fp16
MODEL_ENGINE_SET=encoder,decoder_first,decoder_past
MODEL_ONNX_encoder=${ONNX_DIR}/asr/encoder_model.onnx
MODEL_ONNX_decoder_first=${ONNX_DIR}/asr/decoder_model.onnx
MODEL_ONNX_decoder_past=${ONNX_DIR}/asr/decoder_with_past_model.onnx
MODEL_BUILD_FLAGS_encoder=--shapes=input_features:1x128x3000
MODEL_BUILD_FLAGS_decoder_first=@${KIT_ROOT}/configs/shapes/asr_decoder_first.flags
MODEL_BUILD_FLAGS_decoder_past=@${KIT_ROOT}/configs/shapes/asr_decoder_past.flags
MODEL_ARCH_GFLOPS_encoder=2273.77
MODEL_HZ=20
MODEL_DEADLINE_MS=50
MODEL_E2E_SRC=${KIT_ROOT}/scripts/model_bench/cpp/asr_e2e.cpp
MODEL_E2E_INPUTS=${INPUTS_ROOT}/asr/mel_bin/list.txt
MODEL_E2E_REPEATS=3
MODEL_E2E_HZ=0
MODEL_E2E_DEADLINE_MS=50
```

A VLM measured on both platforms from one manifest, two precisions each:

```bash
MODEL_NAME=vlm_27b
MODEL_CHECKPOINT=${MODEL_ROOT}/checkpoints/vlm-27b
MODEL_BUILDERS_jetson=edgellm
MODEL_BUILDERS_discrete=trtllm
MODEL_PRECISION=fp8
MODEL_PRECISIONS=fp8,nvfp4
MODEL_ENGINE_DIR=${LLM_WORKSPACE}/vlm-27b
MODEL_LLM_BATTERY=${INPUTS_ROOT}/vlm/battery.json
MODEL_LLM_IMAGE=${INPUTS_ROOT}/vlm/sample.jpeg
MODEL_LLM_CHUNK=8
MODEL_HZ=10
MODEL_DEADLINE_MS=100
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
./setup.sh                          # links assets, expands paths, verifies every row resolves
./scripts/run_model_solo.sh --list  # what is registered and what it would build — no GPU
./scripts/run_model_solo.sh         # validate -> build (unlocked) -> measure (locked) -> N, every model
```

`./scripts/run_model_solo.sh --only <name>` runs one model (repeatable);
`--skip-build` measures artifacts that are already built. There are no other
arguments: device config, ceilings run, registry and workspaces are discovered
from the machine through the `env.sh` path contract. `sudo -v` first — the
measure step pins the clocks and refuses to run if it cannot.

`setup.sh` step 6 tells you before you start whether every path resolves —
use it as the check that you filled the manifest correctly, rather than
discovering a typo forty minutes into an engine build.

The stage then runs, for every registered model × builder × precision:

1. **validate** (`model_bench/validate_registry.sh`) — every registration
   error at once, no GPU.
2. **build** (`model_bench/build_models.sh`) — unlocked, since builds are not
   timed; every artifact is validated (engine deserialize + realized-precision
   census + sha256, or checkpoint presence for a serving stack) and the row
   set is written to `${ENGINE_ROOT}/<device_tag>/registry_rows/`.
3. **measure** (`model_bench/measure_models.sh`) — one verified clock lock for
   the whole run, released on exit. Engine rows: 1000-iteration `trtexec` p99,
   nsys timeline, NCU bytes, VRAM. End-to-end rows: the driver over the whole
   engine set. Generative rows: TTFT, decode tokens/s and the control step
   (visual + prefill + chunk × decode). Every row then goes through the same
   scorer (`model_bench/compute_budgets.py`): budgets, U_max, C, L,
   N = min(L, C), cause tag, Score.

Output lands in `results/solo_<device_tag>_<stamp>/` — `results.json` (one
row per measured configuration), `report.md`, `provenance/`, and the per-row
`engine_rows/`, `e2e/`, `generative/` directories with the raw logs.

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

## 7. Registering a mix — `configs/colocation/<mix>.csv`

Stage 5 co-locates rows that stage 4 has already measured. Copy
`configs/colocation/template.csv` to `configs/colocation/<mix>.csv` and keep
only real rows:

```
row,role,hz,deadline_ms,prio,mps_pct,note
depth_int8,frame,30,33.3,0,,paced at 30 Hz
tracker_int8,frame,30,33.3,-5,,short row; priority lever in the streams arm
asr_e2e,side,0,50,,35,real-time ASR; 35 % thread cap in the mps arm
vlm_nvfp4,side,X,100,,,generative step battery back-to-back; X = spec rate undecided
```

| column | meaning |
|---|---|
| `row` | a stage-4 row name (`engines/<tag>/registry_rows/*.jsonl`) that the newest `results/solo_<tag>_*/results.json` scored — that run supplies solo p99, bytes and VRAM; `SOLO_RESULTS=<results.json>` selects another |
| `role` | `frame`: an engine row paced by `row_loop` at `hz`. `side`: an end-to-end or generative row run by its own runtime (ASR driver in real time; generative battery back-to-back), reported by its harness, never in the makespan |
| `hz` | frame rows: the pacing rate (> 0). Side rows: `0` or a number for the composed table, or `X` = rate not decided (charged back-to-back, `time_share_at_spec` on the stage-4 rate grid) |
| `deadline_ms` | the mix deadline for this row (overrides the stage-4 placeholder) |
| `prio` | CUDA stream priority, streams arm only (numerically lower = higher; range from `row_loop --prio-range`, printed by `--list`); blank = 0 |
| `mps_pct` | `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` for this row's process, mps arm only; blank = uncapped |

Rules the validator enforces in one pass: every row registered and scored; at
least one frame row; no duplicates; `frame` only on engine rows and `side` only
on end-to-end / generative rows; `X` only on side rows; `prio` inside the
device's range; `mps_pct` in 1–100; an `--arm` the platform supports (Jetson:
plain, mps, streams — mig is recorded `unsupported` below the supporting L4T;
discrete: plain, mps, streams, mig).

`./scripts/run_colocation.sh --list` shows every mix resolved (engine path,
run flags, stage-4 solo p99 per row) and the arm set; `--mix <m>` limits a run
to one mix. The engines a mix uses are the stage-4 artifacts — nothing is built
here, and a row whose engine is missing fails validation rather than the run.

## 8. Registering an operating point — `configs/power/points_<platform>.csv`

Stage 6 repeats stage 5 at every operating point in the platform's point
table. Add a row; nothing else changes.

```
point,knob,value,default,gated,note
MAXN,nvpmodel,0,1,0,baseline: the mode the device config requires
120W,nvpmodel,1,1,0,GPU clock cap lowered (EMC unchanged); no gating
90W,nvpmodel,2,0,1,TPC gating mask; takes effect at boot - run last, reboot after
```

| column | meaning |
|---|---|
| `point` | the name the knob's read-back must report: Jetson — the mode NAME in `nvpmodel -q` (`NV Power Mode: <name>`); discrete — `<watts>W` for a `pl` row, `lgc<MHz>` for an `lgc` row. The run fails the point when the read-back disagrees |
| `knob` | `nvpmodel` (Jetson mode id), `pl` (`nvidia-smi -pl`, watts), `lgc` (`nvidia-smi -lgc`, MHz cap held for the point) |
| `value` | the argument to the knob |
| `default` | `1` = measured by a run without `--point`; the FIRST default row is the baseline every ratio is taken against — keep it the device's default operating point |
| `gated` | `1` = the point changes the hardware configuration beyond clocks and needs a reboot to take effect (Jetson modes with a GPU gating mask). Gated points run last; after one has been applied the kit refuses ungated points until the next boot |

Rules: names unique; every column present; a `pl` value outside the card's
`power.min_limit`–`power.max_limit` is skipped (status SKIPPED), not failed; a
`lgc` value above `clocks.max.graphics` likewise. `./scripts/run_power.sh
--list` prints the resolved table, the baseline, and the mixes and arms each
point will run. The mixes come from stage 5's registry
(`configs/colocation/*.csv`): by default the ones without side rows; `--mix`
selects any registered mix.

## 9. Figures and the report — nothing to register

Stage 7 (`./scripts/run_figures.sh`) draws whatever the newest runs contain:
a new model appears as a row in the solo figures and the report, a new mix
as a row of the co-location matrix, a new operating point as a column of the
power tables. No file names a row, a mix or a point — if a figure lacks your
model, the run it should come from is missing or invalid, and `--list` shows
which runs the stage resolved.
