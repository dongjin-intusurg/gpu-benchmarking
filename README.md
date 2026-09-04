# gpu-benchmarking

Measurement kit for sizing AI inference workloads against NVIDIA GPUs — Jetson-class
modules and discrete cards — by measurement rather than datasheet. Every capacity
question is answered as demand / capacity, where both sides are measured on the
device under a locked, exclusive, provenance-stamped regime.

Contents are being added in validated stages. Sections marked _(pending)_ describe
what will land there; the scripts are not in this repository yet.

## Layout

```
gpu-benchmarking/
  prereqs.sh          check / install prerequisites          (available)
  env.sh              root variables every script reads      (available)
  configure.sh        expand config templates for this host  (available)
  configs/            mix / manifest / device-config / co-location templates (available)
  ADDING_A_MODEL.md   how to describe a new model            (available)
  setup.sh            symlinks, helper builds, resolve check (available)
  scripts/            measurement code, one directory per stage:
    device_ceilings/  §3 device ceilings                             (available)
    run_model_solo.sh §4 per-model entry point                          (available)
    model_bench/      §4 per-model: registry, build, measure, scorer      (available)
      cpp/            the C++ row loop and the end-to-end ASR driver
      trtllm/         the discrete generative stack helpers (TensorRT-LLM)
    colocation/       §5 co-location: mixes, compose, arms, verdict         (available)
      arms/           one script per arm (plain, mps, streams, mig)
    power/            §6 power sweeps                                       (pending)
    figures/          §7 figures and reports                               (pending)
  results/            small, reviewable result files                       (pending)
  figs/               figures regenerated from results/                    (pending)
```

## 0. Prerequisites — `prereqs.sh` (available)

Checks (and optionally installs) everything a run needs on a fresh machine, and
skips whatever is already present.

```bash
./prereqs.sh                     # report only
./prereqs.sh --install           # apt + pip for whatever is missing (prompts for sudo)
./prereqs.sh --install --tier all
```

| Tier | Covers |
|---|---|
| `--tier figures` | numpy, matplotlib, pillow, envsubst |
| `--tier measure` *(default)* | + TensorRT, CUDA, the dev/plugin headers, Nsight Systems/Compute, and (on Jetson) the power tools |
| `--tier all` | + the serving runtime (TensorRT Edge-LLM on Jetson, TensorRT-LLM on a discrete card) and the quantizer |

- Reports the version of every requirement and prints the exact `apt` / `pip`
  command for anything missing; `--install` runs them.
- Handles PEP 668 by installing Python packages into a directory (`PYL`, default
  `~/tools/bench-pylib`) and detecting packages already importable through
  `PYTHONPATH` instead of calling them missing.
- Builds the serving runtime from source when needed: on Jetson, `--install`
  clones and pins TensorRT Edge-LLM, detects the compute capability for
  `-DCMAKE_CUDA_ARCHITECTURES`, cmake-builds it and pip-installs the package,
  skipping any step already done and never rewriting an existing checkout.
  Override with `EDGELLM_ROOT`, `EDGELLM_REPO`, `EDGELLM_REF`, `CUDA_ARCH`, `PYL`.
- Flags the environment, not just packages: desktop session up, counter
  profiling restricted, `sudo` cached.
- Runs one platform's checks and only that platform's: a Jetson never executes a
  discrete branch and a discrete card never executes a Jetson one. Detection is
  `/etc/nv_tegra_release`; `BENCH_PLATFORM=jetson|discrete` forces it so either
  path can be exercised from the other machine. `env.sh` exports the result as
  `PLATFORM` and gates its own platform-specific lines the same way.
- Resolves the TensorRT headers per platform rather than assuming one layout:
  JetPack keeps them in the multiarch include dir, a discrete box normally has a
  tarball whose headers sit beside the `trtexec` actually on `PATH` (which is what
  gets checked when several `/opt/tensorrt/<version>` trees are installed).
  Override the root with `TENSORRT_ROOT`. A tarball install is never offered an
  `apt` repair, because the `apt` candidate is often a different major version
  and would shadow it.
- Finds a serving runtime that lives outside the system interpreter. TensorRT-LLM
  is usually installed into its own venv with an out-of-tree MPI on
  `LD_LIBRARY_PATH`; set `BENCH_ENV_SH` to a file that establishes that
  environment and/or `BENCH_PY` to the interpreter. An untracked `.bench_env.sh`
  in the repository root is picked up automatically, so a machine can describe
  its own runtime without editing anything tracked. The stages must then run
  under that same environment.

Exits non-zero when something blocks, so it chains: `./prereqs.sh && <next step>`.

Notes: `trtexec` ships in `/usr/src/tensorrt/bin` on Jetson and in
`/opt/tensorrt/<version>/bin` for a discrete tarball — `env.sh` adds whichever
exists and never shadows one already on `PATH`; counter
profiling needs root or `NVreg_RestrictProfilingToAdminUsers=0`; on a discrete
card torch must be a CUDA build matching the toolkit; record the serving-runtime
version with results, decode throughput depends on it.

## 1. Model assets and path contract — `env.sh`, `configure.sh` (available)

Nothing in the repository carries a machine-specific path. Every script reads a
small set of roots from `env.sh`, and every config file refers to assets through
`${ROOT}` placeholders that `configure.sh` expands for the host it runs on. Moving
to a new machine means putting the model assets under one directory and, at most,
exporting that directory's path.

| Root | Default | Holds |
|---|---|---|
| `MODEL_ROOT` | `~/models` | model repositories, ONNX files, calibration caches — **the one you provide** |
| `INPUTS_ROOT` | `$MODEL_ROOT/inputs` | sample inputs for the accuracy gates (`samples_<model>/`) |
| `ONNX_DIR` | `$MODEL_ROOT/onnx` | per-family ONNX directories (`vfm/`, `va/`, `asr/`, …) |
| `ENGINE_ROOT` | `<repo>/engines` | built TensorRT engines — never committed, created on demand |
| `WORK_ROOT` | `<repo>/work` | engine-build scratch, quantization workspaces |
| `RESULTS_ROOT` / `FIGS_ROOT` / `CONFIG_ROOT` | `<repo>/results`, `figs`, `configs` | outputs and inputs of the derivation layer |
| `EDGELLM_ROOT` / `TRTLLM_ROOT` | `~/tools/...` | serving runtimes (Jetson / discrete) |
| `HANDOFF_ROOT` | `$RESULTS_ROOT/handoff` | result directories shipped in from another device |

Lay the assets out (real copies or symlinks — the scripts cannot tell):

```
$MODEL_ROOT/
    <model_a>/            each model repo as cloned: onnx/, calib/, samples/, plugins/
    <model_b>/
    onnx/<family>/        ONNX files for the family build scripts
    inputs/samples_<m>/   real frames for the accuracy battery
```

Then expand the templates and check that every input path resolves:

```bash
export MODEL_ROOT=/path/to/models        # only if not ~/models
. ./env.sh && ./configure.sh
```

`configure.sh` writes machine-local copies of `configs/mixes/*` and
`configs/manifests/*` into `scripts/mixes.local/` and `scripts/manifests.local/`
(git-ignored) and lists any placeholder that stayed unresolved — each one names
an empty root. Accuracy manifests list their samples as
`${INPUTS_ROOT}/samples_<model>/...`, and a model manifest's `ACC_MANIFEST`
points at the expanded copy under `scripts/manifests.local/`.

`configs/` ships the templates only: `mix_template.csv`, `mix_selftest.csv`,
`mix_ceilings_only.csv` and `model_manifest.env.template`. Your own model rows
go in files you add next to them — `ADDING_A_MODEL.md` walks through the four
files a model needs and the rules behind each field.

## 2. Machine setup — `setup.sh` (available)

One idempotent pass that puts a machine in a runnable state and then proves it:

```bash
./setup.sh --model-root $MODEL_ROOT --search $MODEL_ROOT --search <where your ONNX live>
. ./.setup_env          # later shells need only this
```

`--model-root` is normally required: discovery looks for a marker directory that
identifies a model tree, and the default name (`SETUP_MARKERS=model-assets`) will
not match your layout until you set it to one that does.

| § | Does |
|---|---|
| 1 | machine state: platform, power mode (Jetson) or power cap (discrete), online CPUs, desktop session, foreign GPU clients, stale VRAM |
| 2 | finds `MODEL_ROOT` by searching for a marker directory (`SETUP_MARKERS`) |
| 3 | links every `engines_*` tree it finds into `ENGINE_ROOT` and prints engine counts |
| 4 | sources `env.sh`, runs `configure.sh`, checks no placeholder is left |
| 5 | builds the C++ helpers against the TensorRT root derived from the `trtexec` on PATH, at the compute capability read from the device. Every helper reports — built, already built, or source not present — so a missing one is never a silent skip |
| 6 | opens every expanded mix and reports **`N/N paths this device needs resolve`** |

**§6 is the acceptance number**, and it is what makes the stage worth running: it
tells you before a forty-minute engine build whether a manifest has a typo.
Missing rows are listed by path — link or build them and re-run. Rows tagged for
another device (`mix_<tag>_*.csv`, tag from the platform or `MIX_TAG`) and the
shipped `/path/to/...` template rows are skipped and counted separately, and if
those are all there is, the stage says it verified nothing rather than reporting
a pass over an empty set.

Two helpers ship now: `scripts/model_bench/cpp/row_loop.cpp`, the paced engine
loop the co-location stages drive (no Python in the timed path), and
`scripts/device_ceilings/peak_issue_probe.cu`, the tensor issue-rate probe.
`setup.sh` writes `ROW_LOOP` into `.setup_env` only when the binary really
exists, so a downstream stage never inherits a path to something unbuilt.

Exits non-zero on a blocking problem. A desktop session or a non-maximum power
mode is a warning, not a block — those matter for certified numbers, not setup.

## 3. Device ceilings (available)

```bash
sudo -v                                  # the clock lock needs root; keep it primed
./scripts/device_ceilings/run_device_ceilings.sh          # no arguments
./scripts/device_ceilings/validate_ceilings.py results_<tag>_ceilings --device <cfg> [--reference <dir>]
```

Per-precision GEMM sweep (best-over-size is the ceiling, so the large sizes only
prove the peak — they cost most of the wall-clock and can be trimmed), a
bandwidth kernel suite (copy / read / write / triad at several working-set
sizes), CUDA-core fp32, and a sustained-vs-burst pass — all under locked,
verified clocks with a pre-declared drift verdict. Produces the ceilings every
budget divides by, plus a 7-section report with its own datasheet-fraction
sanity bands.

The stage refuses to measure outside the regime — desktop up, a foreign GPU
client, or the wrong operating point each block it before a clock is touched:

- **Jetson**: the required mode is the device config's `required_power_mode` (a
  run that measures at a lower mode declares it there).
- **Discrete**: the operating point is the default power limit, which is also
  the maximum; a stale `-pl <lower>` cap is refused, because dialing down is a
  deliberate sweep, not the baseline.

The device config is chosen from the machine itself (platform, GPU name, power
mode via `pick_device_config.py`), so the run takes **no arguments**. Results are
tagged with the power mode when it is not the default, so a run at one mode never
overwrites another's reference. A sudo keep-alive holds the credential for the
length of the sweep, and the clock-restore fails fast with an actionable message
rather than blocking on an unanswerable prompt if the cache ever lapses.

`validate_ceilings.py` turns a finished run into a PASS / INVESTIGATE verdict over
three gates: **regime integrity** (preflight + lock + drift; a drift FAIL or a
smoke run voids everything under it), **self-sanity** (each ceiling's
datasheet-fraction inside the config band — works with no reference at all), and
**reproducibility** (each ceiling within tolerance of a `--reference` run — the
gate that catches a peak that moved between runs; a GEMM ceiling legitimately
peaks at a mid size, so only a shift between runs is the signal, not the peak
itself).

**Device config.** The stage reads a per-device config (datasheet peaks, clock
targets, sanity bands) and picks the one matching this machine — platform, the
`nvidia-smi` name, and (Jetson) the current power mode. `configs/device_configs/`
ships a filled Jetson example and a `device_config.template.json`. For any other
card, copy the template to `<name>.local.json` (git-ignored) and fill the
`platform`, `device_name_match`, `datasheet` block and `required_power_mode` —
the datasheet peaks are what the sanity bands divide by, so they are required.

**Smoke check.** `CEIL_SMOKE=1 ./run_device_ceilings.sh` runs one GEMM size and
one precision with short passes — ~2 minutes, enough to prove preflight → lock →
verify → measure → drift → report is wired end to end. The numbers are **not**
ceilings (too few sizes to find a peak); it is a plumbing test. `CEIL_GEMM_SIZES`,
`TRT_GEMM_SIZES` and `TRT_GEMM_PRECISIONS` give finer control.

**If you hard-kill a run** (SIGKILL / `kill -9` of the group), the EXIT-trap clock
restore does not run and the clocks stay pinned — silently biasing whatever
measures next. Release by hand: Jetson `sudo jetson_clocks --restore
<run>/provenance/jetson_clocks_saved.conf` (or reboot); discrete `sudo nvidia-smi
-rgc -rmc`. Normal exit, `set -e`, and Ctrl-C are covered by the trap.

## 4. Per-model measurement (available)

```bash
sudo -v                                  # the measure step pins the clocks; it refuses without root
./scripts/run_model_solo.sh --list       # what is registered and what it would build — no GPU
./scripts/run_model_solo.sh              # validate -> build -> measure -> N, every registered model
./scripts/run_model_solo.sh --only <m>   # one model (repeatable);  --skip-build: artifacts exist
```

**One stage, every runtime — every registered model gets a solo measurement
here, and the number that comes out is the same whichever harness produces it:**
budgets per device (time, bandwidth, VRAM), U_max, C = 1/U_max,
L = deadline / p99, **N = min(L, C)**, Score = N × Σ(arch GFLOPs × Hz), with a
cause tag (throughput- vs latency-limited). A model is **registered once** in
its manifest (`ADDING_A_MODEL.md` §2b) with its sources, precisions and
builders; the stage builds every declared configuration and measures each as
its own row.

- **validate** (`model_bench/validate_registry.sh`) — no GPU; every
  registration error reported at once. The platform × builder rule is a hard
  error: Jetson takes `trt` / `adopt` / `edgellm`, a discrete card takes
  `trt` / `adopt` / `trtllm`.
- **build** (`model_bench/build_models.sh`) — **unlocked**, because builds are
  not timed. TensorRT engines from ONNX (single or a multi-graph engine set),
  adopted repo-built engines, Edge-LLM (Jetson) or TensorRT-LLM (discrete)
  quantize → export → build for generative models. Every artifact is validated
  before it can be measured; an artifact that already exists and stamps clean is
  reused, never rebuilt.
- **measure** (`model_bench/measure_models.sh`) — **one verified clock lock for
  the whole run**, released on exit. Engine rows: 1000-iteration `trtexec` p99,
  nsys timeline, NCU byte counters, VRAM sample. End-to-end rows: the model's
  own C++ driver over its whole engine set (streaming ASR: TTFT, decode-step
  p99, real-time factor). Generative rows: vision encoder, prefill, KV-reuse
  prefill, decode tokens/s, and the control step the deadline judges
  (visual + prefill + chunk × decode). All rows then go through the one scorer,
  `model_bench/compute_budgets.py`.

The scorer is deliberately the only place the N formula lives. A row with no
byte measurement is marked `budgets_not_measured` rather than scored as free on
the bandwidth axis; a row measured at 0 Hz (a request-driven model) is scored
against its deadline alone. The ceilings the budgets divide by come from the
newest §3 run on the machine — never re-measured here, never typed in.

Output: `results/solo_<device_tag>_<stamp>/` with `results.json` (one row per
measured configuration, carrying builder, precision, adopted / pre-existing
provenance, latency, bytes, VRAM and the solo block), `report.md`, the
provenance set (preflight, lock verification, saved clock state, drift), and the
raw per-row logs under `engine_rows/`, `e2e/`, `generative/`.

## 5. Co-location and paced runs (available)

```bash
sudo -v                                  # the measure step pins the clocks; it refuses without root
./scripts/run_colocation.sh --list       # mixes, resolved rows, arms this platform supports — no GPU
./scripts/run_colocation.sh              # validate -> compose -> paced solo -> arms -> verdict, every mix x arm
./scripts/run_colocation.sh --mix <m>    # one mix (repeatable);  --arm <a>: one arm (repeatable)
./scripts/run_colocation.sh --skip-solo  # reuse the newest run's paced solos: arms only
RUN_SECONDS=90 SOLO_RESULTS=<results.json> ./scripts/run_colocation.sh   # run length; a §4 run other than the newest
```

§4 prices every configuration alone. The sizing question is the composed one:
what happens when the rows of a target mix share one device. **A mix is
registered once**, as `configs/colocation/<mix>.csv` — rows referenced by their
§4 row name with a role, the mix's rate and deadline, and the per-arm levers
(`prio` for the streams arm, `mps_pct` for the MPS arm; see
`configs/colocation/template.csv`). Engines, run flags and precisions are
resolved from the registry rows and the newest §4 `results.json`, never
re-typed. A variant of a mix is just another CSV.

- `role=frame` — an engine row, paced by `row_loop` at its mix Hz (one
  process per row; one process with one stream per row in the streams arm).
  Every frame carries a launch/done trace.
- `role=side` — the model's own runtime alongside the frames: the ASR driver in
  real time, a generative battery back-to-back (Edge-LLM on Jetson, TensorRT-LLM
  on a discrete card). Reported by its own harness, never part of the makespan.
  `hz=X` on a side row means the spec rate is undecided: it is charged
  back-to-back and the composed table shows `time_share_at_spec` on the §4
  rate grid.

- **validate** (`colocation/validate_mixes.sh`) — no GPU; every error at once:
  unknown row, row absent from the §4 results, a frame role on a non-engine
  row, `X` on a frame row, `prio` outside the device's stream priority range
  (queried from `row_loop --prio-range`), an arm the platform lacks.
- **compose** (`colocation/compose_mix.py`) — no GPU; the arithmetic budget
  of the mix from §4 numbers and the §3 ceilings: Σ time / bandwidth / VRAM
  shares, U_max, C, L, N_predicted, cause, and a per-frame-row prediction of
  the contended p99 (`solo p99 / (1 − U_time of the other residents)`, null
  when the others saturate).
- **measure** (`colocation/measure_mixes.sh`) — **one verified clock lock for
  the run**, released on exit. Per mix: *paced solo* of every row at its mix
  rate with the same driver and duty (the contention reference; reused when it
  exists), then every arm — `plain` (one process per row, ready barrier before
  the first frame), `mps` (daemon started per arm into its own pipe directory,
  existence verified, per-row thread caps from the CSV), `streams` (one process,
  one prioritised stream per row), `mig` (discrete only; a Jetson below the
  supporting L4T records `unsupported`). Every arm writes the same
  `concurrent_mix.json` and a period makespan from the traces.
- **verdict** (`colocation/verdict.py`) — one cell per (mix, arm). `valid`:
  the rows actually overlapped (`aligned`), every row ran the C++ driver, traces
  and drift are in, the MPS daemon was verified where it should be — a cell
  that ran clean but never contended is `invalid`, a field, not a warning.
  `fits`: every frame row misses < 1 % of its deadlines and finishes before its
  own next trigger ≥ 99 % of the time, and every side row holds its rate (ASR
  RTF < 1). Per row: `contention_factor` = contended p99 / paced-solo p99 and
  the error of the composed prediction; per cell: makespan p99 vs period, free
  time = period − Σ paced-solo p99, `N_measured = min(L_contended, C_composed)`
  beside `N_predicted`, and the cause.

Output: `results/coloc_<device_tag>_<stamp>/` with `verdict.json`, `report.md`
(the mixes × arms matrix, the composed table per mix, a row table per cell),
`resolved/` and `composed/` per mix, `paced_solo/`, one directory per cell
(`<mix>/<arm>/concurrent_mix.json`, `period_makespan.json`, `row_*.json`,
`trace_*.csv`, `side_*/`), and the provenance set.

## 6. Power _(pending)_

Power-mode and power-cap sweeps; compute-per-watt and bandwidth-per-watt fits.
The baseline is the device's default operating point (Jetson: the required power
mode; discrete: the default power limit, which is also the maximum — `-pl` only
dials down). Every other point in the sweep is a deliberate excursion from it.

## 7. Figures and reports _(pending)_

Regenerates every figure and table from `results/`; the derivation layer is
deterministic and diffable against the committed outputs.

## Measurement discipline

- Locked, verified clocks for timing passes; nothing else on the GPU.
- Provenance with every number: device, driver / TensorRT / CUDA versions,
  clock state, date.
- Never benchmark while an engine is building.
- Plan against sustained ceilings, not burst.

Reference environment validated so far: Ubuntu 24.04 aarch64, JetPack R38,
TensorRT 10.13.3 / CUDA 13.0, torch 2.12 (cu130).
