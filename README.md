# gpu-benchmarking

Measurement kit for sizing AI inference workloads against NVIDIA GPUs — Jetson-class
modules and discrete cards — by measurement rather than datasheet. Every capacity
question is answered as demand / capacity, where both sides are measured on the
device under a locked, exclusive, provenance-stamped regime.

## Quick start — stage 0 to 6 on one machine

The same commands run on a Jetson module and on a discrete card; the scripts
detect the platform and pick the matching pieces (power tools, serving stack,
clock knobs). Every stage ends in a `report.md` (stages 4–6 under
`results/<stage>_<device_tag>_<stamp>/`) — that file is the deliverable of the
stage. §0–§7 below explain each stage in depth; this is the shortest path
through them.

**Before any measurement stage (3, 4, 5, 6):** work from ssh, take the desktop
down, and prime sudo — the clock lock needs root and the preflight refuses to
run with a display session or a foreign GPU client alive.

```bash
sudo systemctl isolate multi-user.target   # headless; ssh sessions survive (graphical.target brings the desktop back)
sudo -v                                    # keep it primed for long runs
```

### Step 0 — prerequisites

```bash
git clone <this repo> && cd gpu-benchmarking
./prereqs.sh                       # report what is missing
./prereqs.sh --install --tier all  # apt + pip + the serving runtime (Edge-LLM on Jetson, TensorRT-LLM on discrete)
```

`--tier measure` (the default) is enough for stages 3–6 with TensorRT engines
only; `--tier all` adds the generative stack.

### Step 1 — put the model assets in one place

```bash
mkdir -p ~/models            # or anywhere: export MODEL_ROOT=/path/to/models
# ~/models/<model>/{onnx,calib,samples,plugins}   one directory per model (copies or symlinks)
. ./env.sh && ./configure.sh # expand the config templates for this host; lists any unresolved path
```

### Step 2 — machine setup

```bash
./setup.sh --model-root $MODEL_ROOT --search $MODEL_ROOT   # symlinks, helper builds, resolve check
. ./.setup_env                                             # every later shell needs only this line
```

### Step 3 — device ceilings (the capacity side; once per device)

```bash
./scripts/device_ceilings/run_device_ceilings.sh
./scripts/device_ceilings/validate_ceilings.py scripts/results_<device_tag>_ceilings --device configs/device_configs/<device>.json
```

About 30–60 minutes; the run lands in `scripts/results_<device_tag>_ceilings/`
(`ceilings_report.txt` is the readable summary) and every later stage discovers
it by itself. `CEIL_SMOKE=1 ./scripts/device_ceilings/run_device_ceilings.sh` is
a 2-minute rehearsal.

### Step 4 — add a model and measure it solo (the demand side)

1. **Describe the model** (`ADDING_A_MODEL.md` §1–§4, ~15 minutes):
   - copy `configs/manifests/model_manifest.env.template` to
     `configs/manifests/model_manifest_<name>.env`; fill `MODEL_NAME`,
     `MODEL_ONNX`, `MODEL_PRECISION`, `MODEL_HZ`, `MODEL_DEADLINE_MS`, the
     calibration cache / plugins / shapes if the model has them — always through
     `${MODEL_ROOT}`;
   - write `configs/manifests/acc_manifest_<name>.json` (which outputs the
     accuracy gate compares, where the sample inputs are);
   - choose **how it is built** with `MODEL_BUILDERS` (§2b):
     `trt` — a TensorRT engine from the ONNX (default);
     `adopt` — an engine you already built (`MODEL_ENGINE`);
     `edgellm` — a generative model served by TensorRT Edge-LLM (Jetson);
     `trtllm` — the same on a discrete card by TensorRT-LLM;
     `MODEL_BUILDERS_jetson=edgellm` / `MODEL_BUILDERS_discrete=trtllm` declare both from one manifest.
     Generative models add `MODEL_CHECKPOINT` and the `MODEL_LLM_*` fields;
     multi-graph models add `MODEL_ENGINE_SET` (and optionally an end-to-end
     driver, `MODEL_E2E_SRC`).
2. **Register and check** — no GPU:
   ```bash
   ./configure.sh && ./setup.sh --model-root $MODEL_ROOT   # expand the new manifest, verify every path resolves
   ./scripts/run_model_solo.sh --list                     # what is registered and what it would build
   ```
3. **Build and measure:**
   ```bash
   ./scripts/run_model_solo.sh                   # validate -> build (unlocked) -> measure (locked) -> N, every model
   ./scripts/run_model_solo.sh --only <name>     # one model;  --skip-build: artifacts already built
   ```
   Engines build for minutes to an hour (generative quantization longer); the
   measure step is ~10–50 minutes per model (NCU dominates).
4. **Read** `results/solo_<device_tag>_<stamp>/report.md` — one row per
   model × builder × precision with p99, bytes/frame, VRAM, the budgets,
   U_max, C, L, **N = min(L, C)** and the cause tag; `results.json` carries
   the same rows for the next stages. Check the validity list in
   `ADDING_A_MODEL.md` §6 before believing a number.

### Step 5 — co-locate: register a mix and run it

1. **Register a mix** (`ADDING_A_MODEL.md` §7): copy
   `configs/colocation/template.csv` to `configs/colocation/<mix>.csv` — one
   line per stage-4 row: `role` `frame` (paced at `hz`) or `side` (ASR /
   generative, run by its own runtime), `deadline_ms`, an optional stream
   priority (`prio`) and MPS thread cap (`mps_pct`).
2. **Check** — no GPU:
   ```bash
   ./scripts/run_colocation.sh --list            # mixes, resolved rows, arms this platform supports
   ```
3. **Run:**
   ```bash
   ./scripts/run_colocation.sh                   # every mix x arm (plain / mps / streams / mig)
   ./scripts/run_colocation.sh --mix <mix> --arm plain   # one cell;  --skip-solo: reuse the paced solos
   ```
   ~90 s per paced solo and per arm (`RUN_SECONDS`); a mix of three rows
   under four arms is ~15 minutes.
4. **Read** `results/coloc_<device_tag>_<stamp>/report.md`: the mix × arm
   matrix (fits / makespan p99 / N measured vs predicted / worst row) and a
   per-row table per cell.

### Step 6 — power: register an operating point and repeat stage 5 there

1. **Register a point** (`ADDING_A_MODEL.md` §8) in
   `configs/power/points_jetson.csv` (`nvpmodel` modes) or
   `configs/power/points_discrete.csv` (`pl` watts, `lgc` MHz). The shipped
   tables already hold the baseline and one lower point per platform.
2. **Check** — no GPU:
   ```bash
   ./scripts/run_power.sh --list                 # points, mixes and arms per point, sweep args
   ```
3. **Run:**
   ```bash
   ./scripts/run_power.sh                        # every default point: knob -> lock -> stage-5 matrix -> per-watt sweep
   ./scripts/run_power.sh --point <p> --mix <m>  # one point, one mix
   ```
   Roughly the stage-5 time per point plus ~10 minutes of sweeps. A gated
   point (one that needs a reboot) runs last and tells you when to reboot.
4. **Read** `results/power_<device_tag>_<stamp>/report.md`: per point the
   paced-solo p99 / J per frame, the co-location cells and the per-watt
   ceilings, each as a ratio to the baseline point.

### Step 7 — figures and reports

Not in this repository yet; this guide grows by one step when it lands.

## Layout

```
gpu-benchmarking/
  prereqs.sh          check / install prerequisites
  env.sh              root variables every script reads
  configure.sh        expand config templates for this host
  configs/            mix / manifest / device-config / co-location / power-point tables
  ADDING_A_MODEL.md   how to describe a new model
  setup.sh            symlinks, helper builds, resolve check
  scripts/            measurement code, one directory per stage:
    device_ceilings/  §3 device ceilings
    run_model_solo.sh §4 per-model entry point
    model_bench/      §4 per-model: registry, build, measure, scorer
      cpp/            the C++ row loop and the end-to-end ASR driver
      trtllm/         the discrete generative stack helpers (TensorRT-LLM)
    colocation/       §5 co-location: mixes, compose, arms, verdict
      arms/           one script per arm (plain, mps, streams, mig)
    power/            §6 power: points, knobs, per-watt sweeps, verdict
    figures/          §7 figures and reports (not yet in the repository)
  results/            small, reviewable result files (not yet in the repository)
  figs/               figures regenerated from results/ (not yet in the repository)
```

## 0. Prerequisites — `prereqs.sh`

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

## 1. Model assets and path contract — `env.sh`, `configure.sh`

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

## 2. Machine setup — `setup.sh`

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

## 3. Device ceilings

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

## 4. Per-model measurement

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

## 5. Co-location and paced runs

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

## 6. Power

```bash
sudo -v                                  # the knobs and the clock lock need root; refuses without it
./scripts/run_power.sh --list            # points of this platform, mixes and arms per point, sweep args — no GPU
./scripts/run_power.sh                   # every default point: baseline first, then the excursions
./scripts/run_power.sh --point <p>       # one point (repeatable; ungated points always run before gated ones)
./scripts/run_power.sh --mix <m>         # one mix per point (repeatable; default: every mix without side rows)
./scripts/run_power.sh --skip-tops       # mixes only;  --skip-mixes: per-watt sweeps only
RUN_SECONDS=90 SWEEP_ARGS="--n 4096 --seconds 15" ./scripts/run_power.sh   # run length per solo/arm; sweep length
```

§3–§5 price a device at its default operating point. The power question is
the same measurement repeated at every operating point the product may ship
at: **what does a lower envelope cost in latency, what does it buy in joules,
and does the mix still fit.** Nothing is scaled or modelled from the baseline
— every point is re-measured under its own verified clock lock.

**Points** are registered once per platform, in `configs/power/points_<platform>.csv`
(`point,knob,value,default,gated,note`). The first `default=1` row is the
baseline: the device's default operating point (Jetson: the required power
mode; discrete: the default power limit, which is also the maximum — `-pl`
only dials down). Every other row is a deliberate excursion from it:

| platform | knob | value | point name |
|---|---|---|---|
| jetson | `nvpmodel` | mode id | the mode NAME `nvpmodel -q` reports (e.g. `MAXN`, `120W`) |
| discrete | `pl` | watts | `<value>W` — enforced with `nvidia-smi -pl`; out of the card's range → point skipped |
| discrete | `lgc` | MHz | `lgc<value>` — a graphics-clock cap held by `-lgc` for the point |

`gated=1` marks a point that alters the hardware configuration beyond clocks
(a Jetson mode that gates GPU units) and takes effect only after a reboot: the
kit runs gated points last, marks the boot, and refuses to measure an ungated
point until the machine has been rebooted — a point measured under a stale
gating mask would carry the wrong hardware silently.

Per point, in order:

- **apply and read back** (`power/knob_<platform>.sh`) — the knob is applied,
  read back, and the read-back must name the point (a Jetson mode is judged by
  the name `nvpmodel -q` reports, never by the exit status; a power limit by
  `power.limit`); a mismatch fails the point. The read-back is recorded as
  `readback.json`.
- **derive the device config** (`power/derive_device_config.py`) — the base
  device config with the lock targets and required power mode replaced by the
  point's own caps (Jetson: the devfreq `max_freq` of GPU and EMC; discrete:
  the power envelope or the `-lgc` cap). §5's preflight, lock verification,
  clock sampler and drift verdict then adjudicate the point against its own
  caps — the lock is proved at the point's clock, not at the nameplate.
- **paced solos and mixes** — §5's `measure_mixes.sh`, unchanged, on the
  selected mixes under the derived config: every frame row alone at its mix
  rate, then every arm (plain / mps / streams / mig), with the clock sampler
  (power rails, clocks, temperature, throttle events) on every process.
- **per-watt sweeps** (`power/per_watt_sweep.py`) — a paced GEMM per precision
  (fp16, int8, fp8) and a copy kernel, duty-cycled over 10–100 % busy at a
  fixed period, clocks **unlocked** (DVFS is what is being measured; a `-lgc`
  point holds its cap): module W and rail W sampled per target, fitted as
  `W = intercept + slope × throughput` over the ≥ 45 % busy points, with the
  saturation knee where a cap pins the board.

**Verdict** (`power/power_verdict.py`) — one column per point, the baseline
first: per paced row, p99 / miss % / mean W / **J per frame** (mean W × wall /
frames) and the marginal J (idle subtracted); per cell, makespan p99 / fits /
N_measured / per-row p99 / mean W / J per period; per precision, the fit, the
50 % and 100 % points and the delivered per-watt figure. Every excursion column
carries its ratios to the baseline (p99, W, J; makespan, N, fits changed;
slope, per-watt, delivered). Checks: baseline present, read-back names the
point, lock verified, fits linear (R² ≥ 0.95), idle power in band. Cells
INVALID at the baseline stay INVALID at every point for the same reason.

Output: `results/power_<device_tag>_<stamp>/` with `verdict.json`, `report.md`,
`provenance/` (state before, points table, the saved clock state) and one
directory per point: `readback.json`, `device_config.json`, `status.json`, a
full §5 run under `coloc/` and the sweep under `per_watt/`. The knob is
restored and the clocks released on exit, whatever happened.

## 7. Figures and reports

Regenerates every figure and table from `results/`; the derivation layer is
deterministic and diffable against the committed outputs. Not in this repository yet.

## Measurement discipline

- Locked, verified clocks for timing passes; nothing else on the GPU.
- Provenance with every number: device, driver / TensorRT / CUDA versions,
  clock state, date.
- Never benchmark while an engine is building.
- Plan against sustained ceilings, not burst.

Reference environment validated so far: Ubuntu 24.04 aarch64, JetPack R38,
TensorRT 10.13.3 / CUDA 13.0, torch 2.12 (cu130).
