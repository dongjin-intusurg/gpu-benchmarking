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
  configs/            mix and manifest templates             (available; device constants pending)
  ADDING_A_MODEL.md   how to describe a new model            (available)
  setup.sh            symlinks, helper builds, resolve check (available)
  scripts/            measurement code (C++ helpers available;
                      the measurement stages pending)
  results/            small, reviewable result files         (pending)
  figs/               figures regenerated from results/      (pending)
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
shipped `/path/to/...` template rows are skipped and counted separately.

Two helpers ship now: `scripts/model_bench/cpp/row_loop.cpp`, the paced engine
loop the co-location stages drive (no Python in the timed path), and
`scripts/device_ceilings/peak_issue_probe.cu`, the tensor issue-rate probe.
`setup.sh` writes `ROW_LOOP` into `.setup_env` only when the binary really
exists, so a downstream stage never inherits a path to something unbuilt.

Exits non-zero on a blocking problem. A desktop session or a non-maximum power
mode is a warning, not a block — those matter for certified numbers, not setup.

## 3. Device ceilings _(pending)_

Per-precision GEMM sweep, bandwidth kernel suite, CUDA-core and sustained-vs-
burst measurements under locked, verified clocks with drift adjudication.
Produces the ceilings every budget divides by.

## 4. Per-model measurement _(pending)_

Engine build, accuracy gate against an fp32 reference built from the same
ONNX, certified p99, causal bytes/frame via the memory-clock dial, NCU
roofline, and the per-model N / C / L / Score.

## 5. Co-location and paced runs _(pending)_

Multi-model arms (plain / MPS / streams / MIG where available), shared-trigger
period makespan, and the composed budget for a target mix.

## 6. Power _(pending)_

Power-mode and power-cap sweeps; compute-per-watt and bandwidth-per-watt fits.

## 7. Serving runtime (LLM / VLM / ASR) _(pending)_

Quantization ladder, decode dial, teacher-forced fidelity gate, and streaming
ASR end-to-end on TensorRT Edge-LLM (Jetson) or TensorRT-LLM (discrete).

## 8. Figures and reports _(pending)_

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
