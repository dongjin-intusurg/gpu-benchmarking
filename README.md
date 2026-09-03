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
  setup.sh            path contract, symlinks, helper builds (pending)
  env.sh              root variables every script reads      (pending)
  configure.sh        expand config templates for this host  (pending)
  scripts/            measurement, analysis and figure code  (pending)
  configs/            device constants, mixes, manifests     (pending)
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

Exits non-zero when something blocks, so it chains: `./prereqs.sh && <next step>`.

Notes: `trtexec` ships in `/usr/src/tensorrt/bin` (add to `PATH`); counter
profiling needs root or `NVreg_RestrictProfilingToAdminUsers=0`; on a discrete
card torch must be a CUDA build matching the toolkit; record the serving-runtime
version with results, decode throughput depends on it.

## 1. Model assets and path contract _(pending)_

How to lay out model repositories, ONNX files, calibration caches and sample
inputs under `MODEL_ROOT`, and how `env.sh` / `configure.sh` resolve the
`${...}` placeholders in the config files so the same scripts run unchanged on
any host.

## 2. Machine setup — `setup.sh` _(pending)_

Discovers the model root, links engine directories, sources the environment,
builds the C++ helpers (row loop, streaming ASR harness, issue-rate probe) at
the detected compute capability, and verifies every config row resolves.

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
