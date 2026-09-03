# gpu-benchmarking — prerequisites

`prereqs.sh` checks (and optionally installs) everything a GPU benchmark run needs
on a fresh machine, and skips whatever is already present. Written for
Jetson-class and discrete NVIDIA GPUs on Ubuntu.

## Usage

```bash
./prereqs.sh                     # report only
./prereqs.sh --install           # apt + pip for whatever is missing (prompts for sudo)
./prereqs.sh --install --tier all
```

Tiers:

| | Covers |
|---|---|
| `--tier figures` | numpy, matplotlib, pillow, envsubst |
| `--tier measure` *(default)* | + TensorRT, CUDA, the dev/plugin headers, Nsight Systems/Compute, and (on Jetson) the power tools |
| `--tier all` | + the serving runtime (TensorRT Edge-LLM on Jetson, TensorRT-LLM on a discrete card) and the quantizer |

It exits non-zero when something blocks, so it chains: `./prereqs.sh && <next step>`.

## What it does

- **Reports versions** of every requirement, and prints the exact `apt` / `pip`
  command for anything missing. `--install` runs them.
- **Handles PEP 668**: installs Python packages into a directory (`PYL`, default
  `~/tools/bench-pylib`) and tells you to add it to `PYTHONPATH`. It also detects
  packages already importable through an existing `PYTHONPATH` rather than calling
  them missing.
- **Builds the serving runtime from source** when needed. On Jetson, `--install`
  clones and pins TensorRT Edge-LLM, detects the GPU compute capability for
  `-DCMAKE_CUDA_ARCHITECTURES`, cmake-builds it, and pip-installs the package —
  skipping any step already done, and never rewriting a checkout you already have.
  Override with `EDGELLM_ROOT`, `EDGELLM_REPO`, `EDGELLM_REF`, `CUDA_ARCH`, `PYL`.
- **Flags the environment**, not just packages: whether a desktop session is up,
  whether counter profiling is restricted, whether `sudo` is cached.

## Notes

- `trtexec` ships in `/usr/src/tensorrt/bin`; add it to `PATH`.
- Counter profiling needs root, or the driver option
  `NVreg_RestrictProfilingToAdminUsers=0`.
- On a discrete card, torch must be a CUDA build matching your toolkit.
- Record the serving-runtime version with your results — decode throughput
  depends on it.

Reference environment this was validated on: Ubuntu 24.04 aarch64, JetPack R38,
TensorRT 10.13.3 / CUDA 13.0, torch 2.12 (cu130).
