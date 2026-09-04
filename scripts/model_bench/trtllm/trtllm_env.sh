# TensorRT-LLM stack for the discrete capture: isolated venv + a root-less OpenMPI runtime.
export TRTLLM_VENV="$HOME/venv_trtllm"; export TRTLLM_PY="$TRTLLM_VENV/bin/python"
export TRTLLM_SYS="$HOME/venv_trtllm_sysdeps/root/usr"           # Ubuntu openmpi debs extracted without root (libmpi, orted, help files)
export OPAL_PREFIX="$TRTLLM_SYS"; export PATH="$TRTLLM_SYS/bin:$PATH"
export LD_LIBRARY_PATH="$TRTLLM_SYS/lib/x86_64-linux-gnu:$TRTLLM_SYS/lib/x86_64-linux-gnu/openmpi/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TRTLLM_SRC="$HOME/tools/TensorRT-LLM-main"; export MODELOPT_SRC="$HOME/tools/TensorRT-Model-Optimizer"
export TRTLLM_WS="$HOME/trtllm-workspace"                          # quantized checkpoints: <WS>/<model>_<qformat>/
export HF_HOME="${HF_HOME:-$HOME/hf_cache_edgellm}"
