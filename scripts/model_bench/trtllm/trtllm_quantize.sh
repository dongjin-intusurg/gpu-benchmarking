#!/usr/bin/env bash
# Quantize a VLM HF checkpoint for TensorRT-LLM (modelopt hf_ptq, HF export) — runs in ~/venv_modelopt.
# Usage: trtllm_quantize.sh <hf_ckpt> <fp8|nvfp4> <export_dir> [calib_size=256]
# KV cache stays bf16 (--kv_cache_qformat none) for parity with the Edge-LLM ladder (fp16 KV); weights+activations per --qformat.
set -uo pipefail
CK="${1:?ckpt}"; Q="${2:?fp8|nvfp4}"; OUT="${3:?export dir}"; CAL="${4:-256}"
PY="$HOME/venv_modelopt/bin/python"; SRC="$HOME/tools/TensorRT-Model-Optimizer/examples/hf_ptq"
export HF_HOME="${HF_HOME:-$HOME/hf_cache_edgellm}" HF_HUB_ENABLE_HF_TRANSFER=0
[ -f "$OUT/config.json" ] && [ -n "$(ls "$OUT"/*.safetensors 2>/dev/null)" ] && { echo "have $OUT"; exit 0; }
mkdir -p "$OUT"; cd "$SRC"
echo "[$(date +%H:%M:%S)] quantize $(basename "$CK") -> $Q (calib $CAL) -> $OUT"
FMT=(--qformat "$Q" --kv_cache_qformat none); [ "$Q" = nvfp4 ] && FMT=(--recipe general/ptq/nvfp4_qwen35-kv_none)   # NVFP4 with the linear-attention in_proj kept bf16 (TRT-LLM 1.3 loader limitation); KV bf16
"$PY" hf_ptq.py --pyt_ckpt_path "$CK" "${FMT[@]}" --export_path "$OUT" --calib_size "$CAL" --calib_seq 512 --dataset "${CALIB_DATASET:-cnn_dailymail}" \
  --trust_remote_code --skip_generate --no-verbose 2>&1 | tee "$OUT/quantize.log" | grep -vE "it/s\]|^\s*$" | tail -5
[ -f "$OUT/config.json" ] && [ -n "$(ls "$OUT"/*.safetensors 2>/dev/null)" ] && { du -sh "$OUT" | cut -f1; echo "[$(date +%H:%M:%S)] done $OUT"; } || { echo "FATAL: export missing in $OUT (see quantize.log)"; exit 1; }
