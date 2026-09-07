#!/usr/bin/env python3
"""One TensorRT-LLM chunk step under Nsight Compute (stage 4, NCU_GENERATIVE=1).

Runs the same step trtllm_vlm_step.py sweeps (image -> vision -> prefill with reuse -> CHUNK decode tokens):
two warm steps, then exactly one step between cudaProfilerStart/Stop, so `ncu --profile-from-start off`
records that step alone. Three things Nsight needs that a plain run does not:
  - the executor must live in this process (TLLM_WORKER_USE_SINGLE_PROCESS=1, set by the caller);
  - the image tensors are built on the CPU: a CUDA-resident image reaches the executor as a CUDA-IPC
    shared-tensor handle, which cannot be reopened under Nsight ("invalid device context");
  - the executor's hang detector (300 s, not configurable) is raised: a replayed step takes minutes.
"""
import argparse, os, sys
ap = argparse.ArgumentParser()
ap.add_argument('--tool-dir', required=True, help='directory holding trtllm_vlm_step.py')
ap.add_argument('--model', required=True); ap.add_argument('--image', required=True)
ap.add_argument('--chunk', type=int, default=8); ap.add_argument('--warm', type=int, default=2)
ap.add_argument('--prompt', default='Describe this image in one sentence.')
a = ap.parse_args()
sys.path.insert(0, a.tool_dir)
import trtllm_vlm_step as T
import torch
import tensorrt_llm._torch.pyexecutor.py_executor as _pe
_HD = _pe.HangDetector
_pe.HangDetector = lambda timeout=None, on_detected=None, **kw: _HD(timeout=86400, on_detected=on_detected, **kw)
llm = T.build_llm(a.model, reuse=True, kv_frac=0.12)
from tensorrt_llm.inputs import default_multimodal_input_loader
inputs = default_multimodal_input_loader(tokenizer=llm.tokenizer, model_dir=a.model, model_type=T.model_type(a.model),
                                         modality="image", prompts=[a.prompt], media=[[a.image]],
                                         image_data_format="pt", device="cpu")
for entry in inputs:
    entry["prompt"] = T.close_think(entry["prompt"])
for i in range(a.warm):
    T.one_step(llm, inputs, a.chunk); print(f'warm step {i + 1} done', flush=True)
torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStart(); print('PROFILED STEP START', flush=True)
T.one_step(llm, inputs, a.chunk)
torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStop(); print('PROFILED STEP END', flush=True)
try:
    llm.shutdown()
except Exception as e:
    print('shutdown:', e)
