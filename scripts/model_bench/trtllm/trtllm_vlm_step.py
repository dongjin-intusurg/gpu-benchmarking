#!/usr/bin/env python3
"""TensorRT-LLM VLM step tool for a discrete card: step = image + prompt -> vision -> prefill -> chunk tokens.

  probe    --model <ckpt> --image <jpg>                  load, 3 steps, print timings; writes <out>/probe.json
  sweep    ... --out <dir> --tag T --state S [--chunks 1,2,4,8] [--steps 30]   chunks x steps, reuse on/off ->
           <out>/sweep_<T>_<S>.json (compose schema, printed) + <out>/<T>_steps.jsonl (chunk 8)
  loop     ... --out <arm dir> [--chunk 8] [--warm 3]     back-to-back steps until SIGTERM; prints READY after
           warm-up, 'step K: X ms' every 20, 'DONE K steps'; appends <out>/vlm_e2e_steps.jsonl line-buffered
  accuracy --model <ckpt> --battery <json> --out <answers.json>   greedy answers for an Edge-LLM battery
Timings come from the runtime's RequestPerfMetrics (C++ executor clocks), never Python timers.
Run inside the TensorRT-LLM venv with the OpenMPI env (see trtllm_env.sh).
"""
import argparse
import atexit
import json
import os
import signal
import sys
import time

# avoid fragmentation blow-up under the VRAM policy cap; must precede the torch import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# The chat template opens a <think> block; close it (enable_thinking=False, as the Jetson Edge-LLM
# method does) so steps and accuracy measure ANSWER tokens.
NO_THINK = os.environ.get("TRTLLM_THINK", "0") != "1"

# RequestPerfMetrics: total = last_token - arrival; ttft = first_token - arrival;
# prefill = first_token - first_scheduled (vision + prefill; the vision tower is not separately
# exposed -> visual_ms null); decode = last_token - first_token (chunk-1 tokens).
TIMING_KEYS = ("arrival_time", "first_scheduled_time", "first_token_time", "last_token_time")
KV_KEYS = ("num_reused_blocks", "num_total_allocated_blocks", "num_new_allocated_blocks", "kv_cache_hit_rate")

_LIVE_LLMS = []


def _shutdown_all():
    for llm in _LIVE_LLMS:
        try:
            llm.shutdown()
        except Exception:
            pass


atexit.register(_shutdown_all)


def log(*parts):
    print(*parts, file=sys.stderr, flush=True)


def build_llm(model, reuse=True, kv_frac=0.12, max_tokens=8192):
    """kv_frac (not max_tokens) caps VRAM; images need ~2773 context tokens so the token budget stays high."""
    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig
    kv_kwargs = dict(enable_block_reuse=reuse, free_gpu_memory_fraction=kv_frac)
    # rc24: the v2 cache-key breaks on multimodal (id_offset None) -> v1 unless forced
    if os.environ.get("TRTLLM_KV_V2", "0") != "1":
        kv_kwargs["use_kv_cache_manager_v2"] = False
    llm = LLM(model=model, backend="pytorch", kv_cache_config=KvCacheConfig(**kv_kwargs),
              cuda_graph_config=CudaGraphConfig(enable_padding=True),
              max_batch_size=1, max_num_tokens=max_tokens, max_seq_len=max_tokens, trust_remote_code=True)
    _LIVE_LLMS.append(llm)
    return llm


def model_type(model):
    config = json.load(open(os.path.join(model, "config.json")))
    return config.get("model_type", "qwen3_5")


def close_think(prompt):
    if NO_THINK and prompt.endswith("<think>\n"):
        return prompt[:-len("<think>\n")] + "<think>\n\n</think>\n\n"
    return prompt


def make_inputs(llm, model, image, prompt, count, append_text=""):
    from tensorrt_llm.inputs import default_multimodal_input_loader
    tokenizer = llm.tokenizer if not isinstance(llm, str) else None
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model)
    if image:
        inputs = default_multimodal_input_loader(tokenizer=tokenizer, model_dir=model,
                                                 model_type=model_type(model), modality="image",
                                                 prompts=[prompt] * count, media=[[image]] * count,
                                                 image_data_format="pt", device="cuda")
        for entry in inputs:
            entry["prompt"] = close_think(entry["prompt"]) + append_text
        return inputs
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return [close_think(text) + append_text] * count


def _metric_ms(obj, *path):
    """Walk dict / object attribute paths; timedeltas and sub-1e6 floats (seconds) become ms."""
    for key in path:
        if obj is None:
            return None
        obj = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
    if obj is None:
        return None
    if hasattr(obj, "total_seconds"):
        return obj.total_seconds() * 1000.0
    return float(obj) * (1000.0 if isinstance(obj, float) and obj < 1e6 else 1.0)


def perf(output):
    """RequestPerfMetrics -> dict of ms (tolerant to dict / object forms)."""
    metrics = (getattr(output.outputs[0], "request_perf_metrics", None)
               or getattr(output, "request_perf_metrics", None))
    if metrics is None:
        return {}
    timing = {key: _metric_ms(metrics, "timing_metrics", key) for key in TIMING_KEYS}
    kv = {key: _metric_ms(metrics, "kv_cache_metrics", key) for key in KV_KEYS}
    result = {}
    if timing["arrival_time"] is not None and timing["last_token_time"] is not None:
        result["total_ms"] = timing["last_token_time"] - timing["arrival_time"]
        if timing["first_token_time"] is not None:
            result["ttft_ms"] = timing["first_token_time"] - timing["arrival_time"]
            result["decode_ms"] = timing["last_token_time"] - timing["first_token_time"]
            scheduled = timing["first_scheduled_time"]
            result["prefill_ms"] = timing["first_token_time"] - (scheduled if scheduled is not None
                                                                 else timing["arrival_time"])
    result["kv"] = {key: value for key, value in kv.items() if value is not None}
    result["_raw"] = {key: value for key, value in timing.items()}
    return result


def one_step(llm, inputs, chunk):
    from tensorrt_llm import SamplingParams
    # end_id=-1: never stop early -> exactly `chunk` tokens
    sampling = SamplingParams(max_tokens=chunk, temperature=0.0, top_k=1, return_perf_metrics=True, end_id=-1)
    t_start_ns = time.monotonic_ns()
    wall_start = time.perf_counter()
    output = llm.generate(inputs, sampling)[0]
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    metrics = perf(output)
    record = {"t_start_ns": t_start_ns, "wall_ms": wall_ms, "n_tokens": len(output.outputs[0].token_ids),
              "total_ms": metrics.get("total_ms", wall_ms), "visual_ms": None,
              "prefill_ms": metrics.get("prefill_ms"), "decode_ms": metrics.get("decode_ms"),
              "ttft_ms": metrics.get("ttft_ms"), "kv": metrics.get("kv", {}),
              "basis": "RequestPerfMetrics" if metrics.get("total_ms") else "wall"}
    record["reuse"] = (record["kv"].get("num_reused_blocks") or 0) > 0 if record["kv"] else None
    record["text"] = output.outputs[0].text
    return record


def step_line(step, record, stack):
    return {"step": step, "t_start_ns": record["t_start_ns"], "total_ms": record["total_ms"],
            "visual_ms": None, "prefill_ms": record["prefill_ms"], "decode_ms": record["decode_ms"],
            "n_tokens": record["n_tokens"], "reuse": record["reuse"], "stack": stack}


def summarize(records):
    def quantile(key, fraction):
        values = sorted(record[key] for record in records if record.get(key) is not None)
        if not values:
            return None
        return values[min(len(values) - 1, int(round(fraction * (len(values) - 1))))]

    reuse_frac = (sum(1 for record in records if record.get("reuse")) / len(records)) if records else None
    return {"n": len(records), "total_ms_p50": quantile("total_ms", .5),
            "total_ms_p99": quantile("total_ms", .99),
            "ttft_ms_p50": quantile("ttft_ms", .5), "prefill_ms_p50": quantile("prefill_ms", .5),
            "decode_ms_p50": quantile("decode_ms", .5), "wall_ms_p50": quantile("wall_ms", .5),
            "reuse_frac": reuse_frac}


def checkpoint_weight_gb(model):
    return sum(os.path.getsize(os.path.join(root, name))
               for root, _dirs, names in os.walk(model) for name in names
               if name.endswith((".safetensors", ".bin"))) / 1e9


def mode_probe(args):
    llm = build_llm(args.model, True, args.kv_frac)
    inputs = make_inputs(llm, args.model, args.image, args.prompt, 1)
    for i in range(3):
        record = one_step(llm, inputs, args.chunk)
        log(f"probe step {i}: total {record['total_ms']:.1f} ms ttft {record['ttft_ms']} "
            f"prefill {record['prefill_ms']} decode {record['decode_ms']} tokens {record['n_tokens']} "
            f"reuse {record['reuse']} kv {record['kv']}")
    probe = {"ok": True, "model": args.model, "engine_gb": round(checkpoint_weight_gb(args.model), 2),
             "last": {key: value for key, value in record.items() if key != "kv"}}
    json.dump(probe, open(os.path.join(args.out, "probe.json"), "w"), indent=1)
    print(json.dumps(probe, indent=1))


def mode_loop(args):
    """Co-location side load: back-to-back steps until killed."""
    llm = build_llm(args.model, True, args.kv_frac)
    inputs = make_inputs(llm, args.model, args.image, args.prompt, 1)
    for _ in range(args.warm):
        one_step(llm, inputs, args.chunk)
    steps_file = open(os.path.join(args.out, "vlm_e2e_steps.jsonl"), "a", buffering=1)
    stop = {"requested": False}

    def request_stop(*_signal_args):
        stop["requested"] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    print("READY", flush=True)
    step = 0
    while not stop["requested"]:
        record = one_step(llm, inputs, args.chunk)
        step += 1
        line = step_line(step, record, args.stack)
        line.update({"wall_ms": record["wall_ms"], "basis": record["basis"]})
        steps_file.write(json.dumps(line) + "\n")
        if step % 20 == 0:
            print(f"step {step}: {record['total_ms']:.1f} ms", flush=True)
    steps_file.close()
    print(f"DONE {step} steps", flush=True)


def compose_block(engine_gb, chunks, no_reuse):
    """Compose-schema block: decode per token from the chunk-8 pass; prefill_reuse = prefill with the
    repeated prefix; cold = the no-reuse pass."""
    chunk8 = chunks.get("8", {})
    chunk1 = chunks.get("1", {})
    decode_per_token = (chunk8.get("decode_ms_p50") or 0) / 7.0 if chunk8.get("decode_ms_p50") else None
    return {"engine_gb": round(engine_gb, 3), "visual_ms": None,
            "prefill_cold_ms": (no_reuse.get("8") or {}).get("prefill_ms_p50"),
            "prefill_reuse_ms": chunk8.get("prefill_ms_p50"),
            "decode_ms": decode_per_token, "ttft_ms": chunk1.get("total_ms_p50") or chunk8.get("ttft_ms_p50"),
            "steps": {key: {"step_ms": summary.get("total_ms_p50"),
                            "hz": (1000.0 / summary["total_ms_p50"]) if summary.get("total_ms_p50") else None}
                      for key, summary in chunks.items()}}


def mode_sweep(args):
    """Solo matrix: chunks x steps, reuse on and off."""
    engine_gb = checkpoint_weight_gb(args.model)
    result = {"state": args.state, "stack": args.stack, "tag": args.tag, "model": args.model,
              "engine_gb": round(engine_gb, 3), "chunks": {}, "no_reuse": {}}
    llm = build_llm(args.model, True, args.kv_frac)
    inputs = make_inputs(llm, args.model, args.image, args.prompt, 1)
    for _ in range(args.warm):
        one_step(llm, inputs, 8)
    for chunk in [int(value) for value in args.chunks.split(",")]:
        records = [one_step(llm, inputs, chunk) for _ in range(args.steps)]
        result["chunks"][str(chunk)] = summarize(records)
        if chunk == 8:
            with open(os.path.join(args.out, f"{args.tag}_steps.jsonl"), "w") as steps_file:
                for index, record in enumerate(records):
                    steps_file.write(json.dumps(step_line(index + 1, record, args.stack)) + "\n")
        log(f"sweep {args.tag} chunk {chunk}: {result['chunks'][str(chunk)]}")
    del llm
    if not args.no_reuse_pass:
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        llm_cold = build_llm(args.model, False, args.kv_frac)
        inputs_cold = make_inputs(llm_cold, args.model, args.image, args.prompt, 1)
        for _ in range(args.warm):
            one_step(llm_cold, inputs_cold, 8)
        records = [one_step(llm_cold, inputs_cold, 8) for _ in range(max(10, args.steps // 3))]
        result["no_reuse"]["8"] = summarize(records)
        del llm_cold
    result["compose"] = compose_block(engine_gb, result["chunks"], result["no_reuse"])
    json.dump(result, open(os.path.join(args.out, f"sweep_{args.tag}_{args.state}.json"), "w"), indent=1)
    print(json.dumps(result["compose"], indent=1))


def mode_accuracy(args):
    """The CANDIDATE engine's greedy answers; endorsement is done by teacher_forced_gate.py vs the bf16
    HF baseline."""
    from tensorrt_llm import SamplingParams
    battery = json.load(open(args.battery))
    llm = build_llm(args.model, True, args.kv_frac)
    answers = []
    for index, request in enumerate(battery["requests"]):
        content = request["messages"][0]["content"]
        image = next((part["image"] for part in content if part.get("type") == "image"), None)
        text = " ".join(part["text"] for part in content if part.get("type") == "text")
        if image and not os.path.isabs(image):
            image = os.path.join(os.environ.get("EDGELLM_ROOT", ""), image)
        inputs = make_inputs(llm, args.model, image, text, 1)
        sampling = SamplingParams(max_tokens=int(battery.get("max_generate_length", args.max_tokens)),
                                  temperature=0.0, top_k=1)
        output = llm.generate(inputs, sampling)[0]
        answers.append({"request_idx": index, "token_ids": list(output.outputs[0].token_ids),
                        "text": output.outputs[0].text})
    json.dump({"model": args.model, "stack": args.stack, "no_think": NO_THINK, "results": answers},
              open(args.out, "w"), indent=1)
    print(f"generated {len(answers)} candidate answers -> {args.out}")


MODES = {"probe": mode_probe, "sweep": mode_sweep, "loop": mode_loop, "accuracy": mode_accuracy}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["probe", "sweep", "loop", "accuracy"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", default="")
    parser.add_argument("--prompt", default="Describe this image in one sentence.")
    parser.add_argument("--out", default=".")
    parser.add_argument("--tag", default="model")
    parser.add_argument("--state", default="unlocked")
    parser.add_argument("--chunks", default="1,2,4,8")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--warm", type=int, default=3)
    parser.add_argument("--kv-frac", type=float, default=0.12)
    parser.add_argument("--no-reuse-pass", action="store_true")
    parser.add_argument("--battery", default="")
    parser.add_argument("--ref", default="")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--stack", default="trtllm")
    args = parser.parse_args()
    if args.mode in ("probe", "sweep", "loop"):
        os.makedirs(args.out, exist_ok=True)
    MODES[args.mode](args)


if __name__ == "__main__":
    main()
