#!/usr/bin/env python3
"""TensorRT-LLM VLM step tool for a discrete card.

One "VLM step" = one request: image + prompt -> vision tower -> prefill (KV/prefix reuse when the prefix repeats) -> `chunk`
decoded tokens (greedy). Timings come from the runtime's RequestPerfMetrics (C++ executor clocks), never Python timers:
    total_ms   = last_token - arrival          ttft_ms   = first_token - arrival
    prefill_ms = first_token - first_scheduled (vision + prefill; the vision tower is not separately exposed -> visual_ms null)
    decode_ms  = last_token - first_token      (chunk-1 tokens)   ms/token = decode_ms / (n_tokens-1)
Modes:
  probe    load the model, one image step, print tokens/text + timings (feasibility gate)
  sweep    chunks x steps, reuse on (repeated prefix) and off  -> sweep JSON in the compose schema + <V>_steps.jsonl (chunk 8)
  loop     back-to-back steps until SIGTERM (co-location side load); prints READY after warm-up; writes vlm_e2e_steps.jsonl line-buffered
  accuracy greedy answers for a battery JSON (Edge-LLM gate_battery format) -> answers JSON (+ agreement vs a reference answers file)
Run inside ~/venv_trtllm with the OpenMPI env (see trtllm_env.sh). Usage examples:
  trtllm_vlm_step.py probe    --model <ckpt> --image <jpg>
  trtllm_vlm_step.py sweep    --model <ckpt> --image <jpg> --out <dir> --tag 27b_fp8 --state unlocked [--chunks 1,2,4,8] [--steps 30]
  trtllm_vlm_step.py loop     --model <ckpt> --image <jpg> --out <arm dir> [--chunk 8] [--warm 3]
  trtllm_vlm_step.py accuracy --model <ckpt> --battery <json> --out <answers.json> [--ref <answers.json>]
"""
import argparse, atexit, json, os, signal, statistics as st, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # avoid fragmentation blow-up under the VRAM policy cap
_LLMS = []
def _shutdown_all():
    for l in _LLMS:
        try: l.shutdown()
        except Exception: pass
atexit.register(_shutdown_all)

def log(*a): print(*a, file=sys.stderr, flush=True)

def build_llm(model, reuse=True, kv_frac=0.12, max_tokens=8192):   # kv_frac (not max_tokens) caps VRAM; images need ~2773 ctx tokens so keep the budget high
    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig
    kv_kw = dict(enable_block_reuse=reuse, free_gpu_memory_fraction=kv_frac)
    if os.environ.get("TRTLLM_KV_V2", "0") != "1": kv_kw["use_kv_cache_manager_v2"] = False   # rc24: v2 cache-key breaks on multimodal (id_offset None) -> v1 unless forced
    kv = KvCacheConfig(**kv_kw)
    llm = LLM(model=model, backend="pytorch", kv_cache_config=kv, cuda_graph_config=CudaGraphConfig(enable_padding=True),
              max_batch_size=1, max_num_tokens=max_tokens, max_seq_len=max_tokens, trust_remote_code=True)
    _LLMS.append(llm); return llm

def model_type(model):
    c = json.load(open(os.path.join(model, "config.json"))); return c.get("model_type", "qwen3_5")

NO_THINK = os.environ.get("TRTLLM_THINK", "0") != "1"   # the chat template opens a <think> block; close it (enable_thinking=False, as the Jetson Edge-LLM method) so steps/accuracy measure ANSWER tokens
def close_think(prompt):
    return prompt[:-len("<think>\n")] + "<think>\n\n</think>\n\n" if NO_THINK and prompt.endswith("<think>\n") else prompt

def make_inputs(llm, model, image, prompt, n, append_text=""):
    from tensorrt_llm.inputs import default_multimodal_input_loader
    tok = llm.tokenizer if not isinstance(llm, str) else None
    if tok is None:
        from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained(model)
    if image:
        inp = default_multimodal_input_loader(tokenizer=tok, model_dir=model, model_type=model_type(model), modality="image", prompts=[prompt] * n, media=[[image]] * n, image_data_format="pt", device="cuda")
        for x in inp: x["prompt"] = close_think(x["prompt"]) + append_text
        return inp
    msgs = [{"role": "user", "content": prompt}]
    txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return [close_think(txt) + append_text] * n

def perf(o):
    """RequestPerfMetrics -> dict of ms (tolerant to dict / object forms)."""
    m = getattr(o.outputs[0], "request_perf_metrics", None) or getattr(o, "request_perf_metrics", None)
    if m is None: return {}
    def g(obj, *path):
        for p in path:
            if obj is None: return None
            obj = obj.get(p) if isinstance(obj, dict) else getattr(obj, p, None)
        if obj is None: return None
        return obj.total_seconds() * 1000.0 if hasattr(obj, "total_seconds") else float(obj) * (1000.0 if isinstance(obj, float) and obj < 1e6 else 1.0)
    t = {k: g(m, "timing_metrics", k) for k in ("arrival_time", "first_scheduled_time", "first_token_time", "last_token_time")}
    kv = {k: g(m, "kv_cache_metrics", k) for k in ("num_reused_blocks", "num_total_allocated_blocks", "num_new_allocated_blocks", "kv_cache_hit_rate")}
    r = {}
    if t["arrival_time"] is not None and t["last_token_time"] is not None:
        r["total_ms"] = t["last_token_time"] - t["arrival_time"]
        if t["first_token_time"] is not None:
            r["ttft_ms"] = t["first_token_time"] - t["arrival_time"]; r["decode_ms"] = t["last_token_time"] - t["first_token_time"]
            r["prefill_ms"] = t["first_token_time"] - (t["first_scheduled_time"] if t["first_scheduled_time"] is not None else t["arrival_time"])
    r["kv"] = {k: v for k, v in kv.items() if v is not None}; r["_raw"] = {k: v for k, v in t.items()}
    return r

def one_step(llm, inputs, chunk, wall=True):
    from tensorrt_llm import SamplingParams
    sp = SamplingParams(max_tokens=chunk, temperature=0.0, top_k=1, return_perf_metrics=True, end_id=-1)   # end_id=-1: never stop early -> exactly `chunk` tokens
    t0 = time.monotonic_ns(); w0 = time.perf_counter()
    o = llm.generate(inputs, sp)[0]; w1 = time.perf_counter()
    p = perf(o); n = len(o.outputs[0].token_ids)
    rec = {"t_start_ns": t0, "wall_ms": (w1 - w0) * 1000.0, "n_tokens": n, "total_ms": p.get("total_ms", (w1 - w0) * 1000.0), "visual_ms": None,
           "prefill_ms": p.get("prefill_ms"), "decode_ms": p.get("decode_ms"), "ttft_ms": p.get("ttft_ms"), "kv": p.get("kv", {}), "basis": "RequestPerfMetrics" if p.get("total_ms") else "wall"}
    rec["reuse"] = (rec["kv"].get("num_reused_blocks") or 0) > 0 if rec["kv"] else None
    rec["text"] = o.outputs[0].text
    return rec

def summarize(recs):
    def q(k, p):
        v = [r[k] for r in recs if r.get(k) is not None]
        if not v: return None
        v = sorted(v); return v[min(len(v) - 1, int(round(p * (len(v) - 1))))]
    return {"n": len(recs), "total_ms_p50": q("total_ms", .5), "total_ms_p99": q("total_ms", .99), "ttft_ms_p50": q("ttft_ms", .5), "prefill_ms_p50": q("prefill_ms", .5),
            "decode_ms_p50": q("decode_ms", .5), "wall_ms_p50": q("wall_ms", .5), "reuse_frac": (sum(1 for r in recs if r.get("reuse")) / len(recs)) if recs else None}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["probe", "sweep", "loop", "accuracy"])
    ap.add_argument("--model", required=True); ap.add_argument("--image", default=""); ap.add_argument("--prompt", default="Describe this image in one sentence.")
    ap.add_argument("--out", default="."); ap.add_argument("--tag", default="model"); ap.add_argument("--state", default="unlocked")
    ap.add_argument("--chunks", default="1,2,4,8"); ap.add_argument("--steps", type=int, default=30); ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--warm", type=int, default=3); ap.add_argument("--kv-frac", type=float, default=0.12); ap.add_argument("--no-reuse-pass", action="store_true")
    ap.add_argument("--battery", default=""); ap.add_argument("--ref", default=""); ap.add_argument("--max-tokens", type=int, default=64); ap.add_argument("--stack", default="trtllm")
    a = ap.parse_args()
    if a.mode in ("probe", "sweep", "loop"):
        os.makedirs(a.out, exist_ok=True)
        engine_gb = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(a.model) for f in fs if f.endswith((".safetensors", ".bin"))) / 1e9

    if a.mode == "probe":
        llm = build_llm(a.model, True, a.kv_frac); inp = make_inputs(llm, a.model, a.image, a.prompt, 1)
        for i in range(3):
            r = one_step(llm, inp, a.chunk); log(f"probe step {i}: total {r['total_ms']:.1f} ms ttft {r['ttft_ms']} prefill {r['prefill_ms']} decode {r['decode_ms']} tokens {r['n_tokens']} reuse {r['reuse']} kv {r['kv']}")
        pj = {"ok": True, "model": a.model, "engine_gb": round(engine_gb, 2), "last": {k: v for k, v in r.items() if k != "kv"}}
        json.dump(pj, open(os.path.join(a.out, "probe.json"), "w"), indent=1); print(json.dumps(pj, indent=1)); return

    if a.mode == "loop":                                           # co-location side load: back-to-back steps until killed
        llm = build_llm(a.model, True, a.kv_frac); inp = make_inputs(llm, a.model, a.image, a.prompt, 1)
        for _ in range(a.warm): one_step(llm, inp, a.chunk)
        f = open(os.path.join(a.out, "vlm_e2e_steps.jsonl"), "a", buffering=1); stop = {"v": False}
        signal.signal(signal.SIGTERM, lambda *x: stop.__setitem__("v", True)); signal.signal(signal.SIGINT, lambda *x: stop.__setitem__("v", True))
        print("READY", flush=True); k = 0
        while not stop["v"]:
            r = one_step(llm, inp, a.chunk); k += 1
            f.write(json.dumps({"step": k, "t_start_ns": r["t_start_ns"], "total_ms": r["total_ms"], "visual_ms": None, "prefill_ms": r["prefill_ms"], "decode_ms": r["decode_ms"],
                                "n_tokens": r["n_tokens"], "reuse": r["reuse"], "stack": a.stack, "wall_ms": r["wall_ms"], "basis": r["basis"]}) + "\n")
            if k % 20 == 0: print(f"step {k}: {r['total_ms']:.1f} ms", flush=True)
        f.close(); print(f"DONE {k} steps", flush=True); return

    if a.mode == "sweep":                                          # solo matrix: chunks x steps, reuse on and off
        res = {"state": a.state, "stack": a.stack, "tag": a.tag, "model": a.model, "engine_gb": round(engine_gb, 3), "chunks": {}, "no_reuse": {}}
        llm = build_llm(a.model, True, a.kv_frac); inp = make_inputs(llm, a.model, a.image, a.prompt, 1)
        for _ in range(a.warm): one_step(llm, inp, 8)
        for c in [int(x) for x in a.chunks.split(",")]:
            recs = [one_step(llm, inp, c) for _ in range(a.steps)]; res["chunks"][str(c)] = summarize(recs)
            if c == 8:
                with open(os.path.join(a.out, f"{a.tag}_steps.jsonl"), "w") as f:
                    for k, r in enumerate(recs): f.write(json.dumps({"step": k + 1, "t_start_ns": r["t_start_ns"], "total_ms": r["total_ms"], "visual_ms": None, "prefill_ms": r["prefill_ms"], "decode_ms": r["decode_ms"], "n_tokens": r["n_tokens"], "reuse": r["reuse"], "stack": a.stack}) + "\n")
            log(f"sweep {a.tag} chunk {c}: {res['chunks'][str(c)]}")
        del llm
        if not a.no_reuse_pass:
            import gc, torch; gc.collect(); torch.cuda.empty_cache()
            llm2 = build_llm(a.model, False, a.kv_frac); inp2 = make_inputs(llm2, a.model, a.image, a.prompt, 1)
            for _ in range(a.warm): one_step(llm2, inp2, 8)
            recs = [one_step(llm2, inp2, 8) for _ in range(max(10, a.steps // 3))]; res["no_reuse"]["8"] = summarize(recs); del llm2
        # compose-schema block (decode per token from the chunk-8 pass; prefill_reuse = prefill with the repeated prefix; cold = no-reuse pass)
        c8 = res["chunks"].get("8", {}); c1 = res["chunks"].get("1", {})
        dec_tok = (c8.get("decode_ms_p50") or 0) / 7.0 if c8.get("decode_ms_p50") else None
        res["compose"] = {"engine_gb": round(engine_gb, 3), "visual_ms": None, "prefill_cold_ms": (res["no_reuse"].get("8") or {}).get("prefill_ms_p50"), "prefill_reuse_ms": c8.get("prefill_ms_p50"),
                          "decode_ms": dec_tok, "ttft_ms": c1.get("total_ms_p50") or c8.get("ttft_ms_p50"),
                          "steps": {k: {"step_ms": v.get("total_ms_p50"), "hz": (1000.0 / v["total_ms_p50"]) if v.get("total_ms_p50") else None} for k, v in res["chunks"].items()}}
        json.dump(res, open(os.path.join(a.out, f"sweep_{a.tag}_{a.state}.json"), "w"), indent=1); print(json.dumps(res["compose"], indent=1)); return

    if a.mode == "accuracy":                                       # generate the CANDIDATE engine's greedy answers (endorsement is done by teacher_forced_gate.py vs the bf16 HF baseline)
        from tensorrt_llm import SamplingParams
        b = json.load(open(a.battery)); reqs = b["requests"]; llm = build_llm(a.model, True, a.kv_frac); res = []
        for idx, r in enumerate(reqs):
            msg = r["messages"][0]["content"]; img = next((c["image"] for c in msg if c.get("type") == "image"), None); txt = " ".join(c["text"] for c in msg if c.get("type") == "text")
            if img and not os.path.isabs(img): img = os.path.join(os.environ.get("EDGELLM_ROOT", ""), img)
            inp = make_inputs(llm, a.model, img, txt, 1)
            o = llm.generate(inp, SamplingParams(max_tokens=int(b.get("max_generate_length", a.max_tokens)), temperature=0.0, top_k=1))[0]
            res.append({"request_idx": idx, "token_ids": list(o.outputs[0].token_ids), "text": o.outputs[0].text})
        json.dump({"model": a.model, "stack": a.stack, "no_think": NO_THINK, "results": res}, open(a.out, "w"), indent=1)
        print(f"generated {len(res)} candidate answers -> {a.out}"); return

if __name__ == "__main__": main()
