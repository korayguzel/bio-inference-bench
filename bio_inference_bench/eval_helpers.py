"""Shared evaluation helpers for weight quantization experiments.

Provides the core measurement functions used by both Phase 1 and Phase 2
evaluation scripts, preventing duplication and drift.
"""

from __future__ import annotations

import gc
import time

import torch

from bio_inference_bench.kv_int8_chunked import ChunkedInt8KVCache, run_chunked_decode_step
from bio_inference_bench.models import load_model_and_tokenizer, prepare_prompt
from bio_inference_bench.profiler import reset_memory_tracking, take_snapshot


def full_cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()


def get_device_placement(model) -> dict:
    """Check whether model is fully GPU-resident or partially offloaded."""
    if hasattr(model, "hf_device_map"):
        devices = set(str(d) for d in model.hf_device_map.values())
        fully_gpu = all("cuda" in d or d == "0" for d in devices)
        return {"hf_device_map": dict(model.hf_device_map),
                "unique_devices": sorted(devices),
                "fully_gpu_resident": fully_gpu}
    else:
        param_devices = set(str(p.device) for p in model.parameters())
        fully_gpu = all("cuda" in d for d in param_devices)
        return {"param_devices": sorted(param_devices),
                "fully_gpu_resident": fully_gpu}


def run_config_with_sanity(device, weight_quant, kv_mode, prompt_len, max_new, sanity_steps):
    """Run one config with fresh model load, capturing tokens + logits for sanity.

    Returns a result dict with all memory metrics, timing, generated tokens,
    logits for sanity comparison, device placement info, and weight metadata.
    """
    full_cleanup()
    model, tokenizer, metadata = load_model_and_tokenizer(
        "protgpt2", device=str(device), dtype=torch.float16,
        weight_quantization=weight_quant,
    )
    input_ids = prepare_prompt(tokenizer, prompt_token_length=prompt_len).to(device)

    # Warmup
    warmup_ids = prepare_prompt(tokenizer, prompt_token_length=16).to(device)
    with torch.no_grad():
        out = model(warmup_ids, use_cache=True)
        _ = model(out.logits[:, -1, :].argmax(-1, keepdim=True),
                  past_key_values=out.past_key_values, use_cache=True)
    del out, warmup_ids
    full_cleanup()

    weight_mem = torch.cuda.memory_allocated() / (1024**2)
    device_info = get_device_placement(model)

    # --- Generation with logit capture ---
    use_int8 = kv_mode == "int8-triton"

    reset_memory_tracking()
    mem_before = take_snapshot()

    # Prefill
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000

    if use_int8:
        cache = ChunkedInt8KVCache(chunk_size=64)
        std_cache = outputs.past_key_values
        for li in range(len(std_cache.layers)):
            layer = std_cache.layers[li]
            cache.update(layer.keys, layer.values, li)
        next_token = outputs.logits[:, -1, :].argmax(-1, keepdim=True)
        tokens = [next_token]
        logits_list = [outputs.logits[:, -1, :].detach().cpu()]
        del outputs, std_cache
        gc.collect()
        torch.cuda.empty_cache()
    else:
        past = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(-1, keepdim=True)
        tokens = [next_token]
        logits_list = [outputs.logits[:, -1, :].detach().cpu()]
        del outputs

    prefill_snap = take_snapshot()
    prefill_peak = prefill_snap.max_allocated_mb
    after_prefill = prefill_snap.allocated_mb
    torch.cuda.reset_peak_memory_stats()

    # Decode
    step_times = []
    with torch.no_grad():
        for _ in range(max_new - 1):
            torch.cuda.synchronize()
            t = time.perf_counter()
            if use_int8:
                logits = run_chunked_decode_step(model, next_token, cache, use_triton=True)
            else:
                out = model(next_token, past_key_values=past, use_cache=True)
                logits = out.logits
                past = out.past_key_values
                del out
            torch.cuda.synchronize()
            step_times.append((time.perf_counter() - t) * 1000)
            next_token = logits[:, -1, :].argmax(-1, keepdim=True)
            tokens.append(next_token)
            if len(logits_list) < sanity_steps:
                logits_list.append(logits[:, -1, :].detach().cpu())
            del logits

    decode_snap = take_snapshot()
    decode_peak = decode_snap.max_allocated_mb

    decode_ms = sum(step_times)
    actual = len(tokens)
    total_ms = prefill_ms + decode_ms
    decode_tps = (actual - 1) / (decode_ms / 1000) if decode_ms > 0 else 0
    e2e_tps = actual / (total_ms / 1000) if total_ms > 0 else 0
    overall_peak = max(prefill_peak, decode_peak)
    e2e_overhead = overall_peak - mem_before.allocated_mb
    decode_growth = decode_peak - after_prefill

    cache_info = {}
    if use_int8:
        cache_info = cache.compression_summary()

    token_ids = torch.cat(tokens, dim=1).cpu()[0].tolist()

    if not use_int8:
        del past
    else:
        del cache
    del tokens, next_token

    result = {
        "weight_quantization": weight_quant,
        "kv_mode": kv_mode,
        "weight_memory_mb": round(weight_mem, 2),
        "metadata_weight_mb": round(metadata.weight_memory_mb, 2),
        "device_placement": device_info,
        "prompt_token_length": input_ids.shape[1],
        "max_new_tokens": max_new,
        "actual_new_tokens": actual,
        "prefill_ms": round(prefill_ms, 2),
        "decode_ms": round(decode_ms, 2),
        "total_ms": round(total_ms, 2),
        "decode_tokens_per_sec": round(decode_tps, 2),
        "end_to_end_tokens_per_sec": round(e2e_tps, 2),
        "memory_before_generation_mb": round(mem_before.allocated_mb, 2),
        "memory_after_prefill_mb": round(after_prefill, 2),
        "prefill_peak_allocated_mb": round(prefill_peak, 2),
        "decode_peak_allocated_mb": round(decode_peak, 2),
        "overall_peak_allocated_mb": round(overall_peak, 2),
        "end_to_end_generation_overhead_mb": round(e2e_overhead, 2),
        "decode_phase_growth_mb": round(decode_growth, 2),
        "cache_info": cache_info,
        "generated_token_ids": token_ids,
        "logits_for_sanity": logits_list,
    }

    del model, tokenizer
    full_cleanup()
    return result


def sanity(base: dict, proto: dict, max_steps: int) -> dict:
    """Compare token agreement and logit similarity between two runs."""
    bt, pt = base["generated_token_ids"], proto["generated_token_ids"]
    n = min(len(bt), len(pt), max_steps)
    agree = sum(1 for a, b in zip(bt[:n], pt[:n]) if a == b)
    first_diff = next((i for i in range(n) if bt[i] != pt[i]), n)
    bl, pl = base["logits_for_sanity"], proto["logits_for_sanity"]
    ln = min(len(bl), len(pl), max_steps)
    top1 = sum(1 for i in range(ln) if bl[i].float().argmax() == pl[i].float().argmax())
    cosines = [torch.nn.functional.cosine_similarity(
        bl[i].float(), pl[i].float(), dim=-1).item() for i in range(ln)]
    return {
        "token_agreement_pct": round(agree / n * 100, 1) if n else 0,
        "first_divergence_step": first_diff,
        "steps_compared": n,
        "top1_logit_agreement_pct": round(top1 / ln * 100, 1) if ln else 0,
        "avg_logit_cosine_similarity": round(sum(cosines) / len(cosines), 6) if cosines else 0,
    }


def strip(d: dict) -> dict:
    """Remove large non-serializable fields from result dict for JSON output."""
    return {k: v for k, v in d.items() if k not in ("generated_token_ids", "logits_for_sanity")}
