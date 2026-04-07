"""INT8 KV generation orchestrator for ProtGPT2.

Provides two functions:
- generate_int8(): Complete INT8 KV generation pipeline (prefill → transfer → decode)
- generate(): Unified dispatcher that routes between FP16 baseline and INT8-Triton

This module does NOT modify generation.py (baseline stays clean) or the core INT8
modules (kv_int8_chunked.py, triton_int8_attention.py).
"""

from __future__ import annotations

import gc
import time

import torch

from bio_inference_bench.generation import GenerationResult, run_manual_prefill_decode
from bio_inference_bench.kv_int8_chunked import ChunkedInt8KVCache, run_chunked_decode_step
from bio_inference_bench.profiler import reset_memory_tracking, take_snapshot


def _validate_protgpt2(model, model_name: str) -> None:
    """Validate that the model is ProtGPT2 by name and architecture.

    Checks both model name and architecture parameters to prevent:
    - Using a wrong model name that happens to match ProtGPT2's architecture
    - Using a modified ProtGPT2 with different layer/head counts
    """
    config = model.config

    # Name check
    if model_name != "protgpt2":
        raise ValueError(
            f"INT8-Triton KV mode is only supported for ProtGPT2, got model_name='{model_name}'.\n"
            f"The fused Triton kernel is validated for ProtGPT2 (20 heads, 36 layers, head_dim=64) only."
        )

    # Architecture check
    n_layer = getattr(config, "n_layer", getattr(config, "num_hidden_layers", None))
    n_head = getattr(config, "n_head", getattr(config, "num_attention_heads", None))
    n_embd = getattr(config, "n_embd", getattr(config, "hidden_size", None))

    if n_layer is None or n_head is None or n_embd is None:
        raise ValueError(
            f"Cannot verify ProtGPT2 architecture: n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}"
        )

    head_dim = n_embd // n_head
    if n_layer != 36 or n_head != 20 or head_dim != 64:
        raise ValueError(
            f"Model architecture does not match ProtGPT2: "
            f"n_layer={n_layer} (expected 36), n_head={n_head} (expected 20), "
            f"head_dim={head_dim} (expected 64).\n"
            f"The fused Triton kernel is validated for ProtGPT2 only."
        )


def generate_int8(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    chunk_size: int = 64,
    use_triton: bool = True,
) -> dict:
    """Complete INT8 KV generation: prefill -> transfer -> INT8 decode loop.

    Args:
        model: Loaded ProtGPT2 model (FP16)
        tokenizer: Corresponding tokenizer
        input_ids: (1, seq_len) tensor — already tokenized prompt
        max_new_tokens: Number of new tokens to generate
        chunk_size: INT8 KV block size (default 64)
        use_triton: Use fused Triton kernel (default True)

    Returns:
        dict with generated_tokens, timing, memory (dual metrics), cache_info
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    prompt_len = input_ids.shape[1]

    cache = ChunkedInt8KVCache(chunk_size=chunk_size)

    # --- Memory before generation ---
    reset_memory_tracking()
    mem_before_gen = take_snapshot()

    # --- Prefill (standard FP16) ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000

    # Transfer prefill KV to INT8 cache
    std_cache = outputs.past_key_values
    for li in range(len(std_cache.layers)):
        layer = std_cache.layers[li]
        cache.update(layer.keys, layer.values, li)
    next_token = outputs.logits[:, -1, :].argmax(-1, keepdim=True)
    tokens = [next_token]
    del outputs, std_cache

    # Free FP16 prefill intermediates
    gc.collect()
    torch.cuda.empty_cache()

    # Capture prefill peak and post-prefill state
    prefill_snap = take_snapshot()
    prefill_peak_mb = prefill_snap.max_allocated_mb
    mem_after_prefill_mb = prefill_snap.allocated_mb

    # Reset peak tracking for decode-only measurement
    torch.cuda.reset_peak_memory_stats()

    # --- Decode ---
    step_times = []
    torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            t = time.perf_counter()
            logits = run_chunked_decode_step(model, next_token, cache,
                                             use_triton=use_triton)
            torch.cuda.synchronize()
            step_times.append((time.perf_counter() - t) * 1000)
            next_token = logits[:, -1, :].argmax(-1, keepdim=True)
            tokens.append(next_token)
            del logits

    decode_snap = take_snapshot()
    decode_peak_mb = decode_snap.max_allocated_mb

    # --- Compute metrics ---
    decode_ms = sum(step_times)
    actual_new = len(tokens)
    total_ms = prefill_ms + decode_ms
    decode_tps = (actual_new - 1) / (decode_ms / 1000) if decode_ms > 0 else 0
    e2e_tps = actual_new / (total_ms / 1000) if total_ms > 0 else 0

    overall_peak_mb = max(prefill_peak_mb, decode_peak_mb)
    e2e_overhead = overall_peak_mb - mem_before_gen.allocated_mb
    decode_growth = decode_peak_mb - mem_after_prefill_mb
    decode_growth_per_token = decode_growth / (actual_new - 1) if actual_new > 1 else 0

    cache_info = cache.compression_summary()
    token_ids = torch.cat(tokens, dim=1).cpu()[0].tolist()

    del tokens, next_token, cache
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "method": "int8-triton" if use_triton else "int8-python",
        "prompt_token_length": prompt_len,
        "max_new_tokens": max_new_tokens,
        "actual_new_tokens": actual_new,
        "prefill_ms": round(prefill_ms, 2),
        "decode_ms": round(decode_ms, 2),
        "total_ms": round(total_ms, 2),
        "decode_tokens_per_sec": round(decode_tps, 2),
        "end_to_end_tokens_per_sec": round(e2e_tps, 2),
        "memory_before_generation_mb": round(mem_before_gen.allocated_mb, 2),
        "memory_after_prefill_mb": round(mem_after_prefill_mb, 2),
        "prefill_peak_allocated_mb": round(prefill_peak_mb, 2),
        "decode_peak_allocated_mb": round(decode_peak_mb, 2),
        "overall_peak_allocated_mb": round(overall_peak_mb, 2),
        "end_to_end_generation_overhead_mb": round(e2e_overhead, 2),
        "decode_phase_growth_mb": round(decode_growth, 2),
        "decode_growth_per_token_mb": round(decode_growth_per_token, 4),
        "cache_info": cache_info,
        "generated_token_ids": token_ids,
    }


def _run_fp16_baseline(model, input_ids, max_new_tokens, model_name) -> dict:
    """Run FP16 baseline and return result in the same dict format as generate_int8."""
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    prompt_len = input_ids.shape[1]

    reset_memory_tracking()
    mem_before_gen = take_snapshot()

    # Prefill
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000

    past = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(-1, keepdim=True)
    tokens = [next_token]
    del outputs

    # Capture prefill peak + settled state
    prefill_snap = take_snapshot()
    prefill_peak_mb = prefill_snap.max_allocated_mb
    mem_after_prefill_mb = prefill_snap.allocated_mb

    # Reset peak for decode
    torch.cuda.reset_peak_memory_stats()

    # Decode
    step_times = []
    torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            t = time.perf_counter()
            out = model(next_token, past_key_values=past, use_cache=True)
            torch.cuda.synchronize()
            step_times.append((time.perf_counter() - t) * 1000)
            next_token = out.logits[:, -1, :].argmax(-1, keepdim=True)
            tokens.append(next_token)
            past = out.past_key_values
            del out

    decode_snap = take_snapshot()
    decode_peak_mb = decode_snap.max_allocated_mb

    decode_ms = sum(step_times)
    actual_new = len(tokens)
    total_ms = prefill_ms + decode_ms
    decode_tps = (actual_new - 1) / (decode_ms / 1000) if decode_ms > 0 else 0
    e2e_tps = actual_new / (total_ms / 1000) if total_ms > 0 else 0

    overall_peak_mb = max(prefill_peak_mb, decode_peak_mb)
    e2e_overhead = overall_peak_mb - mem_before_gen.allocated_mb
    decode_growth = decode_peak_mb - mem_after_prefill_mb
    decode_growth_per_token = decode_growth / (actual_new - 1) if actual_new > 1 else 0

    token_ids = torch.cat(tokens, dim=1).cpu()[0].tolist()

    del tokens, next_token, past
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "method": "fp16",
        "prompt_token_length": prompt_len,
        "max_new_tokens": max_new_tokens,
        "actual_new_tokens": actual_new,
        "prefill_ms": round(prefill_ms, 2),
        "decode_ms": round(decode_ms, 2),
        "total_ms": round(total_ms, 2),
        "decode_tokens_per_sec": round(decode_tps, 2),
        "end_to_end_tokens_per_sec": round(e2e_tps, 2),
        "memory_before_generation_mb": round(mem_before_gen.allocated_mb, 2),
        "memory_after_prefill_mb": round(mem_after_prefill_mb, 2),
        "prefill_peak_allocated_mb": round(prefill_peak_mb, 2),
        "decode_peak_allocated_mb": round(decode_peak_mb, 2),
        "overall_peak_allocated_mb": round(overall_peak_mb, 2),
        "end_to_end_generation_overhead_mb": round(e2e_overhead, 2),
        "decode_phase_growth_mb": round(decode_growth, 2),
        "decode_growth_per_token_mb": round(decode_growth_per_token, 4),
        "cache_info": {},
        "generated_token_ids": token_ids,
    }


def generate(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    kv_mode: str = "fp16",
    model_name: str = "protgpt2",
    **kwargs,
) -> dict:
    """Unified generation dispatcher: FP16 baseline or INT8-Triton.

    Args:
        model: Loaded model
        tokenizer: Corresponding tokenizer
        input_ids: (1, seq_len) tensor — already tokenized
        max_new_tokens: Decode length
        kv_mode: "fp16" or "int8-triton"
        model_name: Name used to load the model (for validation)
        **kwargs: Passed to generate_int8 (chunk_size, use_triton)

    Returns:
        dict with generation results and memory metrics
    """
    if kv_mode == "fp16":
        return _run_fp16_baseline(model, input_ids, max_new_tokens, model_name)
    elif kv_mode == "int8-triton":
        _validate_protgpt2(model, model_name)
        return generate_int8(model, tokenizer, input_ids, max_new_tokens, **kwargs)
    else:
        raise ValueError(f"Unknown kv_mode: '{kv_mode}'. Use 'fp16' or 'int8-triton'.")
