#!/usr/bin/env python3
"""Phase C: BLOCK_KV tuning sweep for the Triton INT8 attention kernel.

Tests different block sizes on representative workloads and reports decode tok/s.
Uses a single model load (not fresh per run) since we're measuring relative speed
differences, not absolute memory.
"""

from __future__ import annotations

import gc
import math
import time

import torch

from bio_inference_bench.kv_int8_cache import quantize_to_int8
from bio_inference_bench.kv_int8_chunked import ChunkedInt8KVCache, run_chunked_decode_step
from bio_inference_bench.models import load_model_and_tokenizer, prepare_prompt
from bio_inference_bench.triton_int8_attention import triton_int8_attention
from bio_inference_bench.utils import get_device

BLOCK_SIZES = [32, 64, 128, 256]
# Sequence lengths matching our eval configs
SEQ_LENS = [288, 768]  # decode_heavy final, long_decode final
HEADS = 20
HEAD_DIM = 64
WARMUP_ITERS = 10
BENCH_ITERS = 50


def bench_kernel_isolated(seq_len, block_kv, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Benchmark the Triton kernel in isolation (no model, no PyTorch overhead)."""
    device = "cuda"
    scaling = 1.0 / math.sqrt(HEAD_DIM)

    Q = torch.randn(1, HEADS, 1, HEAD_DIM, device=device, dtype=torch.float16)
    K_fp = torch.randn(1, HEADS, seq_len, HEAD_DIM, device=device, dtype=torch.float16)
    V_fp = torch.randn(1, HEADS, seq_len, HEAD_DIM, device=device, dtype=torch.float16)
    K_int8, K_scales = quantize_to_int8(K_fp)
    V_int8, V_scales = quantize_to_int8(V_fp)
    del K_fp, V_fp

    # Warmup
    for _ in range(warmup):
        triton_int8_attention(Q, K_int8, K_scales, V_int8, V_scales, scaling, block_kv=block_kv)
    torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        triton_int8_attention(Q, K_int8, K_scales, V_int8, V_scales, scaling, block_kv=block_kv)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    avg_us = elapsed_ms / iters * 1000  # microseconds per call
    return avg_us


def bench_end_to_end(model, tokenizer, prompt_len, max_new, block_kv, device):
    """Benchmark end-to-end decode with the Triton kernel at a given block size."""
    ids = prepare_prompt(tokenizer, prompt_token_length=prompt_len).to(device)
    cache = ChunkedInt8KVCache(chunk_size=block_kv)

    # Prefill
    with torch.no_grad():
        outputs = model(ids, use_cache=True)
    std_cache = outputs.past_key_values
    for li in range(len(std_cache.layers)):
        layer = std_cache.layers[li]
        cache.update(layer.keys, layer.values, li)
    next_token = outputs.logits[:, -1, :].argmax(-1, keepdim=True)
    del outputs, std_cache
    gc.collect()
    torch.cuda.empty_cache()

    # Monkey-patch the block_kv into the triton call
    # The run_chunked_decode_step uses cache.chunked_attention for non-triton,
    # and triton_int8_attention for triton path. We need to set the block_kv.
    # For now, we'll directly time the decode loop with triton_int8_attention.
    import bio_inference_bench.triton_int8_attention as tmod
    orig_fn = tmod.triton_int8_attention

    def patched_fn(query, k_int8, k_scales, v_int8, v_scales, scaling, **kwargs):
        return orig_fn(query, k_int8, k_scales, v_int8, v_scales, scaling, block_kv=block_kv)

    tmod.triton_int8_attention = patched_fn

    # Decode
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(max_new - 1):
            logits = run_chunked_decode_step(model, next_token, cache, use_triton=True)
            next_token = logits[:, -1, :].argmax(-1, keepdim=True)
            del logits
    torch.cuda.synchronize()
    decode_ms = (time.perf_counter() - t0) * 1000

    tmod.triton_int8_attention = orig_fn
    tps = (max_new - 1) / (decode_ms / 1000) if decode_ms > 0 else 0

    del cache, next_token
    gc.collect()
    torch.cuda.empty_cache()
    return tps, decode_ms


def main():
    print("=" * 60)
    print("  Phase C: BLOCK_KV Tuning Sweep")
    print("=" * 60)

    # Part 1: Isolated kernel benchmark
    print("\n--- Isolated Kernel Benchmark ---")
    print(f"{'block_kv':>10} | ", end="")
    for sl in SEQ_LENS:
        print(f"seq={sl:>4} (µs) | ", end="")
    print()
    print("-" * 50)

    for bk in BLOCK_SIZES:
        print(f"{bk:>10} | ", end="")
        for sl in SEQ_LENS:
            us = bench_kernel_isolated(sl, bk)
            print(f"{us:>11.1f} | ", end="")
        print()

    # Part 2: End-to-end decode benchmark
    print("\n--- End-to-End Decode Benchmark ---")
    device = get_device()
    model, tokenizer, _ = load_model_and_tokenizer("protgpt2", device=str(device), dtype=torch.float16)
    model.train(False)

    # Warmup
    ids = prepare_prompt(tokenizer, prompt_token_length=16).to(device)
    with torch.no_grad():
        out = model(ids, use_cache=True)
        _ = model(out.logits[:, -1, :].argmax(-1, keepdim=True),
                  past_key_values=out.past_key_values, use_cache=True)
    del out
    gc.collect()
    torch.cuda.empty_cache()

    configs = [
        ("decode_heavy", 32, 256),
        ("long_decode", 256, 512),
    ]

    print(f"\n{'block_kv':>10} | ", end="")
    for label, _, _ in configs:
        print(f"{label:>20} tok/s | ", end="")
    print()
    print("-" * 60)

    for bk in BLOCK_SIZES:
        print(f"{bk:>10} | ", end="")
        for label, ptl, mnt in configs:
            tps, _ = bench_end_to_end(model, tokenizer, ptl, mnt, bk, device)
            print(f"{tps:>20.1f} tok/s | ", end="")
        print()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
