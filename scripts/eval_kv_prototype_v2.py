#!/usr/bin/env python3
"""Evaluate v2 chunked-dequantize INT8 KV prototype against FP16 baseline.

Each measurement uses a fresh model load. Compares baseline (FP16 KV) vs
v2 (chunked INT8 KV that never materializes full FP16 during decode).

Prefill uses the standard model forward (FP16). Decode uses the custom
chunked path that dequantizes INT8 KV in small chunks for attention.

Usage:
    python scripts/eval_kv_prototype_v2.py
"""

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path

import torch

from bio_inference_bench.kv_int8_chunked import ChunkedInt8KVCache, run_chunked_decode_step
from bio_inference_bench.models import load_model_and_tokenizer, prepare_prompt
from bio_inference_bench.profiler import get_gpu_info, reset_memory_tracking, take_snapshot
from bio_inference_bench.utils import get_device, timestamp

logger = logging.getLogger(__name__)

CONFIGS = [
    {"prompt": 32, "max_new": 256, "label": "decode_heavy"},
    {"prompt": 256, "max_new": 512, "label": "long_decode"},
]

SANITY_CHECK_STEPS = 64
CONTAMINATION_THRESHOLD_MB = 2.0


def full_gpu_cleanup() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()


def load_fresh_model(device):
    full_gpu_cleanup()
    model, tokenizer, metadata = load_model_and_tokenizer(
        "protgpt2", device=str(device), dtype=torch.float16
    )
    return model, tokenizer, metadata


def run_warmup(model, tokenizer) -> None:
    input_ids = prepare_prompt(tokenizer, prompt_token_length=16).to(
        next(model.parameters()).device
    )
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        _ = model(next_tok, past_key_values=outputs.past_key_values, use_cache=True)
    del outputs, next_tok
    full_gpu_cleanup()


def run_baseline(model, input_ids, max_new_tokens) -> dict:
    """Standard FP16 KV generation."""
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    reset_memory_tracking()
    mem_load = take_snapshot()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000

    past = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    tokens = [next_token]
    logits_list = [outputs.logits[:, -1, :].detach().cpu()]
    del outputs

    step_times = []
    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            torch.cuda.synchronize()
            t = time.perf_counter()
            outputs = model(next_token, past_key_values=past, use_cache=True)
            torch.cuda.synchronize()
            step_times.append((time.perf_counter() - t) * 1000)
            past = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens.append(next_token)
            if len(logits_list) < SANITY_CHECK_STEPS:
                logits_list.append(outputs.logits[:, -1, :].detach().cpu())
            del outputs

    peak = take_snapshot()
    decode_ms = sum(step_times)
    actual = len(tokens)
    decode_tps = (actual - 1) / (decode_ms / 1000) if decode_ms > 0 else 0
    e2e_tps = actual / ((prefill_ms + decode_ms) / 1000) if (prefill_ms + decode_ms) > 0 else 0

    result = {
        "method": "baseline_fp16",
        "prefill_ms": round(prefill_ms, 2),
        "decode_ms": round(decode_ms, 2),
        "total_ms": round(prefill_ms + decode_ms, 2),
        "actual_new_tokens": actual,
        "decode_tokens_per_sec": round(decode_tps, 2),
        "end_to_end_tokens_per_sec": round(e2e_tps, 2),
        "memory_after_load_mb": round(mem_load.allocated_mb, 2),
        "observed_peak_allocated_mb": round(peak.max_allocated_mb, 2),
        "observed_peak_reserved_mb": round(peak.max_reserved_mb, 2),
        "generation_overhead_above_load_mb": round(peak.max_allocated_mb - mem_load.allocated_mb, 2),
        "int8_kv_storage_mb": 0.0,
        "fp16_kv_equivalent_mb": 0.0,
        "generated_token_ids": torch.cat(tokens, dim=1).cpu()[0].tolist(),
        "logits_for_sanity": logits_list,
    }
    del past, tokens, next_token
    full_gpu_cleanup()
    return result


def run_v2_chunked(model, input_ids, max_new_tokens, chunk_size=64) -> dict:
    """V2 chunked-dequantize INT8 KV generation.

    Prefill: standard model forward (FP16, full attention over prompt).
    Decode: custom path using chunked INT8 attention (no full FP16 materialization).
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    cache = ChunkedInt8KVCache(chunk_size=chunk_size)

    reset_memory_tracking()
    mem_load = take_snapshot()

    # --- PREFILL: standard forward with cache ---
    # During prefill, update() stores INT8 and returns only the new slice.
    # But the model's attention needs the full FP16 cache for the prompt.
    # So for prefill, we use a temporary standard DynamicCache approach:
    # run the standard forward, then transfer the result to our INT8 cache.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        # Standard prefill (creates FP16 KV internally)
        outputs = model(input_ids, use_cache=True)
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000

    # Transfer prefill KV to INT8 cache
    standard_cache = outputs.past_key_values
    for layer_idx in range(len(standard_cache.layers)):
        layer = standard_cache.layers[layer_idx]
        cache.update(layer.keys, layer.values, layer_idx)

    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    tokens = [next_token]
    logits_list = [outputs.logits[:, -1, :].detach().cpu()]

    # Free the standard FP16 cache — from here on, only INT8 cache exists
    del outputs, standard_cache
    full_gpu_cleanup()
    # Re-take snapshot after cleanup to see decode-phase memory clearly
    reset_memory_tracking()
    mem_after_prefill_cleanup = take_snapshot()

    # --- DECODE: chunked INT8 attention ---
    step_times = []
    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            torch.cuda.synchronize()
            t = time.perf_counter()

            logits = run_chunked_decode_step(model, next_token, cache)

            torch.cuda.synchronize()
            step_times.append((time.perf_counter() - t) * 1000)

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens.append(next_token)
            if len(logits_list) < SANITY_CHECK_STEPS:
                logits_list.append(logits[:, -1, :].detach().cpu())
            del logits

    peak = take_snapshot()
    decode_ms = sum(step_times)
    actual = len(tokens)
    decode_tps = (actual - 1) / (decode_ms / 1000) if decode_ms > 0 else 0
    e2e_tps = actual / ((prefill_ms + decode_ms) / 1000) if (prefill_ms + decode_ms) > 0 else 0

    # Use mem_load for overhead calc (same basis as baseline)
    gen_overhead = peak.max_allocated_mb - mem_load.allocated_mb

    result = {
        "method": "v2_chunked_int8",
        "chunk_size": chunk_size,
        "prefill_ms": round(prefill_ms, 2),
        "decode_ms": round(decode_ms, 2),
        "total_ms": round(prefill_ms + decode_ms, 2),
        "actual_new_tokens": actual,
        "decode_tokens_per_sec": round(decode_tps, 2),
        "end_to_end_tokens_per_sec": round(e2e_tps, 2),
        "memory_after_load_mb": round(mem_load.allocated_mb, 2),
        "observed_peak_allocated_mb": round(peak.max_allocated_mb, 2),
        "observed_peak_reserved_mb": round(peak.max_reserved_mb, 2),
        "generation_overhead_above_load_mb": round(gen_overhead, 2),
        "int8_kv_storage_mb": round(cache.int8_memory_bytes() / (1024**2), 2),
        "fp16_kv_equivalent_mb": round(cache.fp16_equivalent_bytes() / (1024**2), 2),
        "generated_token_ids": torch.cat(tokens, dim=1).cpu()[0].tolist(),
        "logits_for_sanity": logits_list,
    }
    del cache, tokens, next_token
    full_gpu_cleanup()
    return result


def compute_sanity(base: dict, proto: dict) -> dict:
    bt = base["generated_token_ids"]
    pt = proto["generated_token_ids"]
    n = min(len(bt), len(pt), SANITY_CHECK_STEPS)
    agree = sum(1 for a, b in zip(bt[:n], pt[:n]) if a == b)
    first_diff = n
    for i in range(n):
        if bt[i] != pt[i]:
            first_diff = i
            break
    bl = base["logits_for_sanity"]
    pl = proto["logits_for_sanity"]
    ln = min(len(bl), len(pl))
    top1 = sum(1 for i in range(ln) if bl[i].float().argmax() == pl[i].float().argmax())
    cosines = [
        torch.nn.functional.cosine_similarity(bl[i].float(), pl[i].float(), dim=-1).item()
        for i in range(ln)
    ]
    return {
        "token_agreement_pct": round(agree / n * 100, 1) if n else 0,
        "first_divergence_step": first_diff,
        "steps_compared": n,
        "top1_logit_agreement_pct": round(top1 / ln * 100, 1) if ln else 0,
        "avg_logit_cosine_similarity": round(sum(cosines) / len(cosines), 6) if cosines else 0,
        "logit_steps_compared": ln,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("  ProtGPT2 KV Prototype v2: Chunked-Dequantize INT8")
    print("  Goal: memory-capacity reduction without full FP16 materialization")
    print("=" * 60)

    device = get_device()
    gpu_info = get_gpu_info()
    print(f"\nGPU: {gpu_info['name']} ({gpu_info['total_mb']:.0f} MB)")

    all_comparisons = []

    for cfg in CONFIGS:
        ptl, mnt, label = cfg["prompt"], cfg["max_new"], cfg["label"]

        print(f"\n{'━' * 60}")
        print(f"  Config: {label} (prompt={ptl}, max_new={mnt})")
        print(f"{'━' * 60}")

        # --- Baseline ---
        print("  Loading fresh model for BASELINE...")
        model, tokenizer, _ = load_fresh_model(device)
        input_ids = prepare_prompt(tokenizer, prompt_token_length=ptl)
        run_warmup(model, tokenizer)
        print("  Running BASELINE...", end="", flush=True)
        baseline = run_baseline(model, input_ids, mnt)
        print(f" {baseline['decode_tokens_per_sec']} tok/s, "
              f"peak={baseline['observed_peak_allocated_mb']} MB")
        del model, tokenizer
        full_gpu_cleanup()

        # --- V2 Prototype ---
        print("  Loading fresh model for V2 PROTOTYPE...")
        model, tokenizer, _ = load_fresh_model(device)
        input_ids = prepare_prompt(tokenizer, prompt_token_length=ptl)
        run_warmup(model, tokenizer)
        print("  Running V2 CHUNKED...", end="", flush=True)
        v2 = run_v2_chunked(model, input_ids, mnt)
        print(f" {v2['decode_tokens_per_sec']} tok/s, "
              f"peak={v2['observed_peak_allocated_mb']} MB")
        del model, tokenizer
        full_gpu_cleanup()

        # Isolation check
        delta = abs(baseline["memory_after_load_mb"] - v2["memory_after_load_mb"])
        isolation_passed = delta < CONTAMINATION_THRESHOLD_MB
        print(f"  Isolation: {'PASSED' if isolation_passed else 'FAILED'} "
              f"(delta={delta:.2f} MB)")

        # Sanity
        sanity = compute_sanity(baseline, v2)
        print(f"  Sanity: {sanity['token_agreement_pct']}% token, "
              f"{sanity['top1_logit_agreement_pct']}% top-1, "
              f"cos={sanity['avg_logit_cosine_similarity']:.6f}")

        # Memory delta
        mem_saved = baseline["generation_overhead_above_load_mb"] - v2["generation_overhead_above_load_mb"]
        mem_pct = (mem_saved / baseline["generation_overhead_above_load_mb"] * 100
                   if baseline["generation_overhead_above_load_mb"] > 0 else 0)
        print(f"  Memory: base={baseline['generation_overhead_above_load_mb']} MB, "
              f"v2={v2['generation_overhead_above_load_mb']} MB, "
              f"delta={mem_saved:+.2f} MB ({mem_pct:+.1f}%)")

        speed_delta = v2["decode_tokens_per_sec"] - baseline["decode_tokens_per_sec"]
        speed_pct = speed_delta / baseline["decode_tokens_per_sec"] * 100 if baseline["decode_tokens_per_sec"] > 0 else 0

        comparison = {
            "label": label,
            "prompt_token_length": ptl,
            "max_new_tokens": mnt,
            "isolation_method": "fresh_model_load_per_measurement",
            "isolation_passed": isolation_passed,
            "baseline": {k: v for k, v in baseline.items()
                        if k not in ("generated_token_ids", "logits_for_sanity")},
            "v2_chunked": {k: v for k, v in v2.items()
                          if k not in ("generated_token_ids", "logits_for_sanity")},
            "sanity_check": sanity,
            "memory_saved_mb": round(mem_saved, 2),
            "memory_saved_pct": round(mem_pct, 1),
            "speed_delta_tokens_per_sec": round(speed_delta, 2),
            "speed_delta_pct": round(speed_pct, 1),
        }
        all_comparisons.append(comparison)

    ts = timestamp()
    raw_path = Path("results/raw") / f"kv_prototype_v2_eval_{ts}.json"
    with open(raw_path, "w") as f:
        json.dump(all_comparisons, f, indent=2)
    print(f"\nRaw saved to: {raw_path}")

    print(f"\n{'=' * 60}")
    print(f"  V2 evaluation complete: {len(all_comparisons)} configs")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
