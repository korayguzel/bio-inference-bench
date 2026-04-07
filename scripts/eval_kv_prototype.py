#!/usr/bin/env python3
"""Evaluate the INT8 KV cache prototype against FP16 baseline for ProtGPT2.

Each measurement (baseline or prototype) gets a fresh model load to prevent
cross-run memory contamination. A warmup run precedes each timed measurement.

Usage:
    python scripts/eval_kv_prototype.py
"""

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path

import torch

from bio_inference_bench.kv_int8_cache import Int8KVCache
from bio_inference_bench.models import load_model_and_tokenizer, prepare_prompt
from bio_inference_bench.profiler import get_gpu_info, reset_memory_tracking, take_snapshot
from bio_inference_bench.utils import get_device, timestamp

logger = logging.getLogger(__name__)

EVAL_CONFIGS = [
    {"prompt": 32, "max_new": 256, "label": "decode_heavy"},
    {"prompt": 256, "max_new": 512, "label": "long_decode"},
]

SANITY_CHECK_STEPS = 64
CONTAMINATION_THRESHOLD_MB = 2.0  # flag if baseline/prototype after_load differ by more


def full_gpu_cleanup() -> None:
    """Aggressively free all GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()


def load_fresh_model(device):
    """Load a fresh ProtGPT2 instance. Returns (model, tokenizer, metadata)."""
    full_gpu_cleanup()
    model, tokenizer, metadata = load_model_and_tokenizer(
        "protgpt2", device=str(device), dtype=torch.float16
    )
    return model, tokenizer, metadata


def run_warmup(model, tokenizer) -> None:
    """Single short warmup run to JIT-compile CUDA kernels."""
    input_ids = prepare_prompt(tokenizer, prompt_token_length=16).to(
        next(model.parameters()).device
    )
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        # One decode step
        next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        _ = model(next_tok, past_key_values=outputs.past_key_values, use_cache=True)
    del outputs, next_tok
    full_gpu_cleanup()


def run_generation(
    model, input_ids: torch.Tensor, max_new_tokens: int, cache=None,
) -> dict:
    """Run manual prefill/decode and collect metrics.

    Args:
        cache: None for default (FP16), or an Int8KVCache instance.
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    reset_memory_tracking()
    mem_after_load = take_snapshot()

    torch.cuda.synchronize()
    t_prefill = time.perf_counter()

    with torch.no_grad():
        if cache is not None:
            outputs = model(input_ids, use_cache=True, past_key_values=cache)
        else:
            outputs = model(input_ids, use_cache=True)

    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t_prefill) * 1000

    past = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    tokens = [next_token]
    logits_list = [outputs.logits[:, -1, :].detach().cpu()]
    del outputs  # free prefill intermediates

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
            del outputs  # free per-step intermediates

    peak = take_snapshot()
    decode_ms = sum(step_times)
    total_ms = prefill_ms + decode_ms
    actual_tokens = len(tokens)
    decode_tps = (actual_tokens - 1) / (decode_ms / 1000) if decode_ms > 0 else 0
    e2e_tps = actual_tokens / (total_ms / 1000) if total_ms > 0 else 0

    gen_overhead = peak.max_allocated_mb - mem_after_load.allocated_mb

    # INT8 cache stats
    int8_storage_mb = 0.0
    fp16_equiv_mb = 0.0
    if isinstance(past, Int8KVCache):
        int8_storage_mb = past.int8_memory_bytes() / (1024**2)
        fp16_equiv_mb = past.fp16_equivalent_bytes() / (1024**2)

    all_tokens = torch.cat(tokens, dim=1).cpu()

    result = {
        "prefill_ms": round(prefill_ms, 2),
        "decode_ms": round(decode_ms, 2),
        "total_ms": round(total_ms, 2),
        "actual_new_tokens": actual_tokens,
        "decode_tokens_per_sec": round(decode_tps, 2),
        "end_to_end_tokens_per_sec": round(e2e_tps, 2),
        "memory_after_load_mb": round(mem_after_load.allocated_mb, 2),
        "observed_peak_allocated_mb": round(peak.max_allocated_mb, 2),
        "observed_peak_reserved_mb": round(peak.max_reserved_mb, 2),
        "generation_overhead_above_load_mb": round(gen_overhead, 2),
        "int8_kv_storage_mb": round(int8_storage_mb, 2),
        "fp16_kv_equivalent_mb": round(fp16_equiv_mb, 2),
        "generated_token_ids": all_tokens[0].tolist(),
        "logits_for_sanity": logits_list,
    }

    # Explicit cleanup of cache and generation state
    del past, tokens, next_token, all_tokens
    full_gpu_cleanup()

    return result


def check_isolation(
    label: str, baseline_after_load: float, prototype_after_load: float
) -> bool:
    """Check baseline/prototype started from comparable memory state.

    Returns True if after_load values match within threshold, meaning
    neither run was contaminated by residual allocations from prior runs.
    """
    delta = abs(baseline_after_load - prototype_after_load)
    if delta > CONTAMINATION_THRESHOLD_MB:
        print(f"  CONTAMINATION: baseline after_load={baseline_after_load:.2f} vs "
              f"prototype={prototype_after_load:.2f} (delta={delta:.2f} MB)")
        return False
    print(f"  Isolation check: PASSED (after_load delta={delta:.2f} MB)")
    return True


def compute_sanity_check(baseline: dict, prototype: dict) -> dict:
    """Compare baseline vs prototype token agreement and logits similarity."""
    base_tokens = baseline["generated_token_ids"]
    proto_tokens = prototype["generated_token_ids"]
    n = min(len(base_tokens), len(proto_tokens), SANITY_CHECK_STEPS)

    agree = sum(1 for a, b in zip(base_tokens[:n], proto_tokens[:n]) if a == b)
    agreement_pct = agree / n * 100 if n > 0 else 0

    first_diff = n
    for i in range(n):
        if base_tokens[i] != proto_tokens[i]:
            first_diff = i
            break

    base_logits = baseline["logits_for_sanity"]
    proto_logits = prototype["logits_for_sanity"]
    logit_n = min(len(base_logits), len(proto_logits))
    top1_agree = 0
    cosine_sims = []
    for i in range(logit_n):
        bl = base_logits[i].float()
        pl = proto_logits[i].float()
        if bl.argmax() == pl.argmax():
            top1_agree += 1
        cos = torch.nn.functional.cosine_similarity(bl, pl, dim=-1).item()
        cosine_sims.append(cos)
    top1_agreement_pct = top1_agree / logit_n * 100 if logit_n > 0 else 0
    avg_cosine = sum(cosine_sims) / len(cosine_sims) if cosine_sims else 0

    return {
        "token_agreement_pct": round(agreement_pct, 1),
        "first_divergence_step": first_diff,
        "steps_compared": n,
        "top1_logit_agreement_pct": round(top1_agreement_pct, 1),
        "avg_logit_cosine_similarity": round(avg_cosine, 6),
        "logit_steps_compared": logit_n,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("  ProtGPT2 INT8 KV Cache Prototype Evaluation (Isolated)")
    print("  Each measurement uses a fresh model load.")
    print("=" * 60)

    device = get_device()
    gpu_info = get_gpu_info()
    print(f"\nGPU: {gpu_info['name']} ({gpu_info['total_mb']:.0f} MB)")

    all_comparisons = []

    for cfg in EVAL_CONFIGS:
        ptl = cfg["prompt"]
        mnt = cfg["max_new"]
        label = cfg["label"]

        print(f"\n{'━' * 60}")
        print(f"  Config: {label} (prompt={ptl}, max_new={mnt})")
        print(f"{'━' * 60}")

        # --- BASELINE: fresh model load ---
        print("  Loading fresh model for BASELINE...")
        model, tokenizer, metadata = load_fresh_model(device)
        if model is None:
            print("  FAILED to load ProtGPT2")
            continue

        input_ids = prepare_prompt(tokenizer, prompt_token_length=ptl)
        print("  Warming up baseline...", end="", flush=True)
        run_warmup(model, tokenizer)
        print(" done")

        print("  Running BASELINE (FP16 KV)...", end="", flush=True)
        baseline = run_generation(model, input_ids, mnt, cache=None)
        print(f" {baseline['decode_tokens_per_sec']} tok/s, "
              f"peak={baseline['observed_peak_allocated_mb']} MB, "
              f"after_load={baseline['memory_after_load_mb']} MB")

        # Fully unload baseline model
        del model, tokenizer
        full_gpu_cleanup()

        # --- PROTOTYPE: fresh model load ---
        print("  Loading fresh model for PROTOTYPE...")
        model, tokenizer, metadata = load_fresh_model(device)
        if model is None:
            print("  FAILED to load ProtGPT2")
            continue

        input_ids = prepare_prompt(tokenizer, prompt_token_length=ptl)
        print("  Warming up prototype...", end="", flush=True)
        run_warmup(model, tokenizer)
        print(" done")

        print("  Running PROTOTYPE (INT8 KV)...", end="", flush=True)
        int8_cache = Int8KVCache()
        prototype = run_generation(model, input_ids, mnt, cache=int8_cache)
        print(f" {prototype['decode_tokens_per_sec']} tok/s, "
              f"peak={prototype['observed_peak_allocated_mb']} MB, "
              f"after_load={prototype['memory_after_load_mb']} MB")

        # Fully unload prototype model + cache
        del model, tokenizer, int8_cache
        full_gpu_cleanup()

        # --- Isolation check ---
        isolation_passed = check_isolation(
            label, baseline["memory_after_load_mb"], prototype["memory_after_load_mb"]
        )

        # --- Compare ---
        sanity = compute_sanity_check(baseline, prototype)
        print(f"  Sanity: {sanity['token_agreement_pct']}% token agreement, "
              f"{sanity['top1_logit_agreement_pct']}% top-1 logit, "
              f"cos_sim={sanity['avg_logit_cosine_similarity']:.6f}")

        mem_saved = baseline["generation_overhead_above_load_mb"] - prototype["generation_overhead_above_load_mb"]
        mem_pct = (mem_saved / baseline["generation_overhead_above_load_mb"] * 100
                   if baseline["generation_overhead_above_load_mb"] > 0 else 0)
        print(f"  Memory: baseline gen_overhead={baseline['generation_overhead_above_load_mb']} MB, "
              f"prototype={prototype['generation_overhead_above_load_mb']} MB, "
              f"delta={mem_saved:+.2f} MB ({mem_pct:+.1f}%)")

        if prototype["int8_kv_storage_mb"] > 0:
            print(f"  INT8 KV storage: {prototype['int8_kv_storage_mb']} MB "
                  f"(FP16 equiv: {prototype['fp16_kv_equivalent_mb']} MB, "
                  f"ratio: {prototype['fp16_kv_equivalent_mb'] / prototype['int8_kv_storage_mb']:.2f}x)")

        speed_delta = prototype["decode_tokens_per_sec"] - baseline["decode_tokens_per_sec"]
        speed_pct = speed_delta / baseline["decode_tokens_per_sec"] * 100 if baseline["decode_tokens_per_sec"] > 0 else 0

        comparison = {
            "label": label,
            "prompt_token_length": ptl,
            "max_new_tokens": mnt,
            "isolation_method": "fresh_model_load_per_measurement",
            "isolation_passed": isolation_passed,
            "baseline": {k: v for k, v in baseline.items()
                        if k not in ("generated_token_ids", "logits_for_sanity")},
            "prototype": {k: v for k, v in prototype.items()
                         if k not in ("generated_token_ids", "logits_for_sanity")},
            "sanity_check": sanity,
            "memory_saved_mb": round(mem_saved, 2),
            "memory_saved_pct": round(mem_pct, 1),
            "speed_delta_tokens_per_sec": round(speed_delta, 2),
            "speed_delta_pct": round(speed_pct, 1),
        }
        all_comparisons.append(comparison)

    # Save results
    ts = timestamp()
    raw_path = Path("results/raw") / f"kv_prototype_eval_{ts}.json"
    with open(raw_path, "w") as f:
        json.dump(all_comparisons, f, indent=2)
    print(f"\nRaw comparison saved to: {raw_path}")

    print(f"\n{'=' * 60}")
    print(f"  Evaluation complete: {len(all_comparisons)} configs")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
