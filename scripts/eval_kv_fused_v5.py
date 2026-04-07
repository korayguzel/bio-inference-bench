#!/usr/bin/env python3
"""Evaluate v5 fused Triton INT8 KV attention kernel.

Compares: baseline FP16 vs v2 (Python chunked INT8) vs v5 (Triton fused INT8).
Each measurement uses a fresh model load.

Usage:
    python scripts/eval_kv_fused_v5.py
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

SANITY_STEPS_SHORT = 64
SANITY_STEPS_LONG = 256  # Extended window for long_decode readiness gate
CONTAMINATION_THRESHOLD_MB = 2.0


def full_cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()


def fresh_model(device):
    full_cleanup()
    return load_model_and_tokenizer("protgpt2", device=str(device), dtype=torch.float16)


def warmup(model, tokenizer):
    ids = prepare_prompt(tokenizer, prompt_token_length=16).to(next(model.parameters()).device)
    with torch.no_grad():
        out = model(ids, use_cache=True)
        _ = model(out.logits[:, -1, :].argmax(-1, keepdim=True),
                  past_key_values=out.past_key_values, use_cache=True)
    del out
    full_cleanup()


def run_path(model, input_ids, max_new, cache=None, label="",
             use_triton=False, sanity_steps=SANITY_STEPS_SHORT) -> dict:
    """Run generation, return metrics dict + tokens/logits for sanity.

    Memory semantics (captured identically for all paths):
    - memory_before_generation_mb: allocated just before prefill (model weights only)
    - memory_after_prefill_mb: allocated after prefill is complete and decode is ready
    - prefill_peak_allocated_mb: max allocated during prefill phase
    - decode_peak_allocated_mb: max allocated during decode phase only
    - overall_peak_allocated_mb: max(prefill_peak, decode_peak)
    - end_to_end_generation_overhead_mb: overall_peak - memory_before_generation
    - decode_phase_growth_mb: decode_peak - memory_after_prefill
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    use_chunked = cache is not None

    reset_memory_tracking()
    mem_before_gen = take_snapshot()

    # --- Prefill (always standard FP16) ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000

    if use_chunked:
        # Transfer prefill KV to our cache
        std_cache = outputs.past_key_values
        for li in range(len(std_cache.layers)):
            layer = std_cache.layers[li]
            cache.update(layer.keys, layer.values, li)
        next_token = outputs.logits[:, -1, :].argmax(-1, keepdim=True)
        tokens = [next_token]
        logits_list = [outputs.logits[:, -1, :].detach().cpu()]
        del outputs, std_cache
        # Cleanup FP16 prefill intermediates, keep INT8 cache
        gc.collect()
        torch.cuda.empty_cache()
    else:
        past = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(-1, keepdim=True)
        tokens = [next_token]
        logits_list = [outputs.logits[:, -1, :].detach().cpu()]
        del outputs

    # Capture prefill peak and post-prefill settled state
    prefill_snap = take_snapshot()
    prefill_peak_mb = prefill_snap.max_allocated_mb
    mem_after_prefill_mb = prefill_snap.allocated_mb

    # Reset peak tracking for decode-only measurement
    torch.cuda.reset_peak_memory_stats()

    # --- Decode ---
    step_times = []
    with torch.no_grad():
        for _ in range(max_new - 1):
            torch.cuda.synchronize()
            t = time.perf_counter()
            if use_chunked:
                logits = run_chunked_decode_step(model, next_token, cache,
                                                 use_triton=use_triton)
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
    decode_peak_mb = decode_snap.max_allocated_mb

    decode_ms = sum(step_times)
    actual = len(tokens)
    total_ms = prefill_ms + decode_ms
    decode_tps = (actual - 1) / (decode_ms / 1000) if decode_ms > 0 else 0
    e2e_tps = actual / (total_ms / 1000) if total_ms > 0 else 0

    # Dual overhead metrics (apples-to-apples across all paths)
    overall_peak_mb = max(prefill_peak_mb, decode_peak_mb)
    e2e_overhead = overall_peak_mb - mem_before_gen.allocated_mb
    decode_growth = decode_peak_mb - mem_after_prefill_mb

    cache_info = {}
    if isinstance(cache, ChunkedInt8KVCache):
        cs = cache.compression_summary()
        cache_info = cs

    result = {
        "method": label,
        "prefill_ms": round(prefill_ms, 2),
        "decode_ms": round(decode_ms, 2),
        "total_ms": round(total_ms, 2),
        "actual_new_tokens": actual,
        "decode_tokens_per_sec": round(decode_tps, 2),
        "end_to_end_tokens_per_sec": round(e2e_tps, 2),
        "memory_before_generation_mb": round(mem_before_gen.allocated_mb, 2),
        "memory_after_prefill_mb": round(mem_after_prefill_mb, 2),
        "prefill_peak_allocated_mb": round(prefill_peak_mb, 2),
        "decode_peak_allocated_mb": round(decode_peak_mb, 2),
        "overall_peak_allocated_mb": round(overall_peak_mb, 2),
        "observed_peak_reserved_mb": round(max(prefill_snap.max_reserved_mb,
                                               decode_snap.max_reserved_mb), 2),
        "end_to_end_generation_overhead_mb": round(e2e_overhead, 2),
        "decode_phase_growth_mb": round(decode_growth, 2),
        "cache_info": cache_info,
        "generated_token_ids": torch.cat(tokens, dim=1).cpu()[0].tolist(),
        "logits_for_sanity": logits_list,
    }

    if not use_chunked:
        del past
    del tokens, next_token
    full_cleanup()
    return result


def sanity(base, proto, max_steps=SANITY_STEPS_SHORT):
    """Compare token agreement and logit similarity."""
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


def strip_for_json(d):
    return {k: v for k, v in d.items() if k not in ("generated_token_ids", "logits_for_sanity")}


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("=" * 60)
    print("  ProtGPT2 Fused Kernel v5: Triton INT8 Attention")
    print("=" * 60)

    device = get_device()
    gpu_info = get_gpu_info()
    print(f"\nGPU: {gpu_info['name']} ({gpu_info['total_mb']:.0f} MB)")

    all_comparisons = []

    for cfg in CONFIGS:
        ptl, mnt, label = cfg["prompt"], cfg["max_new"], cfg["label"]
        # Extended sanity window for long_decode
        sanity_steps = SANITY_STEPS_LONG if label == "long_decode" else SANITY_STEPS_SHORT

        print(f"\n{'━' * 60}")
        print(f"  Config: {label} (prompt={ptl}, max_new={mnt}, sanity={sanity_steps})")
        print(f"{'━' * 60}")

        results = {}

        for path_label, make_cache, triton_flag in [
            ("baseline_fp16", lambda: None, False),
            ("v2_python_chunked", lambda: ChunkedInt8KVCache(chunk_size=64), False),
            ("v5_triton_fused", lambda: ChunkedInt8KVCache(chunk_size=64), True),
        ]:
            print(f"  Loading fresh model for {path_label}...")
            model, tokenizer, _ = fresh_model(device)
            ids = prepare_prompt(tokenizer, prompt_token_length=ptl)
            warmup(model, tokenizer)
            print(f"  Running {path_label}...", end="", flush=True)
            cache = make_cache()
            r = run_path(model, ids, mnt, cache=cache, label=path_label,
                         use_triton=triton_flag, sanity_steps=sanity_steps)
            print(f" {r['decode_tokens_per_sec']} tok/s, "
                  f"peak={r['overall_peak_allocated_mb']} MB, "
                  f"e2e_oh={r['end_to_end_generation_overhead_mb']} MB, "
                  f"decode_growth={r['decode_phase_growth_mb']} MB")
            results[path_label] = r
            del model, tokenizer, cache
            full_cleanup()

        # Isolation check: memory_before_generation should be same for all paths
        pre_gen_loads = [results[k]["memory_before_generation_mb"] for k in results]
        max_delta = max(pre_gen_loads) - min(pre_gen_loads)
        isolation = max_delta < CONTAMINATION_THRESHOLD_MB
        print(f"  Isolation: {'PASSED' if isolation else 'FAILED'} "
              f"(pre-gen delta={max_delta:.2f} MB)")

        # Sanity: 64-step (smoke) for all, plus extended window for long_decode
        san_v2_short = sanity(results["baseline_fp16"], results["v2_python_chunked"],
                              max_steps=SANITY_STEPS_SHORT)
        san_v5_short = sanity(results["baseline_fp16"], results["v5_triton_fused"],
                              max_steps=SANITY_STEPS_SHORT)

        # Extended sanity (uses however many logits were captured)
        san_v2_long = sanity(results["baseline_fp16"], results["v2_python_chunked"],
                             max_steps=sanity_steps)
        san_v5_long = sanity(results["baseline_fp16"], results["v5_triton_fused"],
                             max_steps=sanity_steps)

        print(f"  v2 sanity ({SANITY_STEPS_SHORT}): {san_v2_short['token_agreement_pct']}% token, "
              f"cos={san_v2_short['avg_logit_cosine_similarity']:.6f}")
        print(f"  v5 sanity ({SANITY_STEPS_SHORT}): {san_v5_short['token_agreement_pct']}% token, "
              f"cos={san_v5_short['avg_logit_cosine_similarity']:.6f}")
        if sanity_steps > SANITY_STEPS_SHORT:
            print(f"  v2 sanity ({sanity_steps}): {san_v2_long['token_agreement_pct']}% token, "
                  f"cos={san_v2_long['avg_logit_cosine_similarity']:.6f}")
            print(f"  v5 sanity ({sanity_steps}): {san_v5_long['token_agreement_pct']}% token, "
                  f"cos={san_v5_long['avg_logit_cosine_similarity']:.6f}")

        comparison = {
            "label": label,
            "prompt_token_length": ptl,
            "max_new_tokens": mnt,
            "sanity_steps_short": SANITY_STEPS_SHORT,
            "sanity_steps_long": sanity_steps,
            "isolation_passed": isolation,
            "isolation_pre_gen_delta_mb": round(max_delta, 2),
            "baseline": strip_for_json(results["baseline_fp16"]),
            "v2_python_chunked": strip_for_json(results["v2_python_chunked"]),
            "v5_triton_fused": strip_for_json(results["v5_triton_fused"]),
            "sanity_v2_vs_baseline_short": san_v2_short,
            "sanity_v5_vs_baseline_short": san_v5_short,
            "sanity_v2_vs_baseline_long": san_v2_long,
            "sanity_v5_vs_baseline_long": san_v5_long,
            "speed_v2_vs_base_pct": round(
                (results["v2_python_chunked"]["decode_tokens_per_sec"] -
                 results["baseline_fp16"]["decode_tokens_per_sec"]) /
                results["baseline_fp16"]["decode_tokens_per_sec"] * 100, 1),
            "speed_v5_vs_base_pct": round(
                (results["v5_triton_fused"]["decode_tokens_per_sec"] -
                 results["baseline_fp16"]["decode_tokens_per_sec"]) /
                results["baseline_fp16"]["decode_tokens_per_sec"] * 100, 1),
            "speed_v5_vs_v2_ratio": round(
                results["v5_triton_fused"]["decode_tokens_per_sec"] /
                results["v2_python_chunked"]["decode_tokens_per_sec"], 2)
                if results["v2_python_chunked"]["decode_tokens_per_sec"] > 0 else 0,
        }
        all_comparisons.append(comparison)

    ts = timestamp()
    raw_path = Path("results/raw") / f"kv_fused_v5_eval_{ts}.json"
    with open(raw_path, "w") as f:
        json.dump(all_comparisons, f, indent=2)
    print(f"\nRaw saved to: {raw_path}")
    print(f"\n{'=' * 60}")
    print(f"  v5 evaluation complete")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
