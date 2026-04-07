#!/usr/bin/env python3
"""Evaluate v3 boundary-layer-protected INT8 KV prototype.

Compares: baseline FP16 vs v2 (all INT8) vs v3 (boundary-protected INT8).
Each measurement uses a fresh model load.

Usage:
    python scripts/eval_kv_prototype_v3.py
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

# ProtGPT2 has 36 layers (0-35). Protect first 2 + last 2.
V3_PROTECTED_LAYERS = {0, 1, 34, 35}
SANITY_STEPS = 64
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


def run_path(model, input_ids, max_new, cache=None, label="") -> dict:
    """Run generation, return metrics dict + tokens/logits for sanity."""
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    use_chunked = cache is not None

    reset_memory_tracking()
    mem_load = take_snapshot()

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
        full_cleanup()
        reset_memory_tracking()
        mem_load = take_snapshot()  # re-baseline after prefill cleanup
    else:
        past = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(-1, keepdim=True)
        tokens = [next_token]
        logits_list = [outputs.logits[:, -1, :].detach().cpu()]
        del outputs

    # --- Decode ---
    step_times = []
    with torch.no_grad():
        for _ in range(max_new - 1):
            torch.cuda.synchronize()
            t = time.perf_counter()
            if use_chunked:
                logits = run_chunked_decode_step(model, next_token, cache)
            else:
                out = model(next_token, past_key_values=past, use_cache=True)
                logits = out.logits
                past = out.past_key_values
                del out
            torch.cuda.synchronize()
            step_times.append((time.perf_counter() - t) * 1000)
            next_token = logits[:, -1, :].argmax(-1, keepdim=True)
            tokens.append(next_token)
            if len(logits_list) < SANITY_STEPS:
                logits_list.append(logits[:, -1, :].detach().cpu())
            del logits

    peak = take_snapshot()
    decode_ms = sum(step_times)
    actual = len(tokens)
    total_ms = prefill_ms + decode_ms
    decode_tps = (actual - 1) / (decode_ms / 1000) if decode_ms > 0 else 0
    e2e_tps = actual / (total_ms / 1000) if total_ms > 0 else 0
    gen_overhead = peak.max_allocated_mb - mem_load.allocated_mb

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
        "memory_after_load_mb": round(mem_load.allocated_mb, 2),
        "observed_peak_allocated_mb": round(peak.max_allocated_mb, 2),
        "observed_peak_reserved_mb": round(peak.max_reserved_mb, 2),
        "generation_overhead_above_load_mb": round(gen_overhead, 2),
        "cache_info": cache_info,
        "generated_token_ids": torch.cat(tokens, dim=1).cpu()[0].tolist(),
        "logits_for_sanity": logits_list,
    }

    if not use_chunked:
        del past
    del tokens, next_token
    full_cleanup()
    return result


def sanity(base, proto):
    bt, pt = base["generated_token_ids"], proto["generated_token_ids"]
    n = min(len(bt), len(pt), SANITY_STEPS)
    agree = sum(1 for a, b in zip(bt[:n], pt[:n]) if a == b)
    first_diff = next((i for i in range(n) if bt[i] != pt[i]), n)
    bl, pl = base["logits_for_sanity"], proto["logits_for_sanity"]
    ln = min(len(bl), len(pl))
    top1 = sum(1 for i in range(ln) if bl[i].float().argmax() == pl[i].float().argmax())
    cosines = [torch.nn.functional.cosine_similarity(bl[i].float(), pl[i].float(), dim=-1).item() for i in range(ln)]
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
    print("  ProtGPT2 KV Prototype v3: Boundary-Layer Protection")
    print(f"  Protected layers: {sorted(V3_PROTECTED_LAYERS)}")
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

        results = {}

        for path_label, make_cache in [
            ("baseline_fp16", lambda: None),
            ("v2_all_int8", lambda: ChunkedInt8KVCache(chunk_size=64)),
            ("v3_boundary", lambda: ChunkedInt8KVCache(chunk_size=64, protected_layers=V3_PROTECTED_LAYERS)),
        ]:
            print(f"  Loading fresh model for {path_label}...")
            model, tokenizer, _ = fresh_model(device)
            ids = prepare_prompt(tokenizer, prompt_token_length=ptl)
            warmup(model, tokenizer)
            print(f"  Running {path_label}...", end="", flush=True)
            cache = make_cache()
            r = run_path(model, ids, mnt, cache=cache, label=path_label)
            print(f" {r['decode_tokens_per_sec']} tok/s, "
                  f"peak={r['observed_peak_allocated_mb']} MB, "
                  f"overhead={r['generation_overhead_above_load_mb']} MB")
            results[path_label] = r
            del model, tokenizer, cache
            full_cleanup()

        # Isolation check: all three should have same after_load
        loads = [results[k]["memory_after_load_mb"] for k in results]
        max_delta = max(loads) - min(loads)
        isolation = max_delta < CONTAMINATION_THRESHOLD_MB
        print(f"  Isolation: {'PASSED' if isolation else 'FAILED'} (max delta={max_delta:.2f} MB)")

        # Sanity: compare v2 and v3 against baseline
        san_v2 = sanity(results["baseline_fp16"], results["v2_all_int8"])
        san_v3 = sanity(results["baseline_fp16"], results["v3_boundary"])
        print(f"  v2 sanity: {san_v2['token_agreement_pct']}% token, cos={san_v2['avg_logit_cosine_similarity']:.6f}")
        print(f"  v3 sanity: {san_v3['token_agreement_pct']}% token, cos={san_v3['avg_logit_cosine_similarity']:.6f}")

        base_oh = results["baseline_fp16"]["generation_overhead_above_load_mb"]
        v2_oh = results["v2_all_int8"]["generation_overhead_above_load_mb"]
        v3_oh = results["v3_boundary"]["generation_overhead_above_load_mb"]

        comparison = {
            "label": label,
            "prompt_token_length": ptl,
            "max_new_tokens": mnt,
            "protected_layers": sorted(V3_PROTECTED_LAYERS),
            "isolation_passed": isolation,
            "baseline": strip_for_json(results["baseline_fp16"]),
            "v2_all_int8": strip_for_json(results["v2_all_int8"]),
            "v3_boundary": strip_for_json(results["v3_boundary"]),
            "sanity_v2_vs_baseline": san_v2,
            "sanity_v3_vs_baseline": san_v3,
            "memory_v2_vs_base_pct": round((base_oh - v2_oh) / base_oh * 100, 1) if base_oh > 0 else 0,
            "memory_v3_vs_base_pct": round((base_oh - v3_oh) / base_oh * 100, 1) if base_oh > 0 else 0,
            "speed_v2_vs_base_pct": round((results["v2_all_int8"]["decode_tokens_per_sec"] - results["baseline_fp16"]["decode_tokens_per_sec"]) / results["baseline_fp16"]["decode_tokens_per_sec"] * 100, 1),
            "speed_v3_vs_base_pct": round((results["v3_boundary"]["decode_tokens_per_sec"] - results["baseline_fp16"]["decode_tokens_per_sec"]) / results["baseline_fp16"]["decode_tokens_per_sec"] * 100, 1),
        }
        all_comparisons.append(comparison)

    ts = timestamp()
    raw_path = Path("results/raw") / f"kv_prototype_v3_eval_{ts}.json"
    with open(raw_path, "w") as f:
        json.dump(all_comparisons, f, indent=2)
    print(f"\nRaw saved to: {raw_path}")
    print(f"\n{'=' * 60}")
    print(f"  v3 evaluation complete")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
