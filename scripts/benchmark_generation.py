#!/usr/bin/env python3
"""Run a single benchmark with full profiling output.

Usage:
    python scripts/benchmark_generation.py \
        --model protgpt2 \
        --prompt-token-length 64 \
        --max-new-tokens 128
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from bio_inference_bench.generation import run_benchmark
from bio_inference_bench.models import load_model_and_tokenizer, prepare_prompt
from bio_inference_bench.progen2_compat import patch_progen2_model
from bio_inference_bench.report import print_benchmark_result, save_result_json
from bio_inference_bench.utils import MODEL_REGISTRY, get_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark autoregressive generation")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()),
                        help="Model to benchmark")
    parser.add_argument("--prompt-token-length", type=int, default=64,
                        help="Prompt length in tokens (not characters)")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="Maximum tokens to generate")
    parser.add_argument("--kv-mode", choices=["fp16", "int8-triton"], default="fp16",
                        help="KV cache mode: fp16 (default) or int8-triton (ProtGPT2 only)")
    parser.add_argument("--output-dir", type=str, default="results/raw",
                        help="Directory for JSON output")
    args = parser.parse_args()

    # INT8-Triton gate: ProtGPT2 only
    if args.kv_mode == "int8-triton" and args.model != "protgpt2":
        print(f"ERROR: INT8-Triton KV mode is only supported for ProtGPT2.")
        print(f"The fused Triton kernel is validated for ProtGPT2 "
              f"(20 heads, 36 layers, head_dim=64) only.")
        print(f"Got: --model {args.model}")
        raise SystemExit(1)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = get_device()

    print(f"\nLoading {args.model}...")
    model, tokenizer, metadata = load_model_and_tokenizer(
        args.model, device=str(device), dtype=torch.float16
    )

    if model is None or tokenizer is None:
        print(f"\nFAILED: Could not load model '{args.model}'")
        for w in metadata.warnings:
            print(f"  - {w}")
        return

    # Apply compatibility patches if needed
    patches = patch_progen2_model(model)
    for p in patches:
        print(f"  PATCHED: {p}")
        metadata.warnings.append(f"Applied patch: {p}")

    print(f"Preparing prompt (prompt_token_length={args.prompt_token_length})...")
    input_ids = prepare_prompt(tokenizer, prompt_token_length=args.prompt_token_length)

    if args.kv_mode == "int8-triton":
        # INT8-Triton benchmark path — canonical structured output
        from bio_inference_bench.int8_generate import generate
        from bio_inference_bench.kv_estimator import estimate_kv_cache, dtype_to_bytes
        from bio_inference_bench.profiler import get_gpu_info
        from bio_inference_bench.report import format_generation_summary

        print(f"Running INT8-Triton benchmark...")
        result = generate(
            model, tokenizer, input_ids, args.max_new_tokens,
            kv_mode="int8-triton", model_name=args.model,
        )
        print(format_generation_summary(result))

        # Compute theoretical KV for reference
        total_seq = args.prompt_token_length + result.get("actual_new_tokens", 0)
        theoretical_kv = estimate_kv_cache(
            num_layers=metadata.num_layers,
            batch_size=1,
            seq_len=total_seq,
            num_kv_heads=metadata.num_kv_heads,
            head_dim=metadata.head_dim,
            dtype_bytes=dtype_to_bytes(torch.float16),
        )

        # Build canonical benchmark JSON
        benchmark_output = {
            "metadata": metadata.to_dict(),
            "kv_mode": "int8-triton",
            "benchmark_config": {
                "model": args.model,
                "prompt_token_length": args.prompt_token_length,
                "max_new_tokens": args.max_new_tokens,
                "chunk_size": 64,
                "use_triton": True,
            },
            "result": {
                "method": result["method"],
                "actual_new_tokens": result["actual_new_tokens"],
                "total_seq_length": args.prompt_token_length + result["actual_new_tokens"],
                "prefill_ms": result["prefill_ms"],
                "decode_ms": result["decode_ms"],
                "total_ms": result["total_ms"],
                "decode_tokens_per_sec": result["decode_tokens_per_sec"],
                "end_to_end_tokens_per_sec": result["end_to_end_tokens_per_sec"],
                "memory_before_generation_mb": result["memory_before_generation_mb"],
                "memory_after_prefill_mb": result["memory_after_prefill_mb"],
                "prefill_peak_allocated_mb": result["prefill_peak_allocated_mb"],
                "decode_peak_allocated_mb": result["decode_peak_allocated_mb"],
                "overall_peak_allocated_mb": result["overall_peak_allocated_mb"],
                "end_to_end_generation_overhead_mb": result["end_to_end_generation_overhead_mb"],
                "decode_phase_growth_mb": result["decode_phase_growth_mb"],
                "decode_growth_per_token_mb": result["decode_growth_per_token_mb"],
                "cache_info": result.get("cache_info", {}),
            },
            "capacity_metrics": {
                "decode_growth_per_token_mb": result["decode_growth_per_token_mb"],
                "theoretical_fp16_kv_at_final_seq_mb": round(theoretical_kv.total_mb, 2),
                "int8_cache_mb": result.get("cache_info", {}).get("total_cache_mb"),
                "compression_ratio": round(
                    result.get("cache_info", {}).get("fp16_equivalent_mb", 0) /
                    max(result.get("cache_info", {}).get("total_cache_mb", 1), 0.01), 2
                ),
            },
            "notes": [
                "INT8-Triton path: ProtGPT2 only, batch=1, decode-only INT8",
                "Prefill uses standard FP16 attention",
                "Memory semantics: prefill and decode peaks tracked independently",
            ],
            "gpu_info": get_gpu_info(),
        }

        output_dir = Path(args.output_dir)
        json_path = save_result_json(benchmark_output, output_dir,
                                     prefix=f"{args.model}_int8triton")
        print(f"\nResults saved to: {json_path}")
    else:
        # Standard FP16 benchmark path (existing behavior)
        print(f"Running benchmark...")
        results = run_benchmark(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            metadata=metadata,
        )

        print_benchmark_result(results["primary_result"], path_label="PRIMARY: manual prefill/decode")
        print_benchmark_result(results["secondary_result"], path_label="SECONDARY: generate API")

        output_dir = Path(args.output_dir)
        json_path = save_result_json(
            {
                "metadata": metadata.to_dict(),
                "primary_result": results["primary_result"].to_dict(),
                "secondary_result": results["secondary_result"].to_dict(),
                "theoretical_kv_upper_bound": results["theoretical_kv_upper_bound"],
            },
            output_dir,
            prefix=args.model,
        )
        print(f"\nResults saved to: {json_path}")

    # Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
