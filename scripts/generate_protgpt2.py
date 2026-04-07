#!/usr/bin/env python3
"""Generate protein sequences with ProtGPT2 using FP16 or INT8-Triton KV mode.

This is the primary user-facing script for the INT8 KV capacity path.

Usage:
    # FP16 baseline with real prompt
    python scripts/generate_protgpt2.py --prompt "MKTLLILAVL"

    # INT8-Triton mode
    python scripts/generate_protgpt2.py --prompt "MKTLLILAVL" --kv-mode int8-triton

    # Compare both modes (fresh model load per mode, capacity table)
    python scripts/generate_protgpt2.py --compare

    # Benchmark fallback (synthetic prompt)
    python scripts/generate_protgpt2.py --prompt-token-length 64 --kv-mode int8-triton
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

import torch

from bio_inference_bench.int8_generate import generate
from bio_inference_bench.models import load_model_and_tokenizer, prepare_prompt
from bio_inference_bench.profiler import get_gpu_info
from bio_inference_bench.report import format_capacity_table, format_generation_summary
from bio_inference_bench.utils import get_device, timestamp

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "MKTLLILAVLCLGFASATETNEKFNKEMQKFLENIQA"


def tokenize_prompt(tokenizer, args) -> torch.Tensor:
    """Resolve prompt from args and tokenize."""
    if args.prompt is not None:
        ids = tokenizer.encode(args.prompt, return_tensors="pt")
    elif args.prompt_file is not None:
        text = Path(args.prompt_file).read_text().strip().split("\n")[0]
        ids = tokenizer.encode(text, return_tensors="pt")
    elif args.prompt_token_length is not None:
        ids = prepare_prompt(tokenizer, prompt_token_length=args.prompt_token_length)
    else:
        ids = tokenizer.encode(DEFAULT_PROMPT, return_tensors="pt")
    return ids


def full_cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()


def run_single(args):
    """Run a single generation in the selected kv_mode."""
    device = get_device()
    gpu_info = get_gpu_info()

    mode_desc = "INT8-Triton (capacity mode)" if args.kv_mode == "int8-triton" else "FP16 (standard)"
    print(f"ProtGPT2 v5 — {mode_desc}")
    print(f"GPU: {gpu_info['name']} ({gpu_info['total_mb']:.0f} MB)")

    model, tokenizer, metadata = load_model_and_tokenizer(
        "protgpt2", device=str(device), dtype=torch.float16
    )
    input_ids = tokenize_prompt(tokenizer, args)
    print(f"Prompt: {input_ids.shape[1]} tokens | Generating: {args.max_new_tokens} new tokens\n")

    result = generate(
        model, tokenizer, input_ids, args.max_new_tokens,
        kv_mode=args.kv_mode, model_name="protgpt2",
        chunk_size=args.chunk_size,
    )

    print(format_generation_summary(result))

    # Decode tokens
    text = tokenizer.decode(result["generated_token_ids"], skip_special_tokens=True)
    print(f"\nGenerated sequence ({len(result['generated_token_ids'])} tokens):")
    print(text[:500] + ("..." if len(text) > 500 else ""))

    # Save raw JSON
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = timestamp()
    out_path = out_dir / f"generate_{args.kv_mode}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({k: v for k, v in result.items()
                   if k != "generated_token_ids"}, f, indent=2)
    print(f"\nSaved to: {out_path}")


def run_compare(args):
    """Run both modes with fresh model load per mode, print capacity table."""
    device = get_device()
    gpu_info = get_gpu_info()

    print("ProtGPT2 v5 — Capacity Comparison")
    print(f"GPU: {gpu_info['name']} ({gpu_info['total_mb']:.0f} MB)")
    print("\nComparing FP16 baseline vs INT8-Triton v5")
    print("Each mode uses a fresh model load for clean memory measurement.\n")

    results = {}
    labels = {"fp16": "FP16 baseline", "int8-triton": "INT8-Triton v5"}
    for i, kv_mode in enumerate(["fp16", "int8-triton"], 1):
        print(f"  [{i}/2] {labels[kv_mode]}...")
        full_cleanup()
        model, tokenizer, metadata = load_model_and_tokenizer(
            "protgpt2", device=str(device), dtype=torch.float16
        )
        input_ids = tokenize_prompt(tokenizer, args)

        # Warmup
        warmup_ids = prepare_prompt(tokenizer, prompt_token_length=16).to(device)
        with torch.no_grad():
            out = model(warmup_ids, use_cache=True)
            _ = model(out.logits[:, -1, :].argmax(-1, keepdim=True),
                      past_key_values=out.past_key_values, use_cache=True)
        del out, warmup_ids
        full_cleanup()

        r = generate(
            model, tokenizer, input_ids, args.max_new_tokens,
            kv_mode=kv_mode, model_name="protgpt2",
            chunk_size=args.chunk_size,
        )
        print(f"  {format_generation_summary(r)}")
        results[kv_mode] = r

        del model, tokenizer
        full_cleanup()

    # Capacity table
    max_pos = 1024  # ProtGPT2
    print(f"\n{format_capacity_table(results['fp16'], results['int8-triton'], gpu_info, max_pos)}")

    # Save raw JSON
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = timestamp()
    out_path = out_dir / f"capacity_compare_{ts}.json"
    save_data = {
        "gpu": gpu_info,
        "baseline": {k: v for k, v in results["fp16"].items()
                     if k != "generated_token_ids"},
        "int8_triton": {k: v for k, v in results["int8-triton"].items()
                        if k != "generated_token_ids"},
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved to: {out_path}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate protein sequences with ProtGPT2 (FP16 or INT8-Triton)")

    # Prompt input (mutually exclusive)
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", type=str, default=None,
                              help="Direct protein sequence input (primary mode)")
    prompt_group.add_argument("--prompt-file", type=str, default=None,
                              help="Read prompt from file (first line)")
    prompt_group.add_argument("--prompt-token-length", type=int, default=None,
                              help="Benchmark fallback: synthetic prompt of N tokens")

    parser.add_argument("--kv-mode", choices=["fp16", "int8-triton"], default="fp16",
                        help="KV cache mode (default: fp16)")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Number of new tokens to generate (default: 256)")
    parser.add_argument("--chunk-size", type=int, default=64,
                        help="INT8 KV block size (default: 64)")
    parser.add_argument("--compare", action="store_true",
                        help="Run both modes with fresh model loads, print capacity table")
    parser.add_argument("--output-dir", type=str, default="results/raw",
                        help="Output directory for raw JSON (default: results/raw)")

    args = parser.parse_args()

    if args.compare:
        run_compare(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
