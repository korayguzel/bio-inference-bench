#!/usr/bin/env python3
"""Inspect model metadata and print theoretical KV cache table.

Usage:
    python scripts/inspect_model.py --model protgpt2
    python scripts/inspect_model.py --model progen2-small
    python scripts/inspect_model.py --model all
"""

from __future__ import annotations

import argparse
import logging
import sys

from bio_inference_bench.kv_estimator import dtype_to_bytes, estimate_kv_table
from bio_inference_bench.models import load_model_and_tokenizer
from bio_inference_bench.progen2_compat import patch_progen2_model
from bio_inference_bench.report import print_metadata_table
from bio_inference_bench.utils import MODEL_REGISTRY, get_device

import torch


def inspect_one(model_name: str) -> None:
    """Load and inspect a single model."""
    device = get_device()
    print(f"\nLoading {model_name}...")

    model, tokenizer, metadata = load_model_and_tokenizer(
        model_name, device=str(device), dtype=torch.float16
    )

    print_metadata_table(metadata)

    if model is None:
        print(f"  Model failed to load. See warnings above.\n")
        return

    # Apply compatibility patches if needed
    patches = patch_progen2_model(model)
    for p in patches:
        print(f"  PATCHED: {p}")

    # Theoretical KV cache table
    if metadata.num_kv_heads > 0 and metadata.head_dim > 0:
        dtype_bytes = dtype_to_bytes(torch.float16)
        kv_df = estimate_kv_table(
            num_layers=metadata.num_layers,
            num_kv_heads=metadata.num_kv_heads,
            head_dim=metadata.head_dim,
            dtype_bytes=dtype_bytes,
            seq_lengths=[64, 128, 256, 512, 1024],
            batch_sizes=[1],
        )
        print("  Theoretical KV Cache (formula-based, batch_size=1):")
        print(kv_df.to_string(index=False))
        print()

    # Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect model metadata")
    choices = list(MODEL_REGISTRY.keys()) + ["all"]
    parser.add_argument("--model", required=True, choices=choices, help="Model to inspect")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.model == "all":
        for name in MODEL_REGISTRY:
            inspect_one(name)
    else:
        inspect_one(args.model)


if __name__ == "__main__":
    main()
