#!/usr/bin/env python3
"""Phase 2: INT8 weight quantization four-way comparison.

Configs:
  A. FP16 weights + FP16 KV       (baseline)
  C. FP16 weights + INT8 KV       (v5 path)
  E. INT8 weights + FP16 KV       (bnb-int8 weight-only)
  F. INT8 weights + INT8 KV       (stacked: bnb-int8 + INT8 KV Triton)

Each config uses a fresh model load for clean memory measurement.
NF4 results are compared by reference from Phase 1 raw JSON (not rerun).

Usage:
    python scripts/eval_weight_quant_phase2.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from bio_inference_bench.eval_helpers import (
    full_cleanup,
    run_config_with_sanity,
    sanity,
    strip,
)
from bio_inference_bench.profiler import get_gpu_info
from bio_inference_bench.utils import get_device, timestamp

logger = logging.getLogger(__name__)

CONFIGS = [
    {"prompt": 32, "max_new": 256, "label": "decode_heavy"},
    {"prompt": 256, "max_new": 512, "label": "long_decode"},
]

SANITY_STEPS_SHORT = 64
SANITY_STEPS_LONG = 256


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("=" * 60)
    print("  Weight Quantization Phase 2: INT8 Four-Way Comparison")
    print("=" * 60)

    device = get_device()
    gpu_info = get_gpu_info()
    print(f"GPU: {gpu_info['name']} ({gpu_info['total_mb']:.0f} MB)\n")

    paths = [
        ("A_fp16_fp16",    "fp16",     "fp16"),
        ("C_fp16_int8",    "fp16",     "int8-triton"),
        ("E_int8w_fp16",   "bnb-int8", "fp16"),
        ("F_int8w_int8",   "bnb-int8", "int8-triton"),
    ]

    all_results = []

    for cfg in CONFIGS:
        ptl, mnt, label = cfg["prompt"], cfg["max_new"], cfg["label"]
        sanity_steps = SANITY_STEPS_LONG if label == "long_decode" else SANITY_STEPS_SHORT

        print(f"{'━' * 60}")
        print(f"  Config: {label} (prompt={ptl}, max_new={mnt})")
        print(f"{'━' * 60}")

        results = {}
        for path_label, wq, kvm in paths:
            print(f"  [{path_label}] Loading (weights={wq}, kv={kvm})...", end="", flush=True)
            r = run_config_with_sanity(device, wq, kvm, ptl, mnt, sanity_steps)
            print(f" {r['decode_tokens_per_sec']} tok/s, "
                  f"weight={r['weight_memory_mb']:.0f} MB, "
                  f"peak={r['overall_peak_allocated_mb']:.0f} MB, "
                  f"decode_growth={r['decode_phase_growth_mb']:.1f} MB")
            results[path_label] = r

        # Sanity: compare all against A (FP16 baseline)
        baseline = results["A_fp16_fp16"]
        san = {}
        for key in ["C_fp16_int8", "E_int8w_fp16", "F_int8w_int8"]:
            san_short = sanity(baseline, results[key], SANITY_STEPS_SHORT)
            san_long = sanity(baseline, results[key], sanity_steps)
            san[key] = {"short": san_short, "long": san_long}
            print(f"  {key} vs baseline ({SANITY_STEPS_SHORT}): "
                  f"{san_short['token_agreement_pct']}% token, "
                  f"cos={san_short['avg_logit_cosine_similarity']:.6f}")
            if sanity_steps > SANITY_STEPS_SHORT:
                print(f"  {key} vs baseline ({sanity_steps}): "
                      f"{san_long['token_agreement_pct']}% token, "
                      f"cos={san_long['avg_logit_cosine_similarity']:.6f}")

        comparison = {
            "label": label,
            "prompt_token_length": ptl,
            "max_new_tokens": mnt,
            "gpu_info": gpu_info,
        }
        for path_label in [p[0] for p in paths]:
            comparison[path_label] = strip(results[path_label])
        for key in san:
            comparison[f"sanity_{key}_short"] = san[key]["short"]
            comparison[f"sanity_{key}_long"] = san[key]["long"]

        all_results.append(comparison)

    ts = timestamp()
    raw_path = Path("results/raw") / f"weight_quant_phase2_{ts}.json"
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw saved to: {raw_path}")
    print(f"\n{'=' * 60}")
    print(f"  Phase 2 evaluation complete")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
