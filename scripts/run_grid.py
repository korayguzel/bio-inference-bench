#!/usr/bin/env python3
"""Execute the constrained benchmark grid.

Runs each (model, prompt_token_length, max_new_tokens) configuration 3 times
with 1 warm-up run per model (discarded). Reports medians. Uses the canonical
reporting path from bio_inference_bench.grid_report.

Usage:
    python scripts/run_grid.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from bio_inference_bench.generation import GenerationResult, run_benchmark
from bio_inference_bench.grid_report import aggregate_live_runs, generate_grid_report
from bio_inference_bench.models import (
    ModelMetadata,
    _extract_metadata,
    _set_eval_mode,
    _try_load_tokenizer,
    prepare_prompt,
)
from bio_inference_bench.progen2_compat import patch_progen2_model
from bio_inference_bench.profiler import get_gpu_info, reset_memory_tracking
from bio_inference_bench.report import save_result_json
from bio_inference_bench.utils import MODEL_REGISTRY, get_device, timestamp

logger = logging.getLogger(__name__)

# --- Grid parameters (per approved plan) ---
MODELS = ["protgpt2", "progen2-small"]
PROMPT_TOKEN_LENGTHS = [16, 32, 64, 128, 256]
MAX_NEW_TOKENS = [32, 64, 128, 256, 512]
SEQ_CEILING = 900
REPEATS = 3
WARMUP_PROMPT = 16
WARMUP_MAX_NEW = 8


def build_grid() -> list[tuple[int, int]]:
    """Build valid (prompt, max_new) pairs under the sequence ceiling."""
    pairs = []
    for p in PROMPT_TOKEN_LENGTHS:
        for m in MAX_NEW_TOKENS:
            if p + m <= SEQ_CEILING:
                pairs.append((p, m))
    return pairs


def load_and_patch(
    model_name: str, device: torch.device, dtype: torch.dtype = torch.float16
) -> tuple | None:
    """Load a model with candidate fallback and apply patches."""
    entry = MODEL_REGISTRY[model_name]
    for candidate in entry["candidates"]:
        hf_path = candidate["hf_path"]
        trust_remote_code = candidate.get("trust_remote_code", False)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                hf_path, trust_remote_code=trust_remote_code, torch_dtype=dtype,
            ).to(str(device))
            _set_eval_mode(model)
        except Exception:
            continue
        tokenizer, _ = _try_load_tokenizer(hf_path, trust_remote_code)
        if tokenizer is None:
            del model; torch.cuda.empty_cache()
            continue
        patches = patch_progen2_model(model)
        metadata = _extract_metadata(model, hf_path, model_name, dtype)
        metadata.hf_path_loaded = hf_path
        for p in patches:
            metadata.warnings.append(f"Applied patch: {p}")
        return model, tokenizer, metadata
    return None


def run_warmup(model, tokenizer, metadata):
    """Single discarded warm-up run."""
    input_ids = prepare_prompt(tokenizer, prompt_token_length=WARMUP_PROMPT)
    run_benchmark(model, tokenizer, input_ids, WARMUP_MAX_NEW, True, metadata)
    reset_memory_tracking()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("  bio-inference-bench — Grid Benchmark")
    print("  This is a profiling benchmark, NOT an optimization project.")
    print("=" * 60)

    device = get_device()
    gpu_info = get_gpu_info()
    grid_pairs = build_grid()
    total_configs = len(MODELS) * len(grid_pairs)

    print(f"\nGPU: {gpu_info['name']} ({gpu_info['total_mb']:.0f} MB)")
    print(f"Grid: {len(MODELS)} models x {len(grid_pairs)} configs = {total_configs} total")
    print(f"Repeats: {REPEATS} per config, 1 warm-up per model")
    print(f"Sequence ceiling: {SEQ_CEILING}")
    print()

    all_summaries: list[dict] = []
    completed = 0
    failed = 0

    for model_name in MODELS:
        print(f"\n{'━' * 60}")
        print(f"  Model: {model_name}")
        print(f"{'━' * 60}")

        result = load_and_patch(model_name, device)
        if result is None:
            print(f"  FAILED: Could not load {model_name}. Skipping.")
            for ptl, mnt in grid_pairs:
                all_summaries.append({
                    "model_name": model_name, "prompt_token_length": ptl,
                    "max_new_tokens": mnt, "status": "load_failed",
                    "error": "Model load failed",
                })
                failed += 1
            continue

        model, tokenizer, metadata = result
        print(f"  Loaded: {metadata.hf_path_loaded} ({metadata.param_count_str})")

        # Warm-up
        print(f"  Running warm-up...")
        run_warmup(model, tokenizer, metadata)
        print(f"  Warm-up complete.")

        for i, (ptl, mnt) in enumerate(grid_pairs):
            print(f"  [{i+1}/{len(grid_pairs)}] prompt={ptl}, max_new={mnt}", end="", flush=True)

            runs: list[dict] = []
            for rep in range(REPEATS):
                try:
                    input_ids = prepare_prompt(tokenizer, prompt_token_length=ptl)
                    result_dict = run_benchmark(
                        model, tokenizer, input_ids, mnt, True, metadata,
                    )
                    runs.append(result_dict)

                    # Save per-run JSON
                    save_result_json(
                        {
                            "metadata": metadata.to_dict(),
                            "primary_result": result_dict["primary_result"].to_dict(),
                            "secondary_result": result_dict["secondary_result"].to_dict(),
                            "theoretical_kv_upper_bound": result_dict["theoretical_kv_upper_bound"],
                        },
                        Path("results/raw"),
                        prefix=f"grid_{model_name}_{ptl}_{mnt}_r{rep}",
                    )
                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"OOM: {model_name} prompt={ptl} max_new={mnt} rep={rep}")
                    reset_memory_tracking()
                    dummy = GenerationResult(
                        model_name=model_name, prompt_token_length=ptl,
                        max_new_tokens=mnt, method="manual_prefill_decode",
                        error="CUDA OOM",
                    )
                    runs.append({"primary_result": dummy, "secondary_result": dummy,
                                 "theoretical_kv_upper_bound": {}, "metadata": metadata})
                except Exception as e:
                    logger.warning(f"Error: {model_name} prompt={ptl} max_new={mnt}: {e}")
                    dummy = GenerationResult(
                        model_name=model_name, prompt_token_length=ptl,
                        max_new_tokens=mnt, method="manual_prefill_decode",
                        error=str(e),
                    )
                    runs.append({"primary_result": dummy, "secondary_result": dummy,
                                 "theoretical_kv_upper_bound": {}, "metadata": metadata})

            summary = aggregate_live_runs(runs, model_name, ptl, mnt)
            all_summaries.append(summary)

            if summary["status"] == "ok":
                completed += 1
                print(f"  -> {summary['median_decode_tokens_per_sec']} tok/s, "
                      f"peak={summary['observed_peak_allocated_mb']} MB, "
                      f"phase={summary['peak_phase']}")
            else:
                failed += 1
                print(f"  -> FAILED: {summary.get('error', '?')}")

        # Cleanup model
        del model, tokenizer
        torch.cuda.empty_cache()

    # Save summary JSON
    ts = timestamp()
    summary_path = Path("results/summaries") / f"grid_summary_{ts}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nGrid summary saved to: {summary_path}")

    # Generate canonical report
    report_path = Path("results/summaries") / f"grid_report_{ts}.md"
    generate_grid_report(all_summaries, gpu_info, report_path)
    print(f"Grid report saved to: {report_path}")

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"  Grid Complete: {completed} ok, {failed} failed out of {total_configs}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
