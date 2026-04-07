#!/usr/bin/env python3
"""Run the smoke benchmark pair for harness validation.

Runs the canonical smoke config on all registered models.
Produces a comparison table and raw JSON artifacts.

This is harness validation, NOT model comparison.

Fallback semantics: if a candidate loads but fails during generation
(runtime incompatibility), it is recorded as a failed attempt and
the next candidate is tried before concluding the model failed.

Usage:
    python scripts/run_smoke.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from bio_inference_bench.generation import GenerationResult, run_benchmark
from bio_inference_bench.models import (
    ModelMetadata,
    _extract_metadata,
    _set_eval_mode,
    _try_load_tokenizer,
    prepare_prompt,
)
from bio_inference_bench.progen2_compat import patch_progen2_model
from bio_inference_bench.profiler import get_gpu_info
from bio_inference_bench.report import (
    format_comparison_table,
    print_benchmark_result,
    print_metadata_table,
    save_result_json,
)
from bio_inference_bench.utils import MODEL_REGISTRY, SMOKE_CONFIG, get_device

from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)


def _make_failed_entry(
    model_name: str, metadata: ModelMetadata
) -> dict:
    """Create a failed-result entry for the smoke results."""
    error_msg = f"All candidates failed. See warnings: {metadata.warnings}"
    return {
        "primary_result": GenerationResult(
            model_name=model_name,
            prompt_token_length=SMOKE_CONFIG["prompt_token_length"],
            max_new_tokens=SMOKE_CONFIG["max_new_tokens"],
            method="manual_prefill_decode",
            error=error_msg,
        ),
        "secondary_result": GenerationResult(
            model_name=model_name,
            prompt_token_length=SMOKE_CONFIG["prompt_token_length"],
            max_new_tokens=SMOKE_CONFIG["max_new_tokens"],
            method="generate_api",
            error=error_msg,
        ),
        "theoretical_kv_upper_bound": {
            "total_mb": 0, "total_bytes": 0, "per_token_mb": 0,
            "as_pct_of_weights": 0, "growth_per_100_tokens_mb": 0,
            "seq_len_used": 0, "label": "upper_bound_at_configured_max",
        },
        "metadata": metadata,
    }


def try_model_end_to_end(
    model_name: str, device: torch.device, dtype: torch.dtype = torch.float16
) -> dict | None:
    """Try each candidate for a model end-to-end: load + prompt + generation.

    Returns a successful results dict, or None if all candidates fail.
    Populates metadata.warnings with per-candidate failure details.
    """
    entry = MODEL_REGISTRY[model_name]
    candidates = entry["candidates"]
    all_attempted: list[str] = []
    all_warnings: list[str] = []

    for i, candidate in enumerate(candidates):
        hf_path = candidate["hf_path"]
        trust_remote_code = candidate.get("trust_remote_code", False)
        all_attempted.append(hf_path)

        print(f"\n  Candidate {i + 1}/{len(candidates)}: {hf_path}")

        # --- Try loading model ---
        try:
            model = AutoModelForCausalLM.from_pretrained(
                hf_path, trust_remote_code=trust_remote_code, torch_dtype=dtype,
            ).to(str(device))
            _set_eval_mode(model)
        except Exception as e:
            msg = f"Candidate {hf_path}: model load failed — {e}"
            logger.warning(msg)
            all_warnings.append(msg)
            print(f"    LOAD FAILED: {e}")
            continue

        # --- Apply compatibility patches if needed ---
        patches = patch_progen2_model(model)
        for p in patches:
            all_warnings.append(f"Candidate {hf_path}: applied patch — {p}")
            print(f"    PATCHED: {p}")

        # --- Try loading tokenizer ---
        tokenizer, tok_error = _try_load_tokenizer(hf_path, trust_remote_code)
        if tokenizer is None:
            msg = f"Candidate {hf_path}: tokenizer failed — {tok_error}"
            logger.warning(msg)
            all_warnings.append(msg)
            print(f"    TOKENIZER FAILED: {tok_error}")
            del model
            torch.cuda.empty_cache()
            continue

        # --- Extract metadata ---
        metadata = _extract_metadata(model, hf_path, model_name, dtype)
        metadata.hf_path_attempted = all_attempted
        metadata.hf_path_loaded = hf_path
        metadata.warnings.extend(all_warnings)
        if i > 0:
            metadata.warnings.append(f"Loaded from fallback candidate #{i + 1}: {hf_path}")

        print_metadata_table(metadata)

        # --- Try preparing prompt ---
        try:
            input_ids = prepare_prompt(
                tokenizer, prompt_token_length=SMOKE_CONFIG["prompt_token_length"]
            )
        except Exception as e:
            msg = f"Candidate {hf_path}: prompt preparation failed — {e}"
            logger.warning(msg)
            all_warnings.append(msg)
            metadata.warnings.append(msg)
            print(f"    PROMPT FAILED: {e}")
            del model, tokenizer
            torch.cuda.empty_cache()
            continue

        # --- Try running benchmark ---
        print(f"    Running benchmark...")
        results = run_benchmark(
            model=model, tokenizer=tokenizer, input_ids=input_ids,
            max_new_tokens=SMOKE_CONFIG["max_new_tokens"],
            use_cache=SMOKE_CONFIG["use_cache"], metadata=metadata,
        )

        primary = results["primary_result"]
        if primary.error is not None:
            msg = f"Candidate {hf_path}: generation failed — {primary.error}"
            logger.warning(msg)
            all_warnings.append(msg)
            metadata.warnings.append(msg)
            print(f"    GENERATION FAILED: {primary.error}")
            del model, tokenizer
            torch.cuda.empty_cache()
            continue

        # --- Success: this candidate works end-to-end ---
        print(f"    SUCCESS: candidate {hf_path} completed end-to-end")

        # Print results
        print_benchmark_result(primary, path_label="PRIMARY: manual prefill/decode")
        print_benchmark_result(
            results["secondary_result"], path_label="SECONDARY: generate API"
        )

        # Cleanup
        del model, tokenizer
        torch.cuda.empty_cache()
        return results

    # All candidates failed
    meta = ModelMetadata(name=model_name, hf_path=candidates[0]["hf_path"])
    meta.hf_path_attempted = all_attempted
    meta.hf_path_loaded = None
    meta.warnings = all_warnings + [
        f"ALL {len(candidates)} candidates failed end-to-end for '{model_name}'"
    ]
    print_metadata_table(meta)
    return _make_failed_entry(model_name, meta)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("  bio-inference-bench — Smoke Benchmark")
    print("  Purpose: harness validation (NOT model comparison)")
    print("=" * 60)

    device = get_device()
    gpu_info = get_gpu_info()
    print(f"\nGPU: {gpu_info['name']} ({gpu_info['total_mb']:.0f} MB)")
    print(f"Config: prompt_token_length={SMOKE_CONFIG['prompt_token_length']}, "
          f"max_new_tokens={SMOKE_CONFIG['max_new_tokens']}, "
          f"batch_size={SMOKE_CONFIG['batch_size']}, greedy, use_cache=True")
    print()

    all_results: list[dict] = []

    for model_name in MODEL_REGISTRY:
        print(f"\n{'━' * 60}")
        print(f"  Model: {model_name}")
        print(f"{'━' * 60}")

        result = try_model_end_to_end(model_name, device)
        if result is None:
            continue
        all_results.append(result)

        # Always save raw JSON — both successful and failed runs
        meta = result["metadata"]
        save_result_json(
            {
                "metadata": meta.to_dict() if hasattr(meta, "to_dict") else {},
                "primary_result": result["primary_result"].to_dict(),
                "secondary_result": result["secondary_result"].to_dict(),
                "theoretical_kv_upper_bound": result["theoretical_kv_upper_bound"],
            },
            Path("results/raw"),
            prefix=f"smoke_{model_name}",
        )

    # Comparison table
    print(format_comparison_table(all_results))

    print("\nSmoke run complete. Raw artifacts are saved under results/raw/.")


if __name__ == "__main__":
    main()
