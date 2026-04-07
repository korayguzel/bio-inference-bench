"""Autoregressive generation with prefill/decode separation.

Two generation paths:
- PRIMARY: run_manual_prefill_decode() — explicit prefill + token-by-token decode
  with past_key_values. Authoritative for bottleneck analysis.
- SECONDARY: run_generate_api() — HuggingFace model.generate() wrapper.
  End-to-end baseline for cross-validation only.

The primary path is what the mentor packet draws conclusions from.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from bio_inference_bench.kv_estimator import KVEstimate, dtype_to_bytes, estimate_kv_cache
from bio_inference_bench.models import ModelMetadata
from bio_inference_bench.profiler import MemorySnapshot, reset_memory_tracking, take_snapshot

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of a single generation run."""

    model_name: str
    prompt_token_length: int
    max_new_tokens: int
    actual_new_tokens: int = 0
    total_seq_length: int = 0
    batch_size: int = 1
    use_cache: bool = True
    method: str = ""  # "manual_prefill_decode" or "generate_api"
    prefill_time_ms: float | None = None
    decode_time_ms: float | None = None
    total_time_ms: float = 0.0
    decode_tokens_per_sec: float = 0.0
    end_to_end_tokens_per_sec: float = 0.0
    per_step_decode_times_ms: list[float] = field(default_factory=list)
    memory_after_load: dict | None = None
    memory_after_prefill: dict | None = None
    memory_peak: dict | None = None
    theoretical_kv_cache: dict | None = None
    observed_peak_allocated_mb: float = 0.0
    observed_peak_reserved_mb: float = 0.0
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "prompt_token_length": self.prompt_token_length,
            "max_new_tokens": self.max_new_tokens,
            "actual_new_tokens": self.actual_new_tokens,
            "total_seq_length": self.total_seq_length,
            "batch_size": self.batch_size,
            "use_cache": self.use_cache,
            "method": self.method,
            "prefill_time_ms": self.prefill_time_ms,
            "decode_time_ms": self.decode_time_ms,
            "total_time_ms": round(self.total_time_ms, 2),
            "decode_tokens_per_sec": round(self.decode_tokens_per_sec, 2),
            "end_to_end_tokens_per_sec": round(self.end_to_end_tokens_per_sec, 2),
            "per_step_decode_times_ms": [round(t, 3) for t in self.per_step_decode_times_ms],
            "memory_after_load": self.memory_after_load,
            "memory_after_prefill": self.memory_after_prefill,
            "memory_peak": self.memory_peak,
            "theoretical_kv_cache": self.theoretical_kv_cache,
            "observed_peak_allocated_mb": round(self.observed_peak_allocated_mb, 2),
            "observed_peak_reserved_mb": round(self.observed_peak_reserved_mb, 2),
            "error": self.error,
            "warnings": self.warnings,
        }


def run_manual_prefill_decode(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    model_name: str = "",
) -> GenerationResult:
    """PRIMARY profiling path: explicit prefill + token-by-token decode.

    This is the authoritative generation path for bottleneck analysis.
    It separates prefill (single forward pass over prompt) from decode
    (iterative token-by-token generation with past_key_values).
    """
    prompt_token_length = input_ids.shape[1]
    result = GenerationResult(
        model_name=model_name,
        prompt_token_length=prompt_token_length,
        max_new_tokens=max_new_tokens,
        method="manual_prefill_decode",
    )

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    try:
        # --- Memory snapshot after model load ---
        reset_memory_tracking()
        result.memory_after_load = take_snapshot().to_dict()

        # --- PREFILL PHASE ---
        torch.cuda.synchronize()
        t_prefill_start = time.perf_counter()

        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)

        torch.cuda.synchronize()
        prefill_time_s = time.perf_counter() - t_prefill_start
        result.prefill_time_ms = prefill_time_s * 1000

        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        result.memory_after_prefill = take_snapshot().to_dict()

        # --- DECODE PHASE ---
        generated_tokens = [next_token]
        step_times: list[float] = []

        with torch.no_grad():
            for _ in range(max_new_tokens - 1):
                torch.cuda.synchronize()
                t_step = time.perf_counter()

                outputs = model(
                    next_token, past_key_values=past_key_values, use_cache=True
                )

                torch.cuda.synchronize()
                step_times.append((time.perf_counter() - t_step) * 1000)

                past_key_values = outputs.past_key_values
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_tokens.append(next_token)

        # --- Collect results ---
        peak_snap = take_snapshot()
        result.memory_peak = peak_snap.to_dict()
        result.observed_peak_allocated_mb = peak_snap.max_allocated_mb
        result.observed_peak_reserved_mb = peak_snap.max_reserved_mb

        result.actual_new_tokens = len(generated_tokens)
        result.total_seq_length = prompt_token_length + result.actual_new_tokens
        result.per_step_decode_times_ms = step_times
        result.decode_time_ms = sum(step_times) + (
            # First decode token time was part of prefill logit selection,
            # but the remaining (max_new_tokens - 1) steps are timed above.
            # Total decode includes all step times.
            0.0
        )
        result.total_time_ms = result.prefill_time_ms + result.decode_time_ms

        if result.decode_time_ms > 0:
            result.decode_tokens_per_sec = (
                (result.actual_new_tokens - 1) / (result.decode_time_ms / 1000)
            )
        if result.total_time_ms > 0:
            result.end_to_end_tokens_per_sec = (
                result.actual_new_tokens / (result.total_time_ms / 1000)
            )

    except torch.cuda.OutOfMemoryError as e:
        result.error = f"CUDA OOM: {e}"
        result.warnings.append("OOM during manual prefill/decode")
        logger.error(f"OOM in manual_prefill_decode: {e}")
    except Exception as e:
        result.error = f"Error: {e}"
        logger.error(f"Error in manual_prefill_decode: {e}")
    finally:
        # Cleanup
        for name in ("past_key_values", "outputs", "next_token", "generated_tokens"):
            if name in dir():
                pass  # locals cleanup handled by scope exit
        reset_memory_tracking()

    return result


def run_generate_api(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    use_cache: bool = True,
    model_name: str = "",
) -> GenerationResult:
    """SECONDARY path: HuggingFace model.generate() end-to-end baseline.

    Used for cross-validation of timing, not for bottleneck analysis.
    Does not separate prefill from decode.
    """
    prompt_token_length = input_ids.shape[1]
    result = GenerationResult(
        model_name=model_name,
        prompt_token_length=prompt_token_length,
        max_new_tokens=max_new_tokens,
        use_cache=use_cache,
        method="generate_api",
    )

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    try:
        reset_memory_tracking()
        result.memory_after_load = take_snapshot().to_dict()

        torch.cuda.synchronize()
        t_start = time.perf_counter()

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=use_cache,
            )

        torch.cuda.synchronize()
        total_time_s = time.perf_counter() - t_start

        peak_snap = take_snapshot()
        result.memory_peak = peak_snap.to_dict()
        result.observed_peak_allocated_mb = peak_snap.max_allocated_mb
        result.observed_peak_reserved_mb = peak_snap.max_reserved_mb

        result.actual_new_tokens = output_ids.shape[1] - prompt_token_length
        result.total_seq_length = output_ids.shape[1]
        result.total_time_ms = total_time_s * 1000
        if total_time_s > 0:
            result.end_to_end_tokens_per_sec = result.actual_new_tokens / total_time_s

    except torch.cuda.OutOfMemoryError as e:
        result.error = f"CUDA OOM: {e}"
        result.warnings.append("OOM during generate API")
        logger.error(f"OOM in generate_api: {e}")
    except Exception as e:
        result.error = f"Error: {e}"
        logger.error(f"Error in generate_api: {e}")
    finally:
        reset_memory_tracking()

    return result


def _compute_kv_for_result(
    result: GenerationResult, metadata: ModelMetadata
) -> dict:
    """Compute theoretical KV cache based on actual generated length for a result."""
    if result.error or result.total_seq_length == 0:
        return {"total_mb": 0, "per_token_mb": 0, "as_pct_of_weights": 0,
                "growth_per_100_tokens_mb": 0, "seq_len_used": 0}
    dtype_bytes = dtype_to_bytes(torch.float16) if metadata.dtype == "float16" else 4
    kv_est = estimate_kv_cache(
        num_layers=metadata.num_layers,
        batch_size=1,
        seq_len=result.total_seq_length,
        num_kv_heads=metadata.num_kv_heads,
        head_dim=metadata.head_dim,
        dtype_bytes=dtype_bytes,
    )
    if metadata.weight_memory_mb > 0:
        kv_est.as_pct_of_weights = (kv_est.total_mb / metadata.weight_memory_mb) * 100
    return {
        "total_mb": round(kv_est.total_mb, 2),
        "per_token_mb": round(kv_est.per_token_mb, 4),
        "as_pct_of_weights": round(kv_est.as_pct_of_weights, 2),
        "growth_per_100_tokens_mb": round(kv_est.growth_per_100_tokens_mb, 2),
        "seq_len_used": result.total_seq_length,
    }


def run_benchmark(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    use_cache: bool,
    metadata: ModelMetadata,
) -> dict:
    """Run both generation paths and compute theoretical KV cache.

    Theoretical KV is computed per-path using the actual final sequence length,
    not the configured max_new_tokens. An upper-bound estimate (at max_new_tokens)
    is also included for reference.

    Returns a dict with:
        primary_result: GenerationResult from manual prefill/decode
        secondary_result: GenerationResult from generate API
        theoretical_kv_upper_bound: KVEstimate at configured max sequence length
        metadata: ModelMetadata
    """
    model_name = metadata.name
    prompt_token_length = input_ids.shape[1]
    configured_max_seq = prompt_token_length + max_new_tokens
    dtype_bytes = dtype_to_bytes(torch.float16) if metadata.dtype == "float16" else 4

    # Upper-bound KV estimate (at configured max_new_tokens)
    kv_upper = estimate_kv_cache(
        num_layers=metadata.num_layers,
        batch_size=1,
        seq_len=configured_max_seq,
        num_kv_heads=metadata.num_kv_heads,
        head_dim=metadata.head_dim,
        dtype_bytes=dtype_bytes,
    )
    if metadata.weight_memory_mb > 0:
        kv_upper.as_pct_of_weights = (kv_upper.total_mb / metadata.weight_memory_mb) * 100

    # PRIMARY path: manual prefill/decode
    logger.info(f"Running PRIMARY path (manual prefill/decode) for {model_name}")
    primary_result = run_manual_prefill_decode(
        model, input_ids, max_new_tokens, model_name=model_name
    )
    primary_result.theoretical_kv_cache = _compute_kv_for_result(primary_result, metadata)

    # SECONDARY path: generate API
    logger.info(f"Running SECONDARY path (generate API) for {model_name}")
    secondary_result = run_generate_api(
        model, tokenizer, input_ids, max_new_tokens,
        use_cache=use_cache, model_name=model_name,
    )
    secondary_result.theoretical_kv_cache = _compute_kv_for_result(secondary_result, metadata)

    return {
        "primary_result": primary_result,
        "secondary_result": secondary_result,
        "theoretical_kv_upper_bound": {
            "total_mb": round(kv_upper.total_mb, 2),
            "total_bytes": kv_upper.total_bytes,
            "per_token_mb": round(kv_upper.per_token_mb, 4),
            "as_pct_of_weights": round(kv_upper.as_pct_of_weights, 2),
            "growth_per_100_tokens_mb": round(kv_upper.growth_per_100_tokens_mb, 2),
            "seq_len_used": configured_max_seq,
            "label": "upper_bound_at_configured_max",
        },
        "metadata": metadata,
    }
