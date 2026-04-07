"""Results formatting, console output, and JSON export.

All output maintains strict separation between:
- Theoretical KV cache (formula-based)
- Observed peak allocated (CUDA runtime)
- Observed peak reserved (CUDA runtime)

These are NEVER conflated or subtracted from each other.
"""

from __future__ import annotations

import json
from pathlib import Path

from bio_inference_bench.generation import GenerationResult
from bio_inference_bench.models import ModelMetadata
from bio_inference_bench.utils import timestamp


def print_metadata_table(metadata: ModelMetadata) -> None:
    """Print formatted model metadata to console."""
    print(f"\n{'=' * 60}")
    print(f"  Model: {metadata.name}")
    print(f"{'=' * 60}")
    rows = [
        ("HF Path (loaded)", metadata.hf_path_loaded or "FAILED"),
        ("HF Paths Attempted", ", ".join(metadata.hf_path_attempted)),
        ("Architecture", metadata.architecture),
        ("Parameters", f"{metadata.param_count_str} ({metadata.param_count:,})"),
        ("Dtype", metadata.dtype),
        ("Weight Memory", f"{metadata.weight_memory_mb:.2f} MB"),
        ("Hidden Size", str(metadata.hidden_size)),
        ("Num Layers", str(metadata.num_layers)),
        ("Num Attention Heads", str(metadata.num_attention_heads)),
        ("Num KV Heads", str(metadata.num_kv_heads)),
        ("Head Dim", str(metadata.head_dim)),
        ("Max Position Embeddings", str(metadata.max_position_embeddings)),
        ("Vocab Size", str(metadata.vocab_size)),
    ]
    max_label = max(len(r[0]) for r in rows)
    for label, value in rows:
        print(f"  {label:<{max_label}}  {value}")
    if metadata.warnings:
        print(f"\n  Warnings:")
        for w in metadata.warnings:
            print(f"    - {w}")
    print()


def print_benchmark_result(result: GenerationResult, path_label: str = "") -> None:
    """Print formatted benchmark result to console."""
    label = path_label or result.method
    print(f"\n{'─' * 60}")
    print(f"  {label.upper()} — {result.model_name}")
    print(f"{'─' * 60}")

    if result.error:
        print(f"  ERROR: {result.error}")
        if result.warnings:
            for w in result.warnings:
                print(f"    - {w}")
        return

    print(f"  Prompt token length:      {result.prompt_token_length}")
    print(f"  Max new tokens:           {result.max_new_tokens}")
    print(f"  Actual new tokens:        {result.actual_new_tokens}")
    print(f"  Total sequence length:    {result.total_seq_length}")

    print(f"\n  Timing:")
    if result.prefill_time_ms is not None:
        print(f"    Prefill:                {result.prefill_time_ms:.2f} ms")
    if result.decode_time_ms is not None:
        print(f"    Decode:                 {result.decode_time_ms:.2f} ms")
    print(f"    Total:                  {result.total_time_ms:.2f} ms")
    if result.decode_tokens_per_sec > 0:
        print(f"    Decode tokens/sec:      {result.decode_tokens_per_sec:.2f}")
    print(f"    End-to-end tokens/sec:  {result.end_to_end_tokens_per_sec:.2f}")

    print(f"\n  Memory (CUDA observations — NOT KV cache):")
    if result.memory_after_load:
        print(f"    After load allocated:   {result.memory_after_load['allocated_mb']:.2f} MB")
        print(f"    After load reserved:    {result.memory_after_load['reserved_mb']:.2f} MB")
    if result.memory_after_prefill:
        print(f"    After prefill alloc:    {result.memory_after_prefill['allocated_mb']:.2f} MB")
    print(f"    Observed peak allocated:{result.observed_peak_allocated_mb:.2f} MB")
    print(f"    Observed peak reserved: {result.observed_peak_reserved_mb:.2f} MB")

    if result.theoretical_kv_cache:
        kv = result.theoretical_kv_cache
        seq_note = f" at seq_len={kv.get('seq_len_used', '?')}" if kv.get("seq_len_used") else ""
        print(f"\n  Theoretical KV Cache (formula-based{seq_note}):")
        print(f"    Total:                  {kv['total_mb']:.2f} MB")
        print(f"    Per token:              {kv['per_token_mb']:.4f} MB")
        print(f"    As % of weights:        {kv['as_pct_of_weights']:.2f}%")
        print(f"    Growth per 100 tokens:  {kv['growth_per_100_tokens_mb']:.2f} MB")

    if result.warnings:
        print(f"\n  Warnings:")
        for w in result.warnings:
            print(f"    - {w}")
    print()


def save_result_json(result: dict, output_dir: Path, prefix: str) -> Path:
    """Save benchmark results as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{prefix}_{timestamp()}.json"
    path = output_dir / filename

    def _serialize(obj):
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=_serialize)
    return path


def format_comparison_table(all_results: list[dict]) -> str:
    """Format a side-by-side comparison table from primary path results."""
    lines = []
    lines.append(f"\n{'=' * 80}")
    lines.append("  SMOKE BENCHMARK COMPARISON (harness validation — NOT model ranking)")
    lines.append(f"{'=' * 80}")

    header = (
        f"  {'Model':<20} {'Params':<10} {'Weight MB':<12} "
        f"{'Prefill ms':<12} {'Decode ms':<12} {'Tok/s':<10} "
        f"{'TheoKV MB':<12} {'PeakAlloc MB':<14} {'PeakRes MB':<12}"
    )
    lines.append(header)
    lines.append(f"  {'─' * 76}")

    for entry in all_results:
        meta = entry["metadata"]
        pr = entry["primary_result"]

        if pr.error:
            lines.append(f"  {meta.name:<20} FAILED: {pr.error}")
            continue

        kv_mb = pr.theoretical_kv_cache["total_mb"] if pr.theoretical_kv_cache else 0.0
        lines.append(
            f"  {meta.name:<20} {meta.param_count_str:<10} {meta.weight_memory_mb:<12.2f} "
            f"{pr.prefill_time_ms or 0:<12.2f} {pr.decode_time_ms or 0:<12.2f} "
            f"{pr.decode_tokens_per_sec:<10.2f} "
            f"{kv_mb:<12.2f} {pr.observed_peak_allocated_mb:<14.2f} "
            f"{pr.observed_peak_reserved_mb:<12.2f}"
        )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# INT8 KV capacity reporting (productization)
# ---------------------------------------------------------------------------

def format_generation_summary(result: dict) -> str:
    """Single-line generation summary.

    Example: [int8-triton] 256 tokens in 3.4s (74 tok/s) | peak 1512 MB | decode growth 18 MB
    """
    method = result.get("method", "unknown")
    actual = result.get("actual_new_tokens", 0)
    total_s = result.get("total_ms", 0) / 1000
    tps = result.get("decode_tokens_per_sec", 0)
    peak = result.get("overall_peak_allocated_mb", result.get("decode_peak_allocated_mb", 0))
    decode_growth = result.get("decode_phase_growth_mb", 0)
    return (f"[{method}] {actual} tokens in {total_s:.1f}s ({tps:.0f} tok/s) | "
            f"peak {peak:.0f} MB | decode growth {decode_growth:.1f} MB")


def format_capacity_table(
    baseline_result: dict,
    int8_result: dict,
    gpu_info: dict,
    max_position_embeddings: int = 1024,
) -> str:
    """Format a capacity comparison table for user-facing output.

    The table has four sections:
    1. Measured run summary (from actual generation)
    2. Model context limit (feasible new tokens capped by max_position_embeddings)
    3. VRAM required for feasible context (capped decode growth)
    4. Slope-based VRAM projection (uncapped, labeled as theoretical headroom)
    """
    gpu_name = gpu_info.get("name", "Unknown GPU")
    gpu_mb = gpu_info.get("total_mb", 0)

    prompt_len = int8_result.get("prompt_token_length", 0)
    actual_new = int8_result.get("actual_new_tokens", 0)

    b_tps = baseline_result.get("decode_tokens_per_sec", 0)
    i_tps = int8_result.get("decode_tokens_per_sec", 0)
    speed_pct = round(i_tps / b_tps * 100) if b_tps > 0 else 0

    b_growth_per_tok = baseline_result.get("decode_growth_per_token_mb", 0)
    i_growth_per_tok = int8_result.get("decode_growth_per_token_mb", 0)

    feasible_new = max_position_embeddings - prompt_len

    # --- Section 1: Measured run ---
    lines = [
        f"ProtGPT2 INT8 KV Capacity Report — {gpu_name} ({gpu_mb:.0f} MB)",
        "=" * 60,
        f"{'':30s} {'FP16 Baseline':>15s}  {'INT8-Triton v5':>15s}",
        "-" * 60,
        f"{'Decode growth/token':30s} {b_growth_per_tok:>12.4f} MB  {i_growth_per_tok:>12.4f} MB",
        f"{'Decode speed':30s} {b_tps:>11.0f} tok/s  {i_tps:>8.0f} tok/s ({speed_pct}%)",
    ]

    ci = int8_result.get("cache_info", {})
    if ci and ci.get("total_cache_mb", 0) > 0:
        compression = ci.get("fp16_equivalent_mb", 0) / ci["total_cache_mb"]
        lines.append(f"{'Cache compression ratio':30s} {'1.00x':>15s}  {compression:>14.2f}x")

    lines.append("")
    lines.append(f"Measured: {actual_new} new tokens from {prompt_len}-token prompt")

    # --- Section 2: Model context limit ---
    lines.append("")
    lines.append(f"Model context limit: {max_position_embeddings} tokens (max_position_embeddings)")
    lines.append(f"Feasible new tokens for this prompt: {feasible_new} (= {max_position_embeddings} - {prompt_len})")

    # --- Section 3: VRAM for feasible context (capped) ---
    if b_growth_per_tok > 0 and i_growth_per_tok > 0:
        b_feasible_mb = feasible_new * b_growth_per_tok
        i_feasible_mb = feasible_new * i_growth_per_tok
        lines.append("")
        lines.append(f"Decode VRAM for full context ({feasible_new} new tokens):")
        lines.append(f"  FP16:         ~{b_feasible_mb:>6.1f} MB")
        lines.append(f"  INT8-Triton:  ~{i_feasible_mb:>6.1f} MB")
        if b_feasible_mb > 0:
            lines.append(f"  Savings:      ~{b_feasible_mb - i_feasible_mb:.1f} MB "
                         f"({(1 - i_feasible_mb / b_feasible_mb) * 100:.0f}%)")

    # --- Section 4: Slope-based VRAM projection (uncapped, theoretical) ---
    lines.append("")
    lines.append("Slope-based VRAM projection (theoretical headroom, NOT achievable on ProtGPT2):")
    if b_growth_per_tok > 0 and i_growth_per_tok > 0:
        for free_gb in [1, 2, 4]:
            free_mb = free_gb * 1024
            b_est = int(free_mb / b_growth_per_tok)
            i_est = int(free_mb / i_growth_per_tok)
            lines.append(
                f"  {free_gb} GB free VRAM       "
                f"~{b_est:>8,} tokens    ~{i_est:>8,} tokens"
            )
        lines.append(f"  These are uncapped slope extrapolations showing how the decode growth")
        lines.append(f"  rate scales with VRAM. ProtGPT2 is limited to {max_position_embeddings} total positions.")
    else:
        lines.append("  (insufficient data for projection)")

    lines.append("=" * 60)
    return "\n".join(lines)
