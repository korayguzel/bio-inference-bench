"""Grid benchmark aggregation and report generation.

This is the single source of truth for grid report semantics. Both
run_grid.py (live execution) and regenerate_grid_report.py (re-analysis
from saved JSON) use this module.

Key metrics:
- generation_overhead_above_load_mb: primary runtime-growth metric
- overhead_above_weights_mb: coarse audit metric
- peak_phase: whether max_memory_allocated peaked in prefill or decode
- runtime_kv_ratio_pct: theoretical_kv / generation_overhead_above_load
- coarse_kv_ratio_pct: theoretical_kv / overhead_above_weights

Neither ratio is a direct component-wise decomposition.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import torch

from bio_inference_bench.utils import timestamp


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def compute_peak_phase(pr: dict) -> str:
    """Determine whether peak allocated occurred during prefill or decode.

    Compares max_allocated_mb at the after-prefill snapshot vs the final
    peak snapshot. If they match (within 0.5 MB), peak was in prefill.
    """
    prefill_snap = pr.get("memory_after_prefill")
    final_snap = pr.get("memory_peak")
    if not prefill_snap or not final_snap:
        return "unclear"
    prefill_peak = prefill_snap.get("max_allocated_mb", 0)
    final_peak = final_snap.get("max_allocated_mb", 0)
    if abs(final_peak - prefill_peak) < 0.5:
        return "prefill"
    elif final_peak > prefill_peak + 0.5:
        return "decode"
    return "unclear"


def aggregate_group(runs: list[dict]) -> dict:
    """Aggregate repeated runs (from JSON dicts) into a single summary row.

    Expects each run to have keys: primary_result, metadata.
    primary_result fields accessed as dict (already serialized).
    """
    primaries = [r["primary_result"] for r in runs]
    meta = runs[0]["metadata"]
    pr0 = primaries[0]

    successful = [p for p in primaries if p.get("error") is None]
    if not successful:
        return {
            "model_name": pr0["model_name"],
            "prompt_token_length": pr0["prompt_token_length"],
            "max_new_tokens": pr0["max_new_tokens"],
            "status": "all_failed",
            "error": pr0.get("error", "unknown"),
        }

    # Median-time representative run
    totals = [p["total_time_ms"] for p in successful]
    med_val = statistics.median(totals)
    med_idx = min(range(len(totals)), key=lambda i: abs(totals[i] - med_val))
    med = successful[med_idx]

    prefills = [p["prefill_time_ms"] for p in successful if p.get("prefill_time_ms")]
    decodes = [p["decode_time_ms"] for p in successful if p.get("decode_time_ms")]
    decode_tps = [p["decode_tokens_per_sec"] for p in successful if p.get("decode_tokens_per_sec", 0) > 0]

    kv = med.get("theoretical_kv_cache") or {}
    weight_mb = meta["weight_memory_mb"]
    peak_alloc = med.get("observed_peak_allocated_mb", 0)
    load_alloc = (med.get("memory_after_load") or {}).get("allocated_mb", weight_mb)

    overhead_above_weights = peak_alloc - weight_mb
    gen_overhead_above_load = peak_alloc - load_alloc
    kv_mb = kv.get("total_mb", 0)

    peak_phase = compute_peak_phase(med)

    coarse_kv_ratio = (kv_mb / overhead_above_weights * 100) if overhead_above_weights > 0 else 0
    runtime_kv_ratio = (kv_mb / gen_overhead_above_load * 100) if gen_overhead_above_load > 0 else 0

    return {
        "model_name": pr0["model_name"],
        "prompt_token_length": pr0["prompt_token_length"],
        "max_new_tokens": pr0["max_new_tokens"],
        "actual_total_seq_length": med.get("total_seq_length", 0),
        "status": "ok",
        "runs_completed": len(successful),
        # Timing (median across repeats)
        "median_prefill_ms": round(statistics.median(prefills), 2) if prefills else None,
        "min_prefill_ms": round(min(prefills), 2) if prefills else None,
        "max_prefill_ms": round(max(prefills), 2) if prefills else None,
        "median_decode_ms": round(statistics.median(decodes), 2) if decodes else None,
        "min_decode_ms": round(min(decodes), 2) if decodes else None,
        "max_decode_ms": round(max(decodes), 2) if decodes else None,
        "median_total_ms": round(statistics.median(totals), 2),
        "median_decode_tokens_per_sec": round(statistics.median(decode_tps), 2) if decode_tps else None,
        # Memory (from median-time representative run)
        "weight_memory_mb": round(weight_mb, 2),
        "memory_after_load_mb": round(load_alloc, 2),
        "observed_peak_allocated_mb": round(peak_alloc, 2),
        "observed_peak_reserved_mb": round(med.get("observed_peak_reserved_mb", 0), 2),
        "overhead_above_weights_mb": round(overhead_above_weights, 2),
        "generation_overhead_above_load_mb": round(gen_overhead_above_load, 2),
        # KV
        "theoretical_kv_cache_mb": kv_mb,
        "theoretical_kv_cache_seq_len": kv.get("seq_len_used", 0),
        # Attribution (from median-time representative run)
        "peak_phase": peak_phase,
        "coarse_kv_ratio_pct": round(coarse_kv_ratio, 1),
        "runtime_kv_ratio_pct": round(runtime_kv_ratio, 1),
    }


def aggregate_live_runs(runs: list[dict], model_name: str, ptl: int, mnt: int) -> dict:
    """Aggregate runs where primary_result is a GenerationResult object (not yet serialized).

    Converts to dict format and delegates to aggregate_group.
    """
    converted = []
    for r in runs:
        pr = r["primary_result"]
        converted.append({
            "primary_result": pr.to_dict() if hasattr(pr, "to_dict") else pr,
            "secondary_result": r["secondary_result"].to_dict() if hasattr(r["secondary_result"], "to_dict") else r["secondary_result"],
            "metadata": r["metadata"].to_dict() if hasattr(r["metadata"], "to_dict") else r["metadata"],
        })
    return aggregate_group(converted)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_grid_report(
    summaries: list[dict],
    gpu_info: dict,
    output_path: Path,
    metadata_notes: list[str] | None = None,
) -> Path:
    """Write the canonical Markdown grid report.

    Args:
        summaries: List of aggregated summary dicts from aggregate_group.
        gpu_info: From profiler.get_gpu_info().
        output_path: Where to write the .md file.
        metadata_notes: Optional list of notes (e.g., known issues) to include.
    """
    lines: list[str] = []
    ts = timestamp()

    lines.append("# Bio-Inference-Bench — Grid Benchmark Report")
    lines.append(f"\nGenerated: {ts}")
    lines.append("")

    # Environment
    lines.append("## Environment")
    lines.append(f"- GPU: {gpu_info.get('name', 'N/A')} ({gpu_info.get('total_mb', 0):.0f} MB)")
    lines.append(f"- PyTorch: {torch.__version__}")
    try:
        import transformers; lines.append(f"- Transformers: {transformers.__version__}")
    except ImportError:
        pass
    lines.append("- dtype: float16, greedy decode, use_cache=True, batch_size=1")
    lines.append("- Warm-up: 1 run per model (prompt=16, max_new=8), discarded")
    lines.append("- Repeats: 3 per configuration")
    lines.append("- Sequence ceiling: 900 tokens")
    lines.append("")

    # Aggregation semantics
    lines.append("## Aggregation Semantics")
    lines.append("")
    lines.append("- **Timing fields** (prefill_ms, decode_ms, total_ms, tok/s): median across repeats.")
    lines.append("- **Memory and phase fields** (peak_allocated, generation_overhead, peak_phase,")
    lines.append("  runtime_kv_ratio, coarse_kv_ratio): taken from the **median-time representative**")
    lines.append("  run — the repeat whose total_time_ms is closest to the median. Memory allocation")
    lines.append("  is deterministic for a given sequence length, but selecting the median-time run")
    lines.append("  avoids outlier interactions from timing anomalies.")
    lines.append("")

    # Measurement semantics
    lines.append("## Measurement Semantics")
    lines.append("")
    lines.append("- **Theoretical KV cache**: formula-based (`2 * layers * batch * seq * kv_heads * head_dim * 2`), never from CUDA.")
    lines.append("- **Observed peak allocated**: `torch.cuda.max_memory_allocated()` during generation.")
    lines.append("- **Overhead above weights** (coarse audit): peak_allocated - theoretical_weight_memory.")
    lines.append("  Includes load-time overhead (allocator pools, non-weight buffers).")
    lines.append("- **Generation overhead above load** (primary runtime metric): peak_allocated - memory_after_load.")
    lines.append("  Isolates memory growth caused by generation (KV cache, activations, attention matrices, logits, temporaries).")
    lines.append("- **Peak phase**: whether `max_memory_allocated` was reached during prefill or decode,")
    lines.append("  inferred from memory snapshots taken after each phase.")
    lines.append("")
    lines.append("**Neither ratio is a direct component-wise decomposition.** Both are consistency")
    lines.append("checks comparing a formula-based estimate against a CUDA-observed aggregate.")
    lines.append("")

    # Metadata notes
    if metadata_notes:
        lines.append("## Notes")
        lines.append("")
        for note in metadata_notes:
            lines.append(note)
        lines.append("")

    # Per-model tables
    models_seen = sorted(set(s["model_name"] for s in summaries))
    for model_name in models_seen:
        rows = [s for s in summaries if s["model_name"] == model_name]
        ok_rows = sorted(
            [r for r in rows if r["status"] == "ok"],
            key=lambda x: (x["prompt_token_length"], x["max_new_tokens"]),
        )
        failed_rows = [r for r in rows if r["status"] != "ok"]

        lines.append(f"## {model_name}")
        lines.append("")

        if ok_rows:
            lines.append("| Prompt | MaxNew | SeqLen | Phase | Prefill ms | Decode ms | Tok/s | TheoKV | GenOverhead | RuntimeRatio | PeakAlloc |")
            lines.append("|--------|--------|--------|-------|------------|-----------|-------|--------|-------------|--------------|-----------|")
            for r in ok_rows:
                lines.append(
                    f"| {r['prompt_token_length']} | {r['max_new_tokens']} "
                    f"| {r['actual_total_seq_length']} "
                    f"| {r['peak_phase']} "
                    f"| {r['median_prefill_ms']} "
                    f"| {r['median_decode_ms']} "
                    f"| {r['median_decode_tokens_per_sec']} "
                    f"| {r['theoretical_kv_cache_mb']} "
                    f"| {r['generation_overhead_above_load_mb']} "
                    f"| {r['runtime_kv_ratio_pct']}% "
                    f"| {r['observed_peak_allocated_mb']} |"
                )
            lines.append("")
            lines.append("*TheoKV = theoretical KV cache (MB). GenOverhead = generation_overhead_above_load (MB).*")
            lines.append("*RuntimeRatio = TheoKV / GenOverhead. Phase = where peak allocated occurred.*")
            lines.append("")

        if failed_rows:
            lines.append(f"**Failed ({len(failed_rows)}):**")
            for r in failed_rows:
                lines.append(f"- prompt={r['prompt_token_length']}, max_new={r['max_new_tokens']}: {r.get('error', '?')}")
            lines.append("")

    # Scaling analysis
    lines.append("## Scaling Analysis")
    lines.append("")
    lines.append("The same final sequence length can arise from different prompt/decode splits.")
    lines.append("Peak memory behavior may differ: longer prompts concentrate memory pressure in")
    lines.append("prefill (full attention over the prompt), while longer decode runs accumulate")
    lines.append("KV cache incrementally. The peak_phase column captures this distinction.")
    lines.append("")

    for model_name in models_seen:
        ok_rows = sorted(
            [s for s in summaries if s["model_name"] == model_name and s["status"] == "ok"],
            key=lambda x: x["actual_total_seq_length"],
        )
        if not ok_rows:
            continue

        lines.append(f"### {model_name}")
        lines.append("")
        lines.append("| SeqLen | Prompt | MaxNew | Phase | TheoKV | GenOverhead | RuntimeRatio | CoarseRatio |")
        lines.append("|--------|--------|--------|-------|--------|-------------|--------------|-------------|")
        for r in ok_rows:
            lines.append(
                f"| {r['actual_total_seq_length']} "
                f"| {r['prompt_token_length']} | {r['max_new_tokens']} "
                f"| {r['peak_phase']} "
                f"| {r['theoretical_kv_cache_mb']} "
                f"| {r['generation_overhead_above_load_mb']} "
                f"| {r['runtime_kv_ratio_pct']}% "
                f"| {r['coarse_kv_ratio_pct']}% |"
            )
        lines.append("")
        lines.append("*RuntimeRatio = TheoKV / GenOverhead (runtime focus). CoarseRatio = TheoKV / overhead_above_weights (audit).*")
        lines.append("*Neither is a direct component decomposition.*")
        lines.append("")

    # Model-specific findings
    lines.append("## Model-Specific Observations")
    lines.append("")

    for model_name in models_seen:
        ok_rows = [s for s in summaries if s["model_name"] == model_name and s["status"] == "ok"]
        if not ok_rows:
            continue

        lines.append(f"### {model_name}")
        lines.append("")

        short = [r for r in ok_rows if r["actual_total_seq_length"] <= 128]
        long_ = [r for r in ok_rows if r["actual_total_seq_length"] >= 400]
        prefill_peaks = [r for r in ok_rows if r["peak_phase"] == "prefill"]
        decode_peaks = [r for r in ok_rows if r["peak_phase"] == "decode"]

        avg_runtime = statistics.mean(r["runtime_kv_ratio_pct"] for r in ok_rows)
        avg_short = statistics.mean(r["runtime_kv_ratio_pct"] for r in short) if short else 0
        avg_long = statistics.mean(r["runtime_kv_ratio_pct"] for r in long_) if long_ else 0

        lines.append(f"- Configurations tested: {len(ok_rows)}")
        lines.append(f"- Peak phase: {len(prefill_peaks)} prefill, {len(decode_peaks)} decode, "
                     f"{len(ok_rows) - len(prefill_peaks) - len(decode_peaks)} unclear")
        lines.append(f"- Runtime KV ratio (TheoKV / GenOverhead):")
        if short:
            lines.append(f"  - Short sequences (seq <= 128): avg {avg_short:.1f}%")
        if long_:
            lines.append(f"  - Long sequences (seq >= 400): avg {avg_long:.1f}%")
        lines.append(f"  - Overall average: {avg_runtime:.1f}%")
        lines.append("")

        if model_name == "protgpt2":
            lines.append("Theoretical KV becomes large relative to generation-overhead-above-load")
            lines.append("at longer sequences. However, peak attribution depends on the prompt/decode")
            lines.append(f"split: {len(prefill_peaks)} configs peaked during prefill and {len(decode_peaks)}")
            lines.append("peaked during decode. Configurations with the same final sequence length but")
            lines.append("different prompt/decode ratios can exhibit different peak phase behavior.")
            lines.append("This suggests that both prefill-time activations and decode-time KV cache")
            lines.append("growth contribute to peak memory, with the balance shifting based on the")
            lines.append("prompt/decode split.")
        elif model_name == "progen2-small":
            lines.append("Theoretical KV remains a minority of runtime growth across the tested range.")
            lines.append("A substantial non-KV runtime baseline persists even after separating post-load")
            lines.append("growth. This baseline is consistent with allocator reserved pools, activation")
            lines.append("buffers, and logit computation overhead that scale weakly with sequence length.")
            lines.append("KV cache grows linearly but does not dominate the overhead budget at any")
            lines.append("tested sequence length.")
        lines.append("")

    # Combined summary (secondary)
    lines.append("## Combined Pattern (secondary note)")
    lines.append("")
    lines.append("These observations are consistent with materially different memory behavior")
    lines.append("between the two models, driven primarily by architectural differences")
    lines.append("(36 layers vs 12 layers, resulting in 3.75x more KV cache per token for ProtGPT2).")
    lines.append("Neither model's behavior can be fully attributed to a single component")
    lines.append("from these measurements alone.")
    lines.append("")

    # Execution summary
    total = len(summaries)
    completed = sum(1 for s in summaries if s["status"] == "ok")
    failed = total - completed
    ooms = sum(1 for s in summaries if "OOM" in str(s.get("error", "")))

    lines.append("## Execution Summary")
    lines.append("")
    lines.append(f"- Configurations attempted: {total}")
    lines.append(f"- Completed: {completed}")
    lines.append(f"- Failed: {failed}")
    lines.append(f"- OOMs: {ooms}")

    for model_name in models_seen:
        model_ok = sum(1 for s in summaries if s["model_name"] == model_name and s["status"] == "ok")
        model_fail = sum(1 for s in summaries if s["model_name"] == model_name and s["status"] != "ok")
        lines.append(f"- {model_name}: {model_ok} completed, {model_fail} failed")

    lines.append("")
    lines.append("---")
    lines.append("*This is a profiling benchmark. No optimization or model-ranking conclusions are drawn.*")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return output_path
