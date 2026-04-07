# Constrained Benchmark Grid Plan

**Date:** 2026-04-06
**Status:** Proposal — not yet executed. Awaiting mentor approval.

## Objective

Measure how inference memory and timing scale with sequence length across two
protein autoregressive models. The goal is to identify which memory component
(KV cache, attention matrices, activations, allocator overhead) dominates as
sequences grow — not to compare models against each other.

## Models

| Model | Params | Status |
|-------|--------|--------|
| ProtGPT2 (`nferruz/ProtGPT2`) | 774M | Validated in smoke |
| ProGen2-small (`hugohrban/progen2-small`) | 151M | Validated with local patch |

## Parameter Grid

### Prompt token lengths
`[16, 32, 64, 128, 256]`

Rationale: Start well below max context (1024 for both models) and increase
geometrically to observe how prefill time and memory scale with input length.

### Max new tokens
`[32, 64, 128, 256, 512]`

Rationale: Vary decode length to separate KV cache growth from other memory
components. At 512 generated tokens with a 256-token prompt, total sequence
reaches 768 — approaching the 1024 max context.

### Excluded configurations
- `prompt_token_length + max_new_tokens > 900`: Skip to avoid OOM near context
  limit and leave headroom for allocator overhead.
- This means the grid is not a full cross-product; some (prompt, max_new)
  combinations are excluded.

### Fixed parameters
- `batch_size`: 1 (batching not yet supported)
- `do_sample`: False (greedy)
- `num_beams`: 1
- `use_cache`: True
- `dtype`: float16

### Total configurations
Approximate: 2 models x ~18 valid (prompt, max_new) pairs = ~36 runs.

## Warm-up Policy

Each model gets **1 warm-up run** before timed measurements begin:
- Warm-up config: `prompt_token_length=16, max_new_tokens=8`
- Warm-up is timed but results are **discarded** (not included in output)
- Purpose: trigger CUDA kernel JIT compilation and allocator warm-up
- After warm-up, `reset_peak_memory_stats()` is called before the first real run

## Repeats

Each configuration is run **3 times**. Report:
- Median timing (prefill, decode, total)
- Min/max range
- Only the run with median total time has its memory snapshot reported
  (memory is deterministic given the same sequence length, but the median
  run avoids any outlier measurement interaction)

## Output Artifact Structure

```
results/
  raw/
    grid_{model}_{prompt}_{max_new}_{run_idx}_{timestamp}.json
  summaries/
    grid_summary_{timestamp}.json
    grid_report_{timestamp}.md
```

### Per-run JSON (`grid_{model}_...json`)
Same schema as smoke results:
- metadata
- primary_result (manual prefill/decode — authoritative)
- secondary_result (generate API — cross-validation)
- theoretical_kv_upper_bound

### Grid summary JSON (`grid_summary_...json`)
Aggregated table with one row per (model, prompt, max_new) config:
- model_name
- prompt_token_length
- max_new_tokens
- actual_total_seq_length
- median_prefill_ms, min_prefill_ms, max_prefill_ms
- median_decode_ms, min_decode_ms, max_decode_ms
- median_decode_tokens_per_sec
- theoretical_kv_cache_mb (at actual seq length)
- theoretical_kv_upper_bound_mb (at configured max)
- observed_peak_allocated_mb (from median run)
- observed_peak_reserved_mb (from median run)
- weight_memory_mb
- overhead_above_weights_mb (observed_peak_allocated - weight_memory)

### Grid report Markdown (`grid_report_...md`)
Human-readable summary with:
- Scaling tables: how does overhead_above_weights grow with seq length?
- Theoretical KV vs observed overhead ratio at each seq length
- Prefill/decode time breakdown at each seq length
- Per-model scaling plots described (actual rendering deferred)
- Clear separation of theoretical and observed metrics
- No model-vs-model ranking conclusions

## Reporting Guardrails

1. **Theoretical KV cache** is always formula-based. Reported at the actual
   final sequence length per run, plus an upper-bound at configured max.
2. **Observed peak allocated** is `torch.cuda.max_memory_allocated()`.
3. **Observed peak reserved** is `torch.cuda.max_memory_reserved()`.
4. These three metrics are **always reported in separate columns**. The report
   will **never** compute "KV cache = peak - weights" or present observed
   overhead as if it were KV cache.
5. The grid report includes a "Measurement Semantics" section aligned with the
   smoke benchmark methodology.
6. Interpretation uses hedged language: "consistent with" rather than "is".

## Execution Gating

This plan will not be executed until approved by mentor review.
`run_grid.py` remains gated with exit code 1.
