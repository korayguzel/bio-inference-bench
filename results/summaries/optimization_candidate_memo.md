# Optimization Candidate Decision Memo

**Date:** 2026-04-06
**Baseline:** Benchmark v1 (`grid_summary_20260406T191552.json`)
**Profiling:** Operator-level traces from 5 representative configs

## Representative Configs and Selection Rationale

| Label | Model | Prompt | MaxNew | Why selected |
|-------|-------|--------|--------|-------------|
| protgpt2_decode_heavy | ProtGPT2 | 32 | 256 | Runtime KV ratio 98.5%, peak in decode. Tests KV-dominated regime. |
| protgpt2_prefill_heavy | ProtGPT2 | 256 | 32 | Runtime KV ratio 71.9%, peak in prefill. Tests prefill-dominated regime. |
| protgpt2_long_decode | ProtGPT2 | 256 | 512 | Longest config (seq=768), runtime KV ratio 98.0%. Tests scaling ceiling. |
| progen2_representative | ProGen2-small | 64 | 128 | Runtime KV ratio 33.4%. Tests typical non-KV-dominated behavior. |
| progen2_long_decode | ProGen2-small | 256 | 512 | Runtime KV ratio 32.2%. Tests whether long sequences shift the balance. |

## Operator-Level Evidence

### ProtGPT2

**Prefill phase (prompt=32):** Total CUDA time ~38 ms.
- `aten::addmm` (linear layer GEMMs): 10.4% of CUDA time
- Tensor-core GEMM kernels (`ampere_fp16_s16816gemm_*`): ~9% combined
- `scaled_dot_product_attention`: 0.9%
- **Interpretation:** Prefill is dominated by weight-matrix multiplications
  (QKV projections, output projections, FFN layers), not attention score computation.
  This is expected for short prompts where the attention matrix is small (32 x 32).

**Prefill phase (prompt=256):** Total CUDA time ~55 ms.
- `aten::addmm`: 14.4%
- Tensor-core GEMMs: ~13% combined (larger kernels selected for 256-length sequences)
- **Interpretation:** Still GEMM-dominated, but larger GEMM tile sizes selected by cuBLAS.
  The 47% increase in CUDA time (38→55 ms) is roughly proportional to prompt length growth
  (32→256 = 8x) amortized across 36 layers, suggesting linear-layer compute scales with
  prompt length while attention remains a small fraction at batch=1.

**Decode phase (32 steps, all configs):** Total CUDA time ~870-930 ms.
- `aten::addmm` (linear layers): ~12%
- `gemvx::kernel` (batch=1 matrix-vector multiply): ~12% combined (two kernel variants)
- `scaled_dot_product_attention` / flash_attention: ~1.2%
- **Interpretation:** Decode is dominated by **weight-matrix × vector multiplies** through
  the 36-layer stack. The gemvx kernels are cuBLAS's specialized path for when one dimension
  is 1 (batch=1 token generation). Attention (SDPA) accounts for only ~1% of CUDA time.
  This means the per-step decode bottleneck is **weight access / compute**, not
  attention score computation or KV cache read bandwidth.

### ProGen2-small

**Prefill phase:** Total CUDA time 28-41 ms.
- `aten::linear`: 3-5%
- `aten::addmm`: 2-3%
- GEMM kernels: ~3% combined
- **Interpretation:** More distributed operator profile (no single op > 5% excluding
  the profiler wrapper). The model has only 12 layers, so each layer's contribution is
  smaller. Linear layers still dominate but with less concentration.

**Decode phase (32 steps):** Total CUDA time ~540-550 ms.
- `aten::linear`: 4.3%
- `gemvx::kernel`: 3.9%
- `aten::addmm`: 2.8%
- `aten::cat` (tensor concatenation): 1.2-1.4%
- **Interpretation:** Similar pattern to ProtGPT2 (weight-vector multiplies dominate)
  but with a notable `aten::cat` contribution. The `cat` operations are KV cache updates —
  concatenating new K/V vectors onto the growing cache at each step. This is a KV cache
  *management* overhead, not a KV cache *size* bottleneck. It suggests the ProGen2 custom
  code uses tensor concatenation for cache growth rather than pre-allocated buffers.

## Memory vs Compute Summary

| Regime | Compute bottleneck | Memory bottleneck (from grid) |
|--------|-------------------|-------------------------------|
| ProtGPT2 decode-heavy | Weight-vector GEMMs (gemvx) | Generation overhead ~98% consistent with KV cache |
| ProtGPT2 prefill-heavy | Weight-matrix GEMMs (addmm) | Peak in prefill; runtime KV ratio ~72% |
| ProtGPT2 long decode | Weight-vector GEMMs (gemvx) | Generation overhead ~98% consistent with KV cache |
| ProGen2 representative | Weight-vector GEMMs + linear | Non-KV baseline dominates (~67% of gen overhead) |
| ProGen2 long decode | Weight-vector GEMMs + linear + cat | Non-KV baseline dominates (~68% of gen overhead) |

The key finding is a **compute vs memory asymmetry**: for ProtGPT2 decode, the compute
bottleneck is weight access (not attention), while the memory growth is consistent with
KV cache. These are different bottlenecks operating in different dimensions:
- Reducing KV cache memory would enable longer sequences or larger batches
- Reducing KV cache memory would NOT directly speed up per-step decode time
  (because decode time is dominated by weight GEMMs, not KV cache reads)

## Per-Regime Optimization Recommendations

### ProtGPT2 — Decode-Heavy Regime (prompt=32, max_new=256)

**Recommendation: A — KV-cache-focused prototype**

- Runtime KV ratio is ~98.5%, suggesting KV cache is the dominant memory consumer
  during generation.
- The purpose would be to **increase maximum sequence length or batch capacity**,
  not to speed up per-step decode (which is GEMM-bound).
- A quantized KV cache (e.g., INT8 or 4-bit) would reduce memory footprint
  proportionally, freeing headroom for longer sequences.
- **Confidence:** Moderate-high. The memory signal is strong (98.5% ratio) but
  remember this is a consistency check, not a direct decomposition.
- **Risk:** If the ~1.5% non-KV overhead grows nonlinearly at much longer sequences,
  the benefit ceiling may be lower than expected.

### ProtGPT2 — Prefill-Heavy Regime (prompt=256, max_new=32)

**Recommendation: B — Prefill/attention-focused investigation**

- Peak memory occurs during prefill, not decode.
- Runtime KV ratio drops to ~72%, meaning ~28% of generation overhead is
  non-KV (likely prefill-time activations and attention intermediates).
- The compute profile shows prefill is GEMM-dominated, not attention-dominated.
  This means the memory peak is likely from **activation tensors** held during
  the forward pass (all 36 layers' intermediate tensors), not attention scores.
- Possible interventions: activation checkpointing (trade compute for memory),
  or chunked prefill (process the prompt in segments).
- **Confidence:** Moderate. The prefill peak is clear, but the split between
  activations, attention intermediates, and allocator overhead is not directly measured.
- **Risk:** Activation checkpointing may slow prefill substantially for marginal
  memory savings, especially since prefill is already a small fraction of total time
  in decode-heavy workloads. This regime is only relevant when prompts are long
  relative to generation.

### ProtGPT2 — Long Decode Regime (prompt=256, max_new=512)

**Recommendation: A — KV-cache-focused prototype**

- Same rationale as decode-heavy: runtime KV ratio 98.0%, peak in decode.
- At seq_len=768, theoretical KV cache is 135 MB — approaching 10% of
  the RTX 4070's 12 GB. At longer sequences or batch > 1, KV cache would
  become a hard capacity constraint.
- **Confidence:** Moderate-high. Same caveats as decode-heavy regime.

### ProGen2-small — Representative Decode (prompt=64, max_new=128)

**Recommendation: D — No immediate KV optimization justified**

- Runtime KV ratio is only 33.4%. Two-thirds of generation overhead is non-KV.
- The `aten::cat` operations suggest the custom model code manages KV cache
  via tensor concatenation rather than pre-allocated buffers. This concatenation
  overhead is a **KV management cost** that would not be reduced by KV quantization.
- The non-KV baseline (~17-18 MB) is likely allocator pools and activation buffers.
- **Confidence:** Moderate. The non-KV dominance is clear, but the exact composition
  of that baseline is not decomposed by these measurements.
- **If optimization were needed:** Investigate the `aten::cat` overhead first — switching
  to pre-allocated KV cache tensors (as ProtGPT2/GPT2 already does) would remove the
  concatenation cost without any quantization. This is an engineering fix, not a
  research intervention.

### ProGen2-small — Long Decode (prompt=256, max_new=512)

**Recommendation: C — Investigate non-KV runtime baseline first**

- Even at the longest config, runtime KV ratio is only 32.2%.
- The non-KV overhead (~76 MB) grows with sequence length, suggesting it is not
  purely a fixed allocator cost — there may be activation or buffer growth.
- Before any KV-focused work, understanding what constitutes the ~68% non-KV
  portion would be more valuable.
- **Confidence:** Moderate. The pattern is stable across all ProGen2 configs,
  lending consistency to the observation.
- **Possible next steps:** Use `torch.cuda.memory_snapshot()` for a full allocation
  trace, or add per-layer memory checkpoints to identify which components grow.

## Overall Assessment

The operator-level profiling **does not contradict** the Benchmark v1 grid findings.
It adds compute-dimension evidence that refines the picture:

1. **ProtGPT2's decode bottleneck is a compute/memory split:** per-step time is
   GEMM-bound (weight access), but memory growth is KV-dominated. KV cache
   optimization would address the memory dimension (enabling longer sequences
   or batching) without directly accelerating individual decode steps.

2. **ProGen2-small does not currently justify KV-first optimization.** The non-KV
   runtime baseline dominates, and the `aten::cat` KV management pattern suggests
   an engineering-level fix (pre-allocated cache) before any quantization research.

3. **Prompt/decode split matters for ProtGPT2.** The same model has materially
   different bottleneck behavior depending on whether the workload is prompt-heavy
   or decode-heavy. Any optimization strategy must account for the expected
   workload mix.

---
*This memo is decision-support for optimization planning. No optimization has been implemented.*
