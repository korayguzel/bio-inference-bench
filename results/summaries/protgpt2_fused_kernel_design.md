# ProtGPT2 Fused INT8 KV Attention Kernel: Design Memo

**Date:** 2026-04-06
**Baseline:** v2 chunked-dequantize INT8 prototype
**Evidence:** v2 eval, v3 eval (boundary protection), operator profiling, optimization memo
**Scope:** Design only — no implementation in this document

---

## 1. Problem Statement

### What v2 Solved

The v2 chunked-dequantize prototype achieved a **47% reduction in generation overhead**
for ProtGPT2 decode-heavy workloads by storing KV cache in INT8 and dequantizing in
chunks during attention, using the online softmax algorithm. The full FP16 KV cache is
never materialized during decode.

- Behavioral fidelity: 100% token agreement, cosine similarity ~0.99999 across 64 steps
- Cache compression: 1.94x (INT8 + FP16 per-token scales vs FP16)

### What v2 Left Unsolved

The v2 path is **3-6x slower** than the FP16 baseline:

| Config | Baseline tok/s | v2 tok/s | Slowdown |
|--------|---------------|----------|----------|
| decode_heavy (p=32, n=256) | 128.4 | 40.3 | 3.2x |
| long_decode (p=256, n=512) | 124.9 | 21.6 | 5.8x |

The slowdown scales with sequence length because the Python chunked attention loop
processes more chunks at longer sequences. The per-chunk overhead is dominated by:

1. **Python loop overhead:** One iteration per chunk per layer per decode step
2. **Per-chunk tensor allocation:** `dequantize_from_int8()` creates new FP16 tensors
   for each chunk, which must be allocated and freed
3. **Per-chunk kernel launches:** Each dequant + matmul + softmax update dispatches
   multiple small CUDA kernels rather than one fused operation
4. **Memory traffic:** INT8 KV data is read from global memory per chunk, dequantized
   to FP16 intermediates in global memory, then consumed by matmul — three memory round trips

The v3 experiment (boundary-layer protection) showed a ~10% speed improvement when 4 of
36 layers used standard SDPA instead of chunked attention. This confirms that the
bottleneck is in the chunked attention implementation, not in INT8 quantization itself.

### Why This Matters

ProtGPT2's decode is GEMM-bound in compute (weight-vector multiplies dominate per-step
time) but KV-dominated in memory growth (~98% of generation overhead is consistent with
KV cache). This is a **capacity optimization**, not a throughput optimization:

- **Goal:** Recover the v2 memory savings (~47%) while keeping speed regression within
  an acceptable bound
- **Non-goal:** Exceeding baseline FP16 throughput (the per-step decode bottleneck is
  weight GEMMs, not attention)

A fused kernel eliminates the Python/PyTorch overhead layers while preserving the same
INT8 chunked attention algorithm that v2 validated.

---

## 2. Justification for Fused Execution

The roadmap defined four escalation criteria for fused-kernel work:

| Criterion | Required | Status |
|-----------|----------|--------|
| Memory benefit ≥40% | ≥40% | **Met: 47%** (v2) |
| Behavioral stability ≥99.99% top-1 | ≥99.99% at 256+ steps | **Met: 100%** at 64 steps, 2 configs |
| Speed regression >2x | >2x | **Met: 3.2-5.8x** |
| Algorithmic improvements exhausted | v3/v4 explored | **Partially met:** v3 tested (no benefit); v4 (torch.compile) skipped |

### Why Skip v4 (Optimized PyTorch)

The roadmap originally proposed v4 (`torch.compile()` on the chunked loop) as an
intermediate step. We skip it for the following reasons:

1. **`torch.compile()` cannot fuse across the Python chunk loop.** The online softmax
   accumulation happens in Python, and `torch.compile()` would at best optimize
   individual chunk operations, not eliminate the loop or inter-chunk overhead.

2. **The dominant overhead is structural.** Per-chunk tensor allocation, multiple kernel
   launches per chunk, and Python loop iterations are all overhead that `torch.compile()`
   cannot remove — these require a single fused kernel.

3. **Triton development cost is comparable.** A minimal Triton kernel for decode-only
   single-query attention over INT8 KV is ~100-150 lines. The development cost of a
   careful `torch.compile()` investigation (with fallbacks and debugging) is similar,
   with lower ceiling on improvement.

4. **v3's 10% result provides a scaling estimate.** Replacing 4 of 36 layers' chunked
   attention with SDPA gave 10% speedup. A fused kernel replacing all 36 layers'
   chunked attention should recover most of the remaining overhead.

---

## 3. Kernel Target Specification

### Model Parameters (ProtGPT2)

| Parameter | Value |
|-----------|-------|
| num_layers | 36 |
| num_heads | 20 |
| head_dim | 64 |
| n_embd | 1280 |
| batch_size | 1 (single-sequence decode) |
| max_position_embeddings | 1024 |

### Data Types

| Tensor | Dtype | Shape (decode) | Notes |
|--------|-------|-----------------|-------|
| Query (Q) | float16 | (1, 20, 1, 64) | Single new token |
| Key cache (K) | int8 | (1, 20, seq_len, 64) | Per-token absmax quantized |
| Key scales (K_s) | float16 | (1, 20, seq_len, 1) | Per-token scale factor |
| Value cache (V) | int8 | (1, 20, seq_len, 64) | Per-token absmax quantized |
| Value scales (V_s) | float16 | (1, 20, seq_len, 1) | Per-token scale factor |
| Output (O) | float16 | (1, 20, 1, 64) | Attention output |
| Scaling factor | float32 | scalar | 1/sqrt(head_dim) = 1/8 |

### Quantization Scheme

Per-token symmetric absmax quantization along the head_dim dimension:

```
scale = max(|x|) / 127.0          # per (batch, head, position)
x_int8 = round(x / scale)          # clamped to [-128, 127]
x_fp16 = x_int8.to(fp16) * scale   # dequantization
```

This is the exact scheme used in `kv_int8_cache.py:quantize_to_int8()`.

### Attention Computation

**Exact attention** — no approximation. The kernel computes the same mathematical
operation as the v2 Python path:

```
for each KV block of BLOCK_KV positions:
    K_block = dequant(K_int8[start:end], K_scales[start:end])   # INT8 → FP16
    scores = Q @ K_block^T * scaling                             # (1, BLOCK_KV)
    # Online softmax update (Milakov & Gimelshein 2018):
    block_max = max(scores)
    new_max = max(running_max, block_max)
    rescale = exp(running_max - new_max)
    exp_scores = exp(scores - new_max)
    running_sum = running_sum * rescale + sum(exp_scores)
    V_block = dequant(V_int8[start:end], V_scales[start:end])   # INT8 → FP16
    running_output = running_output * rescale + exp_scores @ V_block
    running_max = new_max
output = running_output / running_sum
```

### Block/Tile Layout

| Parameter | Initial Value | Rationale |
|-----------|--------------|-----------|
| BLOCK_KV | 64 | Matches v2 chunk_size; one block fits in SRAM for head_dim=64 |
| Program grid | (batch × num_heads,) | One Triton program per (batch, head) pair |
| Q in registers | Yes | Q is (1, 64) — fits in registers for the full decode step |

Each Triton program:
- Loads Q once (64 × FP16 = 128 bytes)
- Loops over KV in blocks of BLOCK_KV=64 positions
- Per block: loads K_int8 (64×64 = 4096 bytes), K_scales (64×2 = 128 bytes),
  V_int8 (4096 bytes), V_scales (128 bytes)
- Dequantizes K and V to FP16 in SRAM
- Computes dot products and online softmax update
- Writes final output O (64 × FP16 = 128 bytes)

### Memory Access Pattern

Per block iteration, per (batch, head) program:
- **Read:** 4096 + 128 + 4096 + 128 = 8448 bytes of INT8+scales
- **Equivalent FP16 read would be:** 64 × 64 × 2 × 2 = 16384 bytes
- **Bandwidth savings:** 8448 / 16384 = 51.6% (matches 1.94x compression ratio)

The kernel reads INT8 data from global memory → SRAM, dequantizes in SRAM, and never
writes FP16 intermediates back to global memory. This is the key advantage over v2,
which materializes FP16 chunks in global memory.

### What Happens Outside the Kernel

The kernel handles **only** the attention computation for one layer's decode step.
Everything else remains in PyTorch:

| Component | Location | Rationale |
|-----------|----------|-----------|
| Token embedding + position | PyTorch | Runs once, negligible cost |
| QKV projection (c_attn) | PyTorch (cuBLAS) | Already optimized GEMM |
| **INT8 attention** | **Triton kernel** | **This is the fused kernel** |
| Output projection (c_proj) | PyTorch (cuBLAS) | Already optimized GEMM |
| Residual + LayerNorm | PyTorch | Already fused by torch |
| FFN (MLP) | PyTorch (cuBLAS) | Already optimized GEMM |
| LM head | PyTorch | Runs once, negligible cost |
| KV cache update (quantize + append) | PyTorch | Small per-step cost |

The `run_chunked_decode_step()` function in `kv_int8_chunked.py` already implements
this decomposition. The fused kernel replaces only the `cache.chunked_attention()` call.

---

## 4. Precision Choices

| Choice | Decision | Rationale |
|--------|----------|-----------|
| KV storage | INT8 uniform all layers | v3 showed boundary protection unnecessary for ProtGPT2 |
| Scale storage | FP16 | Matches quantization output; FP32 would double scale memory for negligible benefit |
| Accumulation (M, L, O) | FP32 | Online softmax requires stable numerics; FP16 exp() can overflow |
| Score computation | FP16 matmul → FP32 accumulation | Triton's `tl.dot` accumulates in FP32 by default |
| Final output | FP16 | Match model's working dtype |

**FP32 accumulators are critical.** The online softmax involves `exp(scores - max)` and
running sums. In FP16, `exp()` overflows at ~11.1 and underflows near 0.0. FP32
accumulators prevent numerical issues without affecting memory footprint (accumulators
are register-resident, not stored in global memory).

---

## 5. Explicit Out-of-Scope for Kernel v1

| Feature | Status | Rationale |
|---------|--------|-----------|
| Prefill attention | Out of scope | Prefill uses standard FP16 SDPA; INT8 is decode-only |
| Batch size > 1 | Out of scope | Current benchmark is single-sequence; batching adds complexity |
| GQA / MQA | Out of scope | ProtGPT2 uses MHA (num_kv_heads = num_heads = 20) |
| Variable-length batching | Out of scope | Single sequence |
| Causal masking | Out of scope | Decode attention over past KV is fully causal by construction |
| Backward pass | Out of scope | Inference only |
| INT4 / mixed-bit KV | Out of scope | Uniform INT8 is the validated path |
| Boundary-layer protection | Out of scope | v3 showed no benefit for ProtGPT2 |
| Asymmetric K/V precision | Out of scope | Deferred pending fused baseline results |
| Orthogonal rotation | Out of scope | Deferred; adds matmul overhead per dequant |
| torch.compile integration | Out of scope | Kernel is called directly, not via compile |
| FlashAttention-2 tiling | Out of scope | Over-engineered for batch=1, q_len=1 decode |

---

## 6. Phased Implementation Plan

### Phase A — Minimal Correctness Kernel

**Goal:** A Triton kernel that produces numerically identical output to
`ChunkedInt8KVCache.chunked_attention()` for ProtGPT2 decode.

**Deliverables:**
1. `bio_inference_bench/triton_int8_attention.py` — the Triton kernel + Python wrapper
2. `tests/test_triton_int8_attention.py` — correctness tests

**Scope:**
- Single decode step: Q=(1, 20, 1, 64), KV up to 1024 positions
- Fixed BLOCK_KV=64
- FP32 accumulators for online softmax
- No performance tuning

**Correctness validation:**
- Generate random Q, K_int8, K_scales, V_int8, V_scales
- Run v2 `chunked_attention()` as reference
- Run Triton kernel
- Assert: `torch.allclose(triton_output, v2_output, atol=1e-3, rtol=1e-3)`
- Test at seq_len = 1, 63, 64, 65, 128, 256, 512, 768, 1024 (edge cases around
  block boundaries)

**Exit criterion:** All correctness tests pass. No performance measurement yet.

### Phase B — Benchmarkable Kernel

**Goal:** Integrate the Triton kernel into the decode loop and measure end-to-end
performance against baseline and v2.

**Deliverables:**
1. Updated `run_chunked_decode_step()` with a `use_triton=True` path that calls the
   Triton kernel instead of `chunked_attention()`
2. `scripts/eval_kv_fused_v5.py` — evaluation script comparing baseline / v2 / v5
3. `results/summaries/protgpt2_fused_kernel_v5_report.md` — generated from raw JSON

**Scope:**
- Same two configs as v2/v3 eval: decode_heavy (p=32, n=256), long_decode (p=256, n=512)
- Fresh model load per path
- Sanity checks: token agreement, logit cosine similarity vs baseline
- Memory measurement: peak allocated, generation overhead
- Speed measurement: decode tok/s, end-to-end tok/s

**Exit criteria:**
- Memory overhead within 5% of v2 (confirming no regression from kernel integration)
- Speed regression vs baseline < 50% (significant improvement over v2's 3-6x)
- Token agreement ≥ 99% at 64 steps (smoke check)
- Token agreement ≥ 95% at 256 steps on long_decode (readiness gate)
- Cosine similarity ≥ 0.9999 at 64 steps, ≥ 0.999 at 256 steps

### Phase C — Optimization Passes

**Goal:** Tune the kernel for RTX 4070 (Ada Lovelace, SM 8.9) to minimize speed
regression vs baseline.

**Candidate optimizations (ordered by expected impact):**

1. **BLOCK_KV tuning:** Try 32, 64, 128, 256. Larger blocks reduce loop iterations
   but increase SRAM pressure. Profile to find the sweet spot.

2. **Memory coalescing:** Ensure K_int8 loads are coalesced along head_dim (the
   innermost dimension). The current layout (batch, heads, seq_len, head_dim) should
   be favorable since head_dim is contiguous.

3. **Persistent kernel:** If the KV sequence fits in a single program's iteration
   budget, avoid re-launching the kernel per layer. Instead, pass all 36 layers'
   KV pointers and loop inside a single kernel launch.

4. **`tl.dot` configuration:** Experiment with `tl.dot` input precision hints
   (e.g., `input_precision="tf32"` if available for INT8→FP16 matmuls).

5. **Warp-level reductions:** For the online softmax `amax` and `sum` operations,
   ensure Triton generates efficient warp shuffles rather than shared memory reductions.

**Exit criteria:**
- Speed regression vs baseline < 20% (target from roadmap)
- Memory savings ≥ 40% (allowing for kernel-side buffer overhead)
- No behavioral regression from Phase B

**Not in Phase C:**
- Batch size > 1
- Different model architectures
- INT4 or mixed precision

---

## 7. Go/No-Go on TurboQuant-Plus Ideas

Based on v2/v3 evidence, classify each idea from Track 2 of the roadmap:

### Boundary-Layer Protection — Likely Unnecessary

**Evidence:** v3 tested this directly. Protecting layers {0, 1, 34, 35} in FP16
produced zero measurable quality improvement (cosine 0.999993 → 0.999993) while
costing 7 percentage points of memory savings (47% → 40%).

**Decision:** Drop. Do not reopen unless future evidence (longer sequences, different
prompts, different models) shows layer-specific sensitivity.

**Impact on kernel design:** The fused kernel can be uniform — same INT8 dequantize
logic for all 36 layers. No per-layer precision dispatch needed.

### Asymmetric K/V Precision — Defer Until After Fused Baseline

**Evidence:** No direct test yet. The turbogene project showed K and V have different
outlier profiles (K outlier/median 5.5-8.6x, V 6.9-11.7x), but for ProtGPT2 the
uniform INT8 path already achieves cosine ~0.99999.

**Decision:** Defer. The current quality is already near-perfect. Testing asymmetric
precision (e.g., INT4 K + INT8 V) would only matter if:
1. We want to push compression beyond 1.94x (e.g., to 2.5x+ for very long sequences)
2. Quality degrades at longer sequences than currently tested (>64 steps)

**When to reconsider:** After the fused kernel is benchmarkable (Phase B). If the memory
savings with uniform INT8 are sufficient for the target use case, asymmetric precision
adds complexity for diminishing returns. If longer-sequence evaluation reveals quality
degradation, asymmetric precision becomes a mitigation tool.

### Per-Head Calibration — Likely Unnecessary

**Evidence:** No direct test, but the uniform INT8 path shows 100% top-1 logit
agreement across all 20 heads simultaneously. If any individual heads were
disproportionately sensitive, we would expect to see logit disagreement at those
heads — but we see perfect agreement.

**Decision:** Defer indefinitely. The calibration infrastructure (collecting per-head
statistics across prompts, assigning per-head bit widths) is high-cost for a problem
that does not appear to exist for ProtGPT2. Revisit only if a different model or
significantly longer sequence evaluation reveals head-specific degradation.

### Orthogonal Rotation — Defer, Likely Unnecessary for ProtGPT2

**Evidence:** Orthogonal rotation before quantization distributes outlier energy across
dimensions, improving quantization quality. TurboQuant showed +0.014 cosine improvement
in turbogene. However, ProtGPT2's INT8 cosine is already 0.99999 — there is almost no
headroom to improve.

**Decision:** Defer. Rotation adds an O(head_dim²) matmul per quantize/dequantize call.
At head_dim=64, this is a 64×64 matmul per token per layer per K and V — non-trivial
overhead in a fused kernel. The benefit (improving cosine from 0.99999 toward 1.0) is
not worth the cost until quality becomes a problem.

**When to reconsider:** If INT4 quantization is attempted and shows quality degradation,
rotation could be the mitigation that makes INT4 viable. This would be a Track 2a+2c
combined experiment, deferred until after the fused kernel proves the INT8 path is
production-ready.

### Summary Table

| Idea | Classification | Rationale |
|------|---------------|-----------|
| Boundary-layer protection | **Likely unnecessary** | v3 tested, no benefit |
| Asymmetric K/V precision | **Defer** | Quality already sufficient; test after fused baseline |
| Per-head calibration | **Likely unnecessary** | No evidence of head-specific sensitivity |
| Orthogonal rotation | **Defer** | Cosine already ~1.0; rotation adds overhead |

---

## 8. Validation Plan for Fused Kernel

### Metrics and Acceptance Criteria

| Metric | Measurement | Phase B Target | Phase C Target |
|--------|-------------|----------------|----------------|
| Memory: gen overhead vs baseline | `(base_oh - fused_oh) / base_oh` | ≥40% savings | ≥40% savings |
| Memory: gen overhead vs v2 | `(v2_oh - fused_oh) / v2_oh` | Within ±5% of v2 | Within ±5% of v2 |
| Speed: decode tok/s vs baseline | `fused_tps / base_tps` | ≥50% of baseline | ≥80% of baseline |
| Speed: decode tok/s vs v2 | `fused_tps / v2_tps` | ≥2x v2 | ≥3x v2 |
| Token agreement (64 steps) | % matching argmax tokens vs baseline | ≥99% | ≥99% |
| Token agreement (256 steps) | % matching on long_decode config (256+ steps) | ≥95% | ≥95% |
| Top-1 logit agreement (64 steps) | % matching top-1 logit index | ≥99% | ≥99% |
| Logit cosine similarity (64 steps) | avg cosine over 64 steps vs baseline | ≥0.9999 | ≥0.9999 |
| Logit cosine similarity (256 steps) | avg cosine over 256 steps on long_decode | ≥0.999 | ≥0.999 |

The 64-step check is a smoke check; the 256-step check on long_decode is the readiness
gate. Both must pass. Token agreement may drop below 100% due to FP32 accumulator
arithmetic differing slightly from v2's FP16/FP32 mixed path. This is acceptable if
cosine similarity remains high and any divergence is late (step >30).

### Measurement Protocol

1. **Fresh model load per path** (baseline, v2, fused) — prevents cross-run contamination
2. **Same two configs** as v2/v3: decode_heavy (p=32, n=256), long_decode (p=256, n=512)
3. **Per-run dynamic baseline:** Generation overhead computed as
   `observed_peak_allocated_mb - memory_after_load_mb` (measured per run, not hardcoded)
4. **Warmup:** One short generation (16 tokens) before each measurement, with cleanup
5. **Synchronization:** `torch.cuda.synchronize()` before all timing boundaries

### Failure Conditions

The fused kernel should be **abandoned or redesigned** if any of:

| Condition | Interpretation |
|-----------|---------------|
| Memory overhead > v2 + 10% | Kernel introduces buffer overhead that negates compression |
| Speed < 50% of baseline after Phase C | Overhead too high; Python path may be competitive with torch.compile |
| Token agreement < 95% at step 64 | Numerical instability in the kernel |
| Token agreement < 90% at step 256 on long_decode | Drift accumulates over longer sequences |
| Cosine similarity < 0.999 at 64 steps | Dequantization or accumulation error |
| Cosine similarity < 0.99 at 256 steps | Accumulated numerical drift |
| Kernel fails on seq_len not divisible by BLOCK_KV | Block boundary handling incorrect |
| Different results on repeated runs | Non-determinism (uninitialized memory, race condition) |

### Reference Implementation for Bit-Exact Comparison

The reference path is `ChunkedInt8KVCache.chunked_attention()` in
`bio_inference_bench/kv_int8_chunked.py`. Unit tests (Phase A) should compare against
this function directly. End-to-end tests (Phase B) should compare against the full
`run_chunked_decode_step()` path with `use_triton=False`.

---

## Caveats

1. This is a design document. No kernel code has been written.
2. All performance targets are estimates based on v2/v3 measurements and architectural
   reasoning. Actual performance depends on Triton codegen quality and GPU occupancy.
3. The 20% speed regression target (Phase C) is aspirational. The minimum acceptable
   bar is 50% of baseline (Phase B).
4. ProtGPT2-specific. Do not generalize kernel design to ProGen2 or other architectures
   without separate analysis.
5. Batch=1, q_len=1 decode only. Prefill and batched inference are out of scope.

---
*Design memo produced on 2026-04-06. No implementation has started.*
*Evidence base: v2 eval (20260406T195958), v3 eval (20260406T202004), operator profiling, optimization memo.*
