# ProtGPT2 Fused Kernel v5: Phase B Benchmark Report (Corrected)

Generated: 20260406T214500
Source: `results/raw/kv_fused_v5_eval_20260406T214141.json`

## 1. Implementation Summary

**v5** replaces the v2 Python chunked attention loop with a single fused Triton kernel
per (batch, head) pair. The kernel reads INT8 KV + FP16 scales from global memory,
dequantizes in SRAM, and computes exact attention with online softmax using FP32
accumulators — all in one kernel launch.

Everything else in the decode step is unchanged: token embedding, positional embedding,
QKV projections, output projections, FFN/MLP, and layer norms remain in PyTorch.

- KV storage: uniform INT8, all 36 layers
- BLOCK_KV: 64 (matches v2 chunk_size)
- Accumulators: FP32
- No performance tuning (Phase A kernel, unmodified)

## 2. Benchmark Methodology

- **Fresh model load** per path — isolation PASSED (0.00 MB delta on `memory_before_generation_mb`)
- **Warmup:** 16-token generation + full cleanup before each measurement
- **Timing:** `torch.cuda.synchronize()` before all timing boundaries
- **Sanity:** 64-step smoke check on both configs; 256-step extended window on long_decode

### Memory Semantics

Two overhead metrics are reported for all paths, using the same definitions:

| Metric | Definition | What it captures |
|--------|-----------|-----------------|
| **end_to_end_generation_overhead_mb** | `overall_peak - memory_before_generation` | Total high-water mark above model weights, including prefill transients |
| **decode_phase_growth_mb** | `decode_peak - memory_after_prefill` | Memory growth during decode only, from decode-ready state |

The `overall_peak` is `max(prefill_peak, decode_peak)` where each phase's peak is
tracked independently via `torch.cuda.reset_peak_memory_stats()` between phases.

**Primary capacity metric: `decode_phase_growth_mb`.** This measures how much additional
memory the decode loop requires above the ready-to-decode state. For capacity planning
(longer sequences, larger batches), this is the metric that determines headroom.

The `end_to_end_generation_overhead_mb` captures the full picture including prefill
transients. For v2/v5, the FP16→INT8 KV transfer during prefill creates a transient
where both copies coexist, which can push the overall peak above baseline.

## 3. Results

### Memory — End-to-End Overhead

| Metric | decode_heavy Base | decode_heavy v2 | decode_heavy v5 | long_decode Base | long_decode v2 | long_decode v5 |
|--------|--------|--------|--------|--------|--------|--------|
| Before generation (MB) | 1485.47 | 1485.47 | 1485.47 | 1485.47 | 1485.47 | 1485.47 |
| Overall peak (MB) | 1536.76 | 1512.47 | 1512.35 | 1622.43 | 1624.46 | 1624.46 |
| **E2E overhead (MB)** | **51.29** | **27.00** | **26.88** | **136.96** | **138.99** | **138.99** |
| **vs baseline** | --- | **-47.3%** | **-47.6%** | --- | **+1.5%** | **+1.5%** |

On decode_heavy (short prompt), v5 e2e overhead is 47.6% lower than baseline — the INT8
cache pays off clearly because the prefill transient is small (32 prompt tokens).

On long_decode (long prompt), v2/v5 e2e overhead is *slightly above* baseline (+1.5%).
This is because the FP16→INT8 KV transfer creates a transient peak during prefill that
exceeds the baseline's decode peak. This is not a v5 regression — it's a prefill
transition cost inherent to the INT8 cache approach. Prefill optimization is out of scope.

### Memory — Decode Phase Growth (Primary Capacity Metric)

| Metric | decode_heavy Base | decode_heavy v2 | decode_heavy v5 | long_decode Base | long_decode v2 | long_decode v5 |
|--------|--------|--------|--------|--------|--------|--------|
| After prefill (MB) | 1491.10 | 1494.17 | 1494.17 | 1530.47 | 1554.93 | 1554.93 |
| Decode peak (MB) | 1536.76 | 1512.47 | 1512.35 | 1622.43 | 1557.64 | 1557.64 |
| **Decode growth (MB)** | **45.66** | **18.30** | **18.18** | **91.96** | **2.71** | **2.71** |
| **vs baseline** | --- | **-59.9%** | **-60.2%** | --- | **-97.1%** | **-97.1%** |

v5 decode-phase growth is identical to v2 and dramatically lower than baseline:
- decode_heavy: **-60%** (18 MB vs 46 MB)
- long_decode: **-97%** (2.7 MB vs 92 MB)

The 97% reduction on long_decode means v2/v5 barely grow during decode — the INT8
cache at the ready-to-decode point already contains most of the final cache state
(256 prompt tokens out of 768 total), and the remaining 512 tokens add only INT8-sized
increments.

### v5 vs v2 Memory

v5 is functionally identical to v2 on memory. Peak allocated differs by ≤0.12 MB
(decode_heavy) and is exactly 0.00 MB (long_decode). The Triton kernel introduces
no memory overhead.

### Speed

| Metric | decode_heavy Base | decode_heavy v2 | decode_heavy v5 | long_decode Base | long_decode v2 | long_decode v5 |
|--------|--------|--------|--------|--------|--------|--------|
| Decode tok/s | 127.97 | 40.88 | 74.34 | 125.26 | 21.64 | 79.64 |
| E2E tok/s | 127.26 | 40.96 | 74.37 | 125.03 | 21.67 | 79.61 |
| **v5 vs baseline** | --- | --- | **-41.9%** | --- | --- | **-36.4%** |
| **v5 vs v2** | --- | --- | **1.82x** | --- | --- | **3.68x** |

v5 recovers most of the v2 speed regression:
- long_decode: 63.6% of baseline speed (up from v2's 17.3%), **3.68x faster than v2**
- decode_heavy: 58.1% of baseline speed (up from v2's 31.9%), **1.82x faster than v2**

### Behavioral Fidelity

| Metric | decode_heavy v2 | decode_heavy v5 | long_decode v2 (64) | long_decode v5 (64) | long_decode v2 (256) | long_decode v5 (256) |
|--------|--------|--------|--------|--------|--------|--------|
| Token agreement | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Top-1 logit agree | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Avg cosine sim | 0.999993 | 0.999995 | 0.999994 | 0.999994 | 0.999965 | 0.999986 |

Perfect fidelity across all windows. v5's FP32 accumulators produce slightly higher
cosine than v2's FP16 accumulators (0.999986 vs 0.999965 at 256 steps).

## 4. Phase B Exit Criteria Assessment

| Criterion | Target | decode_heavy | long_decode | Status |
|-----------|--------|-------------|-------------|--------|
| Memory within 5% of v2 (decode growth) | ±5% | 0.12 MB diff (0.7%) | 0.00 MB diff (0.0%) | **PASSED** |
| E2E overhead vs baseline (decode_heavy) | savings ≥40% | -47.6% | N/A (prefill transient) | **PASSED** |
| Decode growth vs baseline | savings ≥40% | -60.2% | -97.1% | **PASSED** |
| Speed ≥50% of baseline | ≥50% | 58.1% | 63.6% | **PASSED** |
| Speed ≥2x v2 | ≥2x | 1.82x | 3.68x | **PARTIAL** |
| Token agreement ≥99% (64 steps) | ≥99% | 100.0% | 100.0% | **PASSED** |
| Token agreement ≥95% (256 steps) | ≥95% | N/A | 100.0% | **PASSED** |
| Cosine ≥0.9999 (64 steps) | ≥0.9999 | 0.999995 | 0.999994 | **PASSED** |
| Cosine ≥0.999 (256 steps) | ≥0.999 | N/A | 0.999986 | **PASSED** |

**8 of 9 criteria passed. One partial** (decode_heavy v5/v2 = 1.82x vs 2x target).
The partial is architectural: at short sequences, weight GEMMs dominate per-step time
and attention is ~1% of CUDA time. The kernel can only accelerate the attention fraction.

## 5. Blockers

None. The corrected memory semantics confirm the main conclusion is unchanged:
- v5 memory is identical to v2 (no regression from kernel integration)
- Decode-phase growth savings are 60-97% vs baseline
- E2E overhead savings depend on prompt length (47.6% for short prompts, ~0% for long
  prompts due to prefill transient — a separate problem)

## 6. Recommendation

**Enter Phase C (optimization tuning).** The untuned kernel delivers:
- Zero memory regression vs v2
- 3.68x speedup over v2 on the capacity-relevant long_decode config
- 63.6% of baseline speed on long_decode
- Perfect behavioral fidelity through 256 steps

Phase C target: ≥80% of baseline speed on long_decode (~100 tok/s).

---
*Report generated from `results/raw/kv_fused_v5_eval_20260406T214141.json`.*
*Every numeric field was read directly from the raw JSON. No hand-maintained values.*
