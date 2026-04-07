# Phase A Completion Note: Triton INT8 KV Attention Kernel

**Date:** 2026-04-06
**Status:** PASSED — all 39 correctness tests pass
**Module:** `bio_inference_bench/triton_int8_attention.py`
**Tests:** `tests/test_triton_int8_attention.py`

---

## 1. What Was Implemented

A single Triton kernel (`_int8_kv_attention_kernel`) that computes exact attention
over INT8-quantized KV cache for single-token decode (batch=1, q_len=1).

The kernel fuses:
- INT8 → FP16 dequantization of K and V blocks
- Q × K^T dot product computation
- Online softmax accumulation (Milakov & Gimelshein 2018)
- Weighted V accumulation
- Final normalization

All operations happen in a single kernel launch per (batch, head) pair.
FP32 accumulators are used for online softmax numerics (M, L, O).

**Configuration:** BLOCK_KV=64, one Triton program per head (grid = 20 for ProtGPT2).

## 2. Correctness Results

### vs FP16 reference (standard softmax on dequantized KV)

| seq_len | Max abs diff | Cosine similarity |
|---------|-------------|-------------------|
| 1 | 0.000000 | 1.00000012 |
| 63 | 0.000488 | 1.00000000 |
| 64 | 0.000488 | 1.00000000 |
| 65 | 0.000488 | 0.99999994 |
| 128 | 0.000244 | 0.99999994 |
| 256 | 0.000244 | 0.99999994 |
| 512 | 0.000244 | 0.99999988 |
| 768 | 0.000122 | 1.00000000 |
| 1024 | 0.000122 | 1.00000000 |

**Worst case:** max_diff=0.000488 (at seq_len=63, 64, 65)
**Min cosine:** 0.99999988 (at seq_len=512)

### vs v2 chunked attention (the Python reference path)

| seq_len | Max abs diff | Cosine similarity |
|---------|-------------|-------------------|
| 1 | 0.000000 | 1.00000012 |
| 63 | 0.000488 | 0.99999976 |
| 64 | 0.000549 | 0.99999976 |
| 65 | 0.000488 | 0.99999976 |
| 128 | 0.000488 | 0.99999970 |
| 256 | 0.000366 | 0.99999970 |
| 512 | 0.000732 | 0.99999934 |
| 768 | 0.000366 | 0.99999952 |
| 1024 | 0.000366 | 0.99999940 |

**Worst case:** max_diff=0.000732 (at seq_len=512)
**Min cosine:** 0.99999934 (at seq_len=512)

The slightly larger diff vs v2 (compared to vs FP16 reference) is expected: the v2
chunked path uses FP16 intermediates in PyTorch, while the Triton kernel uses FP32
accumulators. Both paths introduce independent rounding relative to the FP16 reference,
so v2-vs-Triton has up to 2× the error of either vs the reference.

## 3. Test Coverage

| Test category | Tests | Status |
|---------------|-------|--------|
| Triton vs FP16 reference (9 seq_lens) | 9 | PASSED |
| Triton vs v2 chunked (9 seq_lens) | 9 | PASSED |
| Determinism (3 seq_lens) | 3 | PASSED |
| Output shape/dtype (3 seq_lens) | 3 | PASSED |
| No NaN/Inf (9 seq_lens) | 9 | PASSED |
| Uniform V invariant | 1 | PASSED |
| Different head counts (1, 4, 12, 20) | 4 | PASSED |
| Aggregate error report | 1 | PASSED |
| **Total** | **39** | **39 PASSED** |

## 4. Observations

- **Block-boundary edge cases work correctly.** seq_len=63 (one position short of a
  full block), seq_len=65 (one position into a second block), and seq_len=1 (single
  position) all produce correct output with proper masking.

- **Determinism confirmed.** Repeated calls with identical inputs produce bit-exact
  identical outputs across 3 tested seq_lens.

- **Head count flexibility.** The kernel works for 1, 4, 12, and 20 heads without
  modification (grid size adapts automatically).

- **Max error is well within FP16 precision.** The worst-case max_diff of 0.000732
  corresponds to ~3 ULP at the observed output magnitudes. This is expected when
  comparing FP32-accumulated online softmax (Triton) against FP16-accumulated
  chunked softmax (v2 Python path).

## 5. Caveats

1. **No performance measurement.** Phase A is correctness-only. Speed benchmarking
   is deferred to Phase B.
2. **Random inputs only.** These tests use random Gaussian KV, not actual ProtGPT2
   activations. Real model activations may have different numerical characteristics
   (outliers, scale distributions).
3. **Single-step attention only.** Tests validate one attention call in isolation,
   not multi-step decode where errors could accumulate.
4. **batch=1, q_len=1 only.** Not tested for batched or multi-query scenarios
   (out of scope for Phase A).

## 6. Phase A Exit Criterion

**MET.** All 39 correctness tests pass. The Triton kernel produces output within
FP16 precision of both the FP16 reference and the v2 Python chunked attention path
across all tested seq_len values (1 through 1024), including block-boundary edge cases.

**Ready for Phase B** (benchmark integration).

---
*Phase A completed 2026-04-06. Test runtime: 1.91s (39 tests).*
