# ProtGPT2 KV Prototype v3: Boundary-Layer Protection

Generated: 20260406T202433
Source: `results/raw/kv_prototype_v3_eval_20260406T202004.json`
v2 reference: `results/raw/kv_prototype_v2_eval_20260406T195958.json`

## Artifact Integrity

- Authoritative raw JSON: `results/raw/kv_prototype_v3_eval_20260406T202004.json`
- All numbers in this report are generated from that file.
- Isolation method: fresh model load per measurement.
- decode_heavy: baseline after_load=1485.47 MB (matches expected 1485.47 MB)
- long_decode: baseline after_load=1485.47 MB (matches expected 1485.47 MB)

**Note on isolation flag:** The raw JSON reports `isolation_passed: false`
because v2/v3 paths intentionally re-baseline `memory_after_load_mb` after
transferring prefill KV into the cache. This is a measurement design choice,
not cross-run contamination. All three paths start from the same
1485.47 MB model-weight baseline, confirmed by the
baseline path's `memory_after_load_mb` being identical across configs.
All overhead comparisons below use `observed_peak_allocated_mb - 1485.47` as the common baseline.

## 1. What Changed from v2

**v2 (all INT8):** All 36 transformer layers store KV in INT8 with chunked
dequantization during decode. Achieved -47% generation overhead vs FP16.

**v3 (boundary-layer protection):** Layers [0, 1, 34, 35] (first 2 + last 2)
keep full FP16 KV cache. The remaining 32 middle layers use v2's INT8
chunked path. Protected layers use standard SDPA for attention; INT8
layers use chunked dequantize attention.

**Hypothesis:** Boundary layers (embedding-adjacent and output-adjacent)
may be more sensitive to INT8 quantization. Protecting them could improve
cosine similarity toward 1.0, justifying the memory cost.

## 2. Memory Comparison (Common Baseline)

All overhead values computed as: `observed_peak_allocated_mb - 1485.47`

| Metric | decode_heavy Base | decode_heavy v2 | decode_heavy v3 | long_decode Base | long_decode v2 | long_decode v3 |
|--------|--------|--------|--------|--------|--------|--------|
| Peak allocated (MB) | 1536.76 | 1512.47 | 1516.08 | 1622.43 | 1557.64 | 1567.68 |
| Peak reserved (MB) | 1712.00 | 1658.00 | 1664.00 | 1856.00 | 1712.00 | 1724.00 |
| **Gen overhead (MB)** | 51.29 | 27.00 | 30.61 | 136.96 | 72.17 | 82.21 |
| **vs baseline** | --- | **-47.4%** | **-40.3%** | --- | **-47.3%** | **-40.0%** |

## 3. Cache Storage Breakdown

| Metric | decode_heavy v2 | decode_heavy v3 | long_decode v2 | long_decode v3 |
|--------|--------|--------|--------|--------|
| INT8 layers | 36 | 32 | 36 | 32 |
| FP16 protected layers | 0 | 4 | 0 | 4 |
| INT8 storage (MB) | 26.01 | 23.12 | 69.52 | 61.79 |
| FP16 protected (MB) | 0.00 | 5.61 | 0.00 | 14.98 |
| Total cache (MB) | 26.01 | 28.73 | 69.52 | 76.77 |
| FP16 equivalent (MB) | 50.45 | 50.45 | 134.82 | 134.82 |
| **Compression ratio** | 1.94x | 1.76x | 1.94x | 1.76x |

## 4. Speed Comparison

| Metric | decode_heavy Base | decode_heavy v2 | decode_heavy v3 | long_decode Base | long_decode v2 | long_decode v3 |
|--------|--------|--------|--------|--------|--------|--------|
| Decode tok/s | 128.36 | 40.29 | 44.29 | 124.91 | 21.60 | 23.80 |
| E2E tok/s | 127.67 | 40.37 | 44.37 | 124.69 | 21.63 | 23.83 |
| **vs baseline** | --- | **-68.6%** | **-65.5%** | --- | **-82.7%** | **-80.9%** |
| **v3 vs v2** | --- | --- | **+9.9%** | --- | --- | **+10.2%** |

v3 is faster than v2 because protected layers use standard SDPA instead of
chunked attention. With 4 of 36 layers bypassing the Python chunked loop,
~10% of the chunked attention overhead is eliminated.

## 5. Behavior Sanity Check

| Metric | decode_heavy v2 | decode_heavy v3 | long_decode v2 | long_decode v3 |
|--------|--------|--------|--------|--------|
| Token agreement (64 steps) | 100.0% | 100.0% | 100.0% | 100.0% |
| Top-1 logit agreement | 100.0% | 100.0% | 100.0% | 100.0% |
| Avg logit cosine sim | 0.999993 | 0.999992 | 0.999994 | 0.999994 |
| First divergence step | 64 | 64 | 64 | 64 |

## 6. Did Boundary Protection Earn Its Complexity?

### Quality

- v2 avg cosine: 0.999993
- v3 avg cosine: 0.999993
- Difference: -0.000000

**No measurable quality improvement.** Both v2 and v3 achieve 100% token
agreement and cosine similarity ~0.99999 across 64 sanity steps. Boundary-layer
protection does not improve quantization quality for ProtGPT2. The INT8
quantization noise at layers 0, 1, 34, and 35 is already sub-threshold —
protecting them provides no benefit that survives to the argmax selection.

### Memory Cost

- decode_heavy: v2 saves 47.4%, v3 saves 40.3% (boundary protection costs 7.0pp)
- long_decode: v2 saves 47.3%, v3 saves 40.0% (boundary protection costs 7.3pp)

Protecting 4 of 36 layers (11.1%) in FP16 reduces the memory benefit by
~7 percentage points. The cache compression ratio drops from 1.94x to 1.76x.

### Speed Benefit

- decode_heavy: v3 is 9.9% faster than v2 (44.3 vs 40.3 tok/s)
- long_decode: v3 is 10.2% faster than v2 (23.8 vs 21.6 tok/s)

The speed improvement comes from bypassing the slow Python chunked attention
loop for 4 layers. This is a minor benefit (~10%) against a still-dominant
3-5x speed regression vs baseline.

### Verdict

**Boundary protection does NOT earn its complexity for ProtGPT2.**

- Quality: unchanged (cosine ~0.99999 with or without protection)
- Memory: 7pp worse (47% → 40% savings)
- Speed: 10% better than v2 (but still 3-5x slower than baseline)
- Complexity: mixed FP16/INT8 per-layer logic in cache and attention

The experiment answered its question: ProtGPT2's boundary layers do not
exhibit measurably different sensitivity to INT8 quantization compared to
middle layers. This means the fused kernel (v5) does **not** need mixed-precision
layer support, simplifying its design.

## 7. Recommendation

**Drop boundary protection. Proceed with uniform INT8 for all layers.**

The v3 experiment conclusively shows that ProtGPT2 does not benefit from
boundary-layer FP16 protection. This simplifies the design for subsequent steps:

1. **v4 (optimized PyTorch):** Optimize the v2 chunked attention with
   `torch.compile()`, larger chunk sizes, and reduced per-chunk overhead.
   No mixed-precision logic needed.
2. **v5 (fused kernel):** Build a uniform INT8 dequantize-fused SDPA kernel.
   No per-layer precision dispatch needed.

The fused-kernel escalation criteria from the roadmap:
- Memory benefit confirmed: ~47% (v2) ✓
- Behavioral stability confirmed: 100% agreement at 64 steps ✓
- Speed regression quantified: 3-6x ✓
- Algorithmic improvements explored: v3 tested, no quality benefit found ✓

## Caveats

1. Sanity check covers 64 decode steps on one prompt per config — not exhaustive.
2. ProtGPT2-specific result; other architectures may have different layer sensitivity.
3. The isolation flag in raw JSON reports `false` due to intentional re-baselining
   of chunked paths, not actual cross-run contamination (see Artifact Integrity).
4. Speed regression (3-5x) remains the dominant limitation of the chunked approach.
5. Cosine similarity at 64 steps may not predict behavior at longer sequences.

---
*Report generated from `results/raw/kv_prototype_v3_eval_20260406T202004.json` on 20260406T202433.*
*Every numeric field was read directly from the raw JSON. No hand-maintained values.*
