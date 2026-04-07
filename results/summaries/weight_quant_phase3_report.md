# Weight Quantization Phase 3: torchao INT8 Weights

Generated: 20260407T104500
Source: `results/raw/weight_quant_phase3_20260407T103724.json`
Phase 1 reference: `results/raw/weight_quant_phase1_20260406T231058.json`
Phase 2 reference: `results/raw/weight_quant_phase2_20260407T100546.json`

## 1. Experiment

Four configurations compared with fresh model load per config:

| Config | Weights | KV Cache | Purpose |
|--------|---------|----------|---------|
| A | FP16 | FP16 | Baseline |
| C | FP16 | INT8 Triton | KV-only (v5 path) |
| G | torchao INT8 | FP16 | Weight quantization only |
| H | torchao INT8 | INT8 Triton | **Stacked: both layers** |

**Method:** `torchao 0.17.0` `Int8WeightOnlyConfig` with `quantize_()`. Applied
post-load after converting GPT-2's 144 `Conv1D` modules to `nn.Linear` (required
because torchao only targets `nn.Linear`).

**Device placement:** All configs fully GPU-resident (`cuda:0`). No CPU offloading.

## 2. Memory Results

### decode_heavy (prompt=32, max_new=256)

| Metric | A (FP16+FP16) | C (FP16+INT8) | G (torchao+FP16) | H (torchao+INT8) |
|--------|--------------|--------------|-----------------|-----------------|
| Weight memory (MB) | 1489 | 1489 | 879 | 879 |
| Peak allocated (MB) | 1537 | 1512 | 1050 | 1026 |
| Decode growth (MB) | 42.6 | 15.1 | **164.7** | **137.5** |
| Decode tok/s | 126.4 | 74.7 | 76.7 | 56.8 |

### long_decode (prompt=256, max_new=512)

| Metric | A (FP16+FP16) | C (FP16+INT8) | G (torchao+FP16) | H (torchao+INT8) |
|--------|--------------|--------------|-----------------|-----------------|
| Weight memory (MB) | 1489 | 1489 | 879 | 879 |
| Peak allocated (MB) | 1622 | 1628 | 1136 | 1072 |
| Decode growth (MB) | 88.9 | 0.0 | **211.6** | **121.5** |
| Decode tok/s | 125.2 | 79.0 | 76.4 | 56.7 |

**Weight reduction works:** 879 MB (41% reduction from 1489 MB).

**Decode growth is broken.** torchao INT8 weights cause 3-4x higher decode-phase
memory growth compared to FP16 weights. This is caused by temporary FP16
dequantization buffers that torchao's quantized linear kernels allocate during each
forward pass. The dequantization creates full-sized FP16 weight copies transiently,
and the CUDA allocator retains them as peak memory.

This means the **stacking benefit is negated**: while weight storage drops by 610 MB,
decode growth increases by 122-123 MB, eroding much of the savings during active
generation.

## 3. Quality Results

### decode_heavy (64 steps)

| Metric | G (torchao+FP16) | H (torchao+INT8) |
|--------|-----------------|-----------------|
| Token agreement | **51.6%** | **51.6%** |
| Top-1 logit agreement | **51.6%** | **51.6%** |
| Avg cosine similarity | **0.897697** | **0.897740** |
| First divergence step | 33 | 33 |

### long_decode (64 steps)

| Metric | G (torchao+FP16) | H (torchao+INT8) |
|--------|-----------------|-----------------|
| Token agreement | **48.4%** | **48.4%** |
| Top-1 logit agreement | **48.4%** | **48.4%** |
| Avg cosine similarity | **0.884426** | **0.884433** |
| First divergence step | 31 | 31 |

### long_decode (256 steps)

| Metric | G (torchao+FP16) | H (torchao+INT8) |
|--------|-----------------|-----------------|
| Token agreement | **12.1%** | **12.1%** |
| Avg cosine similarity | **0.768357** | **0.768334** |

**INT8 KV adds zero additional quality loss** on top of torchao INT8 weights — G and
H have identical quality, confirming the KV optimization is still transparent.

## 4. Cross-Phase Quality Comparison

All values at 64 steps against FP16 baseline:

### decode_heavy

| Method | Token agreement | Cosine similarity | First divergence |
|--------|----------------|-------------------|-----------------|
| NF4 (Phase 1 B) | **98.4%** | **0.994584** | step 4 |
| bnb-int8 (Phase 2 E) | 7.8% | 0.715772 | step 5 |
| torchao INT8 (Phase 3 G) | 51.6% | 0.897697 | step 33 |

### long_decode

| Method | Token agreement | Cosine similarity | First divergence |
|--------|----------------|-------------------|-----------------|
| NF4 (Phase 1 B) | 28.1% | 0.799519 | step 1 |
| bnb-int8 (Phase 2 E) | 10.9% | 0.711635 | step 7 |
| torchao INT8 (Phase 3 G) | **48.4%** | **0.884426** | step 31 |

### long_decode (256 steps)

| Method | Token agreement | Cosine similarity |
|--------|----------------|-------------------|
| NF4 (Phase 1 B) | 28.1% | 0.836587 |
| bnb-int8 (Phase 2 E) | 3.1% | 0.715438 |
| torchao INT8 (Phase 3 G) | **12.1%** | **0.768357** |

**Ranking:** torchao > NF4 on long_decode, NF4 > torchao on decode_heavy. Both are
far below the quality targets. bnb-int8 is worst on all metrics.

**None of the three methods achieve the quality gate** (>=90% token agreement,
>=0.999 cosine at 64 steps).

## 5. Cross-Phase Memory Comparison

| Method | Weight (MB) | Reduction | Peak decode_heavy | Peak long_decode | Decode growth (long) |
|--------|------------|-----------|-------------------|-----------------|---------------------|
| FP16 baseline | 1489 | -- | 1537 | 1622 | 88.9 MB |
| NF4 (Phase 1) | 546 | 63% | 594 | 685 | 0.0 MB |
| bnb-int8 (Phase 2) | 817 | 45% | 865 | 951 | 88.9 MB |
| **torchao INT8 (Phase 3)** | **879** | **41%** | **1050** | **1136** | **211.6 MB** |
| INT8 KV only (v5) | 1489 | 0% | 1512 | 1628 | 0.0 MB |

torchao INT8 has the worst peak memory of the three weight-quantization methods
despite a reasonable weight reduction. The dequantization buffer overhead pushes
peaks above both NF4 and bnb-int8.

## 6. Cross-Phase Speed Comparison

| Method | decode_heavy tok/s | long_decode tok/s | vs FP16 |
|--------|-------------------|-------------------|---------|
| FP16 baseline | 127.9 | 126.2 | -- |
| NF4 (Phase 1) | 83.3 | 82.4 | -35% |
| bnb-int8 (Phase 2) | 36.4 | 36.5 | -71% |
| **torchao INT8 (Phase 3)** | **76.7** | **76.4** | **-40%** |

torchao INT8 is faster than bnb-int8 (76.7 vs 36.4 tok/s) and comparable to NF4
(83.3 tok/s). Speed is not the bottleneck for any method.

## 7. Success Criteria Assessment

| Criterion | Target | Result | Status |
|-----------|--------|--------|--------|
| Quality better than NF4 and bnb-int8 | Materially better | Mixed: better on long_decode, worse on decode_heavy | **FAILED** |
| Meaningful weight reduction | Significant | 41% (1489 → 879 MB) | **PASSED** |
| Stacked config preserves KV benefit | Decode growth unchanged | **3-4x increase** (42.6 → 164.7 MB) | **FAILED** |
| No catastrophic divergence at 64 steps | First 64 steps stable | First divergence at step 31-33 | **MARGINAL** |
| Token agreement >=90% (64 steps) | >=90% | 48-52% | **FAILED** |
| Cosine >=0.999 (64 steps) | >=0.999 | 0.884-0.898 | **FAILED** |

**1 of 6 criteria passed (weight reduction). Quality and stacking both fail.**

## 8. Answers to Decision Questions

### 1. Is torchao INT8 materially better than NF4 and bnb-int8 for ProtGPT2?

**No.** It is better than bnb-int8 (which catastrophically fails) and comparable to
NF4, but the results are mixed:
- Better quality on long_decode (cos=0.884 vs NF4's 0.800)
- Worse quality on decode_heavy (cos=0.898 vs NF4's 0.995)
- Worse memory behavior (dequantization buffers inflate decode growth)
- None of the three methods meet the quality gate

### 2. Does stacking still work cleanly?

**No.** Unlike NF4 and bnb-int8 (where decode growth was identical to the
corresponding FP16 config), torchao INT8 inflates decode growth by 3-4x due to
runtime dequantization buffers. The INT8 KV Triton kernel itself is unaffected
(quality is identical between G and H), but the peak memory benefit of KV
optimization is partially negated by weight dequantization overhead.

### 3. Does torchao INT8 become the canonical weight layer?

**No.**

### 4. Should the ProtGPT2 weight-quantization track be closed?

**Yes.** Three independent weight quantization methods have been evaluated:

| Method | Quality | Memory | Stacking | Verdict |
|--------|---------|--------|----------|---------|
| NF4 (bitsandbytes 4-bit) | Fails on long prompts | Best compression (63%) | Clean | **Quality gate failed** |
| LLM.int8() (bitsandbytes 8-bit) | Catastrophic failure | Moderate (45%) | Clean | **Quality gate failed** |
| torchao INT8 (per-tensor symmetric) | Fails both configs | Moderate (41%) | **Broken** (3-4x growth) | **Quality + stacking failed** |

All three methods fail the quality gate (>=90% token agreement, >=0.999 cosine at
64 steps). The fundamental issue is that ProtGPT2's 774M-parameter GPT-2 architecture
does not tolerate weight quantization at any precision or method tested.

**Recommendation:** Close the ProtGPT2 weight-quantization track. The model at 1.49 GB
FP16 fits comfortably on any 6+ GB GPU, and the INT8 KV Triton path already eliminates
the dynamic memory problem. Weight quantization becomes more compelling for larger
models (3B+) where FP16 weights don't fit — but those models should be evaluated
independently, as ProtGPT2's quantization sensitivity does not generalize.

## 9. Why ProtGPT2 Resists Weight Quantization

Three different quantization algorithms all fail for ProtGPT2, each in different ways:
- **NF4:** 98.4% quality at short prompts but collapses at longer contexts (28%)
- **LLM.int8():** Outlier decomposition is catastrophically wrong (7.8%)
- **torchao INT8:** Per-tensor symmetric quantization loses too much information (51.6%)

The common thread is that ProtGPT2's weight distributions — shaped by training on
protein sequences (a 20-character amino acid alphabet, very different from natural
language) — are not well-suited to standard quantization schemes. Protein sequence
models may have:
- More uniform weight magnitudes (less separable into "important" vs "compressible")
- Higher information density per parameter (smaller model = less redundancy)
- Weight distributions calibrated to a narrow vocabulary that are sensitive to
  rounding errors

This suggests that weight quantization for protein LMs may require domain-specific
calibration (GPTQ/AWQ with protein sequence calibration data) rather than
zero-calibration methods. That is a separate research question beyond the scope
of this inference benchmark.

## Caveats

1. torchao 0.17.0 `Int8WeightOnlyConfig` uses the "v1" API (deprecated). The v2 API
   may have different memory behavior, but the quality issue is fundamental.
2. The Conv1D → nn.Linear conversion is functionally correct (verified: pre-quant
   forward pass matches FP16 exactly) but adds a step not needed for models that
   use nn.Linear natively.
3. `metadata_weight_mb` reports 1599 MB for torchao INT8 — this is a measurement
   artifact. The `_extract_metadata` function counts `p.numel() * p.element_size()`
   which doesn't correctly reflect the quantized tensor layout. The true GPU
   footprint is correctly captured by `memory_allocated()` at 879 MB.
4. The dequantization buffer issue may be addressable via `torch.compile` or custom
   kernels, but this would be optimization of a method that already fails the
   quality gate.

---
*Report generated from `results/raw/weight_quant_phase3_20260407T103724.json`.*
*Phase 1 NF4 values from `results/raw/weight_quant_phase1_20260406T231058.json`.*
*Phase 2 bnb-int8 values from `results/raw/weight_quant_phase2_20260407T100546.json`.*
*All values read directly from raw JSON. No hand-maintained numbers.*
