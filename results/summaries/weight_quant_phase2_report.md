# Weight Quantization Phase 2: INT8 Weights (bitsandbytes)

Generated: 20260407T101000
Source: `results/raw/weight_quant_phase2_20260407T100546.json`
Phase 1 reference: `results/raw/weight_quant_phase1_20260406T231058.json`

## 1. Experiment

Four configurations compared with fresh model load per config:

| Config | Weights | KV Cache | Purpose |
|--------|---------|----------|---------|
| A | FP16 | FP16 | Baseline |
| C | FP16 | INT8 Triton | KV-only (v5 path) |
| E | INT8 (bnb load_in_8bit) | FP16 | Weight quantization only |
| F | INT8 (bnb load_in_8bit) | INT8 Triton | **Stacked: both layers** |

**Device placement:** All configs fully GPU-resident (`cuda:0`). bitsandbytes INT8
did not offload any parameters to CPU with `device_map="auto"`.

## 2. Memory Results

### decode_heavy (prompt=32, max_new=256)

| Metric | A (FP16+FP16) | C (FP16+INT8) | E (INT8w+FP16) | F (INT8w+INT8) |
|--------|--------------|--------------|---------------|---------------|
| Weight memory (MB) | 1489 | 1489 | 817 | 817 |
| Peak allocated (MB) | 1537 | 1512 | 865 | 840 |
| Decode growth (MB) | 42.6 | 15.1 | 42.6 | 15.1 |
| Decode tok/s | 127.9 | 74.3 | 36.4 | 31.0 |

### long_decode (prompt=256, max_new=512)

| Metric | A (FP16+FP16) | C (FP16+INT8) | E (INT8w+FP16) | F (INT8w+INT8) |
|--------|--------------|--------------|---------------|---------------|
| Weight memory (MB) | 1489 | 1489 | 817 | 817 |
| Peak allocated (MB) | 1622 | 1628 | 951 | 956 |
| Decode growth (MB) | 88.9 | 0.0 | 88.9 | 0.0 |
| Decode tok/s | 126.2 | 79.2 | 36.5 | 30.8 |

**Memory stacking works correctly.** Config F achieves:
- Weight memory: 817 MB (45% reduction from 1489 MB)
- Peak VRAM: 840-956 MB (45-41% reduction)
- Decode growth: identical to C (INT8 KV optimization is unaffected by weight quantization)

**Measured weight memory (817 MB) exceeds naive estimate (740 MB).** bitsandbytes
LLM.int8() stores per-channel absmax scales and outlier indices alongside the INT8
weights. The 77 MB overhead (10.4%) is the cost of mixed-precision decomposition metadata.

## 3. Quality Results

### decode_heavy (64 steps)

| Metric | E (INT8w+FP16) | F (INT8w+INT8) |
|--------|---------------|---------------|
| Token agreement | **7.8%** | **7.8%** |
| Top-1 logit agreement | **7.8%** | **7.8%** |
| Avg cosine similarity | **0.715772** | **0.723778** |
| First divergence step | 5 | 5 |

### long_decode (64 steps)

| Metric | E (INT8w+FP16) | F (INT8w+INT8) |
|--------|---------------|---------------|
| Token agreement | **10.9%** | **18.8%** |
| Top-1 logit agreement | **10.9%** | **18.8%** |
| Avg cosine similarity | **0.711635** | **0.765549** |
| First divergence step | 7 | 7 |

### long_decode (256 steps)

| Metric | E (INT8w+FP16) | F (INT8w+INT8) |
|--------|---------------|---------------|
| Token agreement | **3.1%** | **14.8%** |
| Avg cosine similarity | **0.715438** | **0.780089** |

**INT8 KV reference (Config C) remains perfect:** 100.0% token agreement, cos>=0.999986
across all configs. The KV optimization path is not the quality problem.

## 4. Comparison with Phase 1 NF4

| Metric (decode_heavy, 64 steps) | NF4 (Phase 1 B) | INT8w (Phase 2 E) |
|---------------------------------|-----------------|-------------------|
| Token agreement | 98.4% | **7.8%** |
| Top-1 logit agreement | 98.4% | **7.8%** |
| Avg cosine similarity | 0.994584 | **0.715772** |
| First divergence step | 4 | 5 |

| Metric (long_decode, 64 steps) | NF4 (Phase 1 B) | INT8w (Phase 2 E) |
|--------------------------------|-----------------|-------------------|
| Token agreement | 28.1% | **10.9%** |
| Avg cosine similarity | 0.799519 | **0.711635** |

| Metric | NF4 (Phase 1 B) | INT8w (Phase 2 E) |
|--------|-----------------|-------------------|
| Weight memory (MB) | 546 | 817 |
| Weight reduction vs FP16 | 63% | 45% |
| Peak (decode_heavy) | 594 MB | 865 MB |
| Decode tok/s (decode_heavy) | 83.3 | 36.4 |

**bitsandbytes INT8 is worse than NF4 on every quality metric while also being
larger, slower, and less compressed.** This is the opposite of the expected outcome.

## 5. Analysis

### Why is INT8 (8-bit) worse than NF4 (4-bit)?

The result is counterintuitive: 8-bit quantization (256 levels) should be more precise
than 4-bit (16 levels). The explanation is that these are fundamentally different
quantization algorithms:

- **NF4 (bitsandbytes `load_in_4bit`):** Applies per-block NormalFloat quantization
  uniformly to all weight channels. Every channel gets the same treatment.

- **LLM.int8() (bitsandbytes `load_in_8bit`):** Applies mixed-precision decomposition
  via the algorithm from Dettmers et al. (2022). It identifies "outlier features"
  (columns with magnitude > threshold, default 6.0) and keeps those in FP16 while
  quantizing the rest to INT8. The quality depends critically on the outlier threshold
  correctly identifying which features carry information.

For ProtGPT2, LLM.int8()'s outlier detection appears catastrophically miscalibrated:
- Divergence begins at step 5-7 (during very early decode)
- Cosine similarity is 0.71-0.78 (far below the 0.99+ expected for 8-bit)
- Quality is worse than random top-1 agreement at longer contexts

This suggests ProtGPT2's weight distributions do not have the outlier structure that
LLM.int8() assumes. The algorithm was designed for large (6B+) transformer LMs where
a small fraction of hidden dimensions carry outsized activation magnitudes. ProtGPT2
at 774M params may have a more uniform weight distribution, causing the decomposition
to either miss critical features or create numerical instability in the reconstruction.

### Memory: stacking still works

The composability finding from Phase 1 holds:
- E's decode growth matches A exactly (42.6 / 88.9 MB) — INT8 weights don't affect FP16 KV growth
- F's decode growth matches C exactly (15.1 / 0.0 MB) — INT8 weights don't affect INT8 KV growth
- F's weight memory matches E exactly (817 MB) — INT8 KV doesn't affect weight footprint

### Speed: 71-76% regression

| Config | decode_heavy tok/s | long_decode tok/s | vs baseline |
|--------|-------------------|-------------------|-------------|
| A (FP16+FP16) | 127.9 | 126.2 | -- |
| C (FP16+INT8) | 74.3 | 79.2 | -37% to -42% |
| E (INT8w+FP16) | 36.4 | 36.5 | **-71%** |
| F (INT8w+INT8) | 31.0 | 30.8 | **-76%** |

LLM.int8() is significantly slower than NF4 (36 tok/s vs 83 tok/s) due to the
outlier decomposition overhead: each linear layer requires splitting inputs, running
separate FP16 and INT8 matmuls, and merging outputs.

## 6. Success Criteria Assessment

| Criterion | Target | Result | Status |
|-----------|--------|--------|--------|
| Weight memory reduction | >=50% | 45% (1489 -> 817 MB) | **FAILED** |
| Total VRAM < 800 MB | < 800 MB | 840-956 MB | **FAILED** |
| Decode growth unchanged | Within 10% of v5 | Identical (0%) | **PASSED** |
| Token agreement >=90% (64 steps) | >=90% | **7.8-18.8%** | **FAILED** |
| Cosine >=0.999 (64 steps) | >=0.999 | **0.71-0.77** | **FAILED** |
| Decode tok/s >=40% of baseline | >=51 tok/s | 31-36 tok/s | **FAILED** |

**5 of 6 criteria failed. Only decode growth composability passes.**

## 7. Decision

**Does bnb-int8 become the canonical weight layer: NO.**

bitsandbytes `load_in_8bit` (LLM.int8()) is categorically unsuitable for ProtGPT2:
- Quality catastrophically fails (cos=0.72, far below the 0.999 target)
- Worse quality than the already-rejected NF4 method
- Larger weight footprint than NF4 (817 MB vs 546 MB)
- Slower than NF4 (36 tok/s vs 83 tok/s)
- Fails to meet the VRAM target (840 MB > 800 MB)

**Both bitsandbytes quantization methods fail the quality gate for ProtGPT2.**

## 8. Recommendation

The bitsandbytes library provides two weight quantization methods, and both fail
for ProtGPT2:

| Method | Quality (cos, decode_heavy 64 steps) | Weight MB | Verdict |
|--------|--------------------------------------|-----------|---------|
| NF4 (Phase 1) | 0.995 (borderline) | 546 | Failed on long_decode |
| LLM.int8() (Phase 2) | **0.716** (catastrophic) | 817 | Failed everywhere |

**Next options (if weight quantization is still pursued):**

1. **`torch.ao.quantization.quantize_dynamic` (INT8):** PyTorch's built-in dynamic
   weight-only INT8 quantization. Uses per-tensor or per-channel symmetric INT8
   without outlier decomposition. May behave better for ProtGPT2's weight distributions.
   Zero additional dependencies.

2. **GPTQ with calibration data:** Calibration-based 4-bit quantization that optimizes
   quantization parameters using representative protein sequences. Higher setup cost
   but much better quality than zero-shot methods.

3. **Accept FP16 weights for ProtGPT2.** At 1.49 GB, ProtGPT2 fits comfortably on
   any 4+ GB GPU. Weight quantization is a stronger motivation for larger models
   (3B+) where FP16 doesn't fit. The INT8 KV Triton path already eliminates the
   dynamic memory problem.

## Caveats

1. LLM.int8()'s outlier threshold (default 6.0) was not tuned. A different threshold
   might improve quality, but the magnitude of the failure (cos=0.72) suggests the
   fundamental decomposition approach is wrong for ProtGPT2, not just miscalibrated.
2. These results are specific to ProtGPT2 (774M, GPT-2 architecture). LLM.int8()
   may work well for larger models with different weight distributions.
3. `torch.ao` dynamic INT8 uses a completely different quantization strategy and
   should not be assumed to fail just because bitsandbytes INT8 failed.

---
*Report generated from `results/raw/weight_quant_phase2_20260407T100546.json`.*
*Phase 1 NF4 values from `results/raw/weight_quant_phase1_20260406T231058.json`.*
*All values read directly from raw JSON. No hand-maintained numbers.*
