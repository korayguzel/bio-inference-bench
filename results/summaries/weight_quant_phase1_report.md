# Weight Quantization Phase 1: NF4 Four-Way Comparison

Generated: 20260406T231200
Source: `results/raw/weight_quant_phase1_20260406T231058.json`

## 1. Experiment

Four configurations compared with fresh model load per config:

| Config | Weights | KV Cache | Purpose |
|--------|---------|----------|---------|
| A | FP16 | FP16 | Baseline |
| B | NF4 (bnb 4-bit) | FP16 | Weight quantization only |
| C | FP16 | INT8 Triton | KV optimization only (v5) |
| D | NF4 (bnb 4-bit) | INT8 Triton | **Stacked: both layers** |

## 2. Memory Results

### decode_heavy (prompt=32, max_new=256)

| Metric | A (FP16+FP16) | B (NF4+FP16) | C (FP16+INT8) | D (NF4+INT8) |
|--------|--------------|-------------|--------------|-------------|
| Weight memory (MB) | 1489 | 546 | 1489 | 546 |
| Peak allocated (MB) | 1537 | 594 | 1512 | 569 |
| Decode growth (MB) | 42.6 | 42.6 | 15.1 | 15.1 |
| Decode tok/s | 127.9 | 83.3 | 74.8 | 59.9 |

### long_decode (prompt=256, max_new=512)

| Metric | A (FP16+FP16) | B (NF4+FP16) | C (FP16+INT8) | D (NF4+INT8) |
|--------|--------------|-------------|--------------|-------------|
| Weight memory (MB) | 1489 | 546 | 1489 | 546 |
| Peak allocated (MB) | 1622 | 680 | 1628 | 685 |
| Decode growth (MB) | 88.9 | 89.0 | 0.0 | 0.0 |
| Decode tok/s | 126.6 | 82.4 | 80.1 | 60.0 |

**Memory stacking works.** Config D achieves:
- Weight memory: 546 MB (63% reduction from 1489 MB)
- Peak VRAM: 569-685 MB (58-63% reduction from 1537-1622 MB)
- Decode growth: identical to C (INT8 KV optimization is unaffected by weight quantization)

## 3. Quality Results

### decode_heavy (64 steps)

| Metric | B (NF4+FP16) | C (FP16+INT8) | D (NF4+INT8) |
|--------|-------------|--------------|-------------|
| Token agreement | 98.4% | 100.0% | 98.4% |
| Top-1 logit agreement | 98.4% | 100.0% | 98.4% |
| Avg cosine similarity | 0.994584 | 0.999995 | 0.994641 |

### long_decode (64 steps)

| Metric | B (NF4+FP16) | C (FP16+INT8) | D (NF4+INT8) |
|--------|-------------|--------------|-------------|
| Token agreement | 28.1% | 100.0% | 28.1% |
| Top-1 logit agreement | 28.1% | 100.0% | 28.1% |
| Avg cosine similarity | 0.799519 | 0.999994 | 0.799523 |

### long_decode (256 steps)

| Metric | B (NF4+FP16) | C (FP16+INT8) | D (NF4+INT8) |
|--------|-------------|--------------|-------------|
| Token agreement | 28.1% | 100.0% | 28.5% |
| Avg cosine similarity | 0.836587 | 0.999986 | 0.837051 |

## 4. Analysis

### Memory: both layers compose cleanly

Weight quantization and KV optimization are independent layers that stack without
interference:
- B's decode growth matches A exactly (42.6 / 89.0 MB) — NF4 weights don't affect FP16 KV growth
- D's decode growth matches C exactly (15.1 / 0.0 MB) — NF4 weights don't affect INT8 KV growth
- D's weight memory matches B exactly (546 MB) — INT8 KV doesn't affect weight footprint

The stacking is arithmetic: D = B's weight footprint + C's decode growth.

### Quality: NF4 is too aggressive for ProtGPT2

**NF4 weight quantization introduces substantial quality degradation:**
- At 64 steps on decode_heavy: 98.4% token agreement, cos=0.995 — borderline acceptable
- At 64 steps on long_decode: **28.1% token agreement, cos=0.80** — below failure threshold

The quality degradation is entirely from NF4 weights:
- D (NF4+INT8) has the same quality as B (NF4+FP16): cos 0.994641 vs 0.994584
- INT8 KV adds zero additional quality loss on top of NF4 weights

The prompt-length dependence (98.4% at prompt=32 vs 28.1% at prompt=256) suggests
that NF4 quantization noise accumulates through the prefill pass — a longer prompt
feeds more quantization errors into the KV cache and initial hidden state, causing
greater divergence during decode.

### Speed: NF4 is slower (35% regression)

| Config | decode_heavy tok/s | long_decode tok/s | vs baseline |
|--------|-------------------|-------------------|-------------|
| A (FP16+FP16) | 127.9 | 126.6 | — |
| B (NF4+FP16) | 83.3 | 82.4 | -35% |
| C (FP16+INT8) | 74.8 | 80.1 | -37% to -41% |
| D (NF4+INT8) | 59.9 | 60.0 | -53% |

NF4 weights are slower than FP16 due to dequantization overhead in bitsandbytes'
Linear4bit modules. This is expected for a capacity optimization, not a speed concern.

## 5. Phase 1 Success Criteria Assessment

| Criterion | Target | Result | Status |
|-----------|--------|--------|--------|
| Weight memory reduction | >=60% | 63% (1489 → 546 MB) | **PASSED** |
| Total VRAM < 800 MB | < 800 MB | 569-685 MB | **PASSED** |
| Decode growth unchanged | Within 10% of v5 | Identical (0%) | **PASSED** |
| Token agreement >=90% (64 steps) | >=90% | 98.4% (decode_heavy), **28.1% (long_decode)** | **FAILED** |
| Cosine >=0.999 (64 steps) | >=0.999 | 0.995 (decode_heavy), **0.80 (long_decode)** | **FAILED** |

**Memory targets met. Quality targets failed on long_decode.**

## 6. Recommendation

**NF4 weight quantization fails the quality gate for ProtGPT2 at longer prompts.**

The memory stacking is validated — weight quantization and KV optimization compose
cleanly without interference. But NF4 (4-bit) is too aggressive for ProtGPT2's
weight distributions, particularly for longer prompt contexts.

**Next step: try INT8 weight quantization (Phase 2).**

INT8 weights would give:
- ~50% weight reduction (1489 → ~740 MB) instead of NF4's 63%
- Likely much better quality (8-bit has 256 quantization levels vs NF4's 16)
- Options: `bitsandbytes load_in_8bit=True` or `torch.ao.quantization.quantize_dynamic`

The quality floor is clear: NF4's 4-bit resolution is insufficient for ProtGPT2.
INT8 weight quantization (via `load_in_8bit=True`) is the natural next test — it
provides a less aggressive compression point that may preserve quality while still
delivering meaningful weight reduction.

## Caveats

1. Quality at 64 steps on decode_heavy (98.4%, cos=0.995) is borderline — it may
   be acceptable for some use cases but not for high-fidelity generation.
2. The prompt-length sensitivity (98% at 32 tokens vs 28% at 256 tokens) suggests
   NF4 quality depends on prompt complexity, not just decode length.
3. Speed regression (35-53%) is not a concern for a capacity track but should be
   noted for user expectations.
4. bitsandbytes NF4 is one specific 4-bit implementation — other 4-bit methods
   (GPTQ, AWQ) might perform better with calibration, but are deferred.

---
*Report generated from `results/raw/weight_quant_phase1_20260406T231058.json`.*
*All values read directly from raw JSON. No hand-maintained numbers.*
