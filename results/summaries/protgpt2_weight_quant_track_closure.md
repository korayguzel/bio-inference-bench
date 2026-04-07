# ProtGPT2 Weight Quantization Track: Closure Memo

**Date:** 2026-04-07
**Decision:** Track closed. No further ProtGPT2 weight-quantization experiments.

---

## 1. What was attempted

Three independent zero-calibration weight-quantization methods were evaluated for
ProtGPT2 (774M params, GPT-2 architecture) across three phases:

| Phase | Method | Library | Bits | Approach |
|-------|--------|---------|------|----------|
| 1 | NF4 | bitsandbytes `load_in_4bit` | 4 | Per-block NormalFloat quantization |
| 2 | LLM.int8() | bitsandbytes `load_in_8bit` | 8 | Mixed-precision outlier decomposition |
| 3 | INT8 weight-only | torchao `Int8WeightOnlyConfig` | 8 | Per-tensor symmetric INT8 |

Each method was tested in four configurations per phase:
- Weight-only (quantized weights + FP16 KV)
- Stacked (quantized weights + INT8-Triton KV)
- Against FP16 baseline and INT8-KV-only reference

Two prompt/decode scenarios: decode_heavy (prompt=32, max_new=256) and long_decode
(prompt=256, max_new=512). Fresh model load per configuration.

## 2. Three-method comparison

### Memory

| Method | Weight memory | Reduction | Peak (decode_heavy) | Peak (long_decode) |
|--------|-------------|-----------|--------------------|--------------------|
| FP16 baseline | 1489 MB | -- | 1537 MB | 1622 MB |
| NF4 (Phase 1) | 546 MB | 63% | 594 MB | 685 MB |
| bnb-int8 (Phase 2) | 817 MB | 45% | 865 MB | 951 MB |
| torchao INT8 (Phase 3) | 879 MB | 41% | 1050 MB | 1136 MB |

### Quality (vs FP16 baseline, 64 decode steps)

| Method | Token agr. (decode_heavy) | Cosine (decode_heavy) | Token agr. (long_decode) | Cosine (long_decode) |
|--------|--------------------------|----------------------|-------------------------|---------------------|
| **Target** | **>=90%** | **>=0.999** | **>=90%** | **>=0.999** |
| NF4 | 98.4% | 0.994584 | 28.1% | 0.799519 |
| bnb-int8 | 7.8% | 0.715772 | 10.9% | 0.711635 |
| torchao INT8 | 51.6% | 0.897697 | 48.4% | 0.884426 |

### Stacking with INT8 KV Triton

| Method | Decode growth unchanged? | Quality unchanged vs weight-only? |
|--------|-------------------------|----------------------------------|
| NF4 | Yes (identical to FP16 KV baseline) | Yes (INT8 KV adds zero quality loss) |
| bnb-int8 | Yes (identical to FP16 KV baseline) | Yes (INT8 KV adds zero quality loss) |
| torchao INT8 | **No (3-4x inflation from dequant buffers)** | Yes (INT8 KV adds zero quality loss) |

### Per-method failure reason

**NF4 (Phase 1):**
- Best compression (63%), fastest of the three (83 tok/s)
- Passes quality on short prompts (98.4% at prompt=32) but catastrophically fails
  on longer prompts (28.1% at prompt=256)
- Not acceptable: prompt-length-dependent quality collapse

**bitsandbytes LLM.int8() (Phase 2):**
- Moderate compression (45%), slowest of the three (36 tok/s)
- Catastrophic quality failure at all prompt lengths (7.8% token agreement)
- LLM.int8()'s outlier decomposition is miscalibrated for ProtGPT2's weight
  distributions
- Not acceptable: worst quality of all three methods

**torchao INT8 (Phase 3):**
- Moderate compression (41%), moderate speed (77 tok/s)
- Quality between the other two (51.6%) but still far below the 90% target
- Additionally breaks the stacking benefit: dequantization buffers inflate decode
  growth by 3-4x, partially negating weight savings during active generation
- Required Conv1D → nn.Linear conversion (GPT-2 architecture quirk)
- Not acceptable: fails quality gate and breaks stacking

## 3. What passed

- **Memory composability (NF4, bnb-int8):** Weight quantization and INT8 KV
  optimization compose independently — decode growth is identical regardless of
  weight quantization method (except torchao, which has dequant buffer overhead).
- **Weight reduction:** All three methods deliver meaningful weight memory reduction
  (41-63%).
- **INT8 KV quality transparency:** Adding INT8 KV on top of quantized weights adds
  zero additional quality loss in all three phases. The KV optimization layer is
  robust.
- **Experimental infrastructure:** The shared evaluation helpers (`eval_helpers.py`),
  fresh-load methodology, and dual-peak memory tracking worked correctly across
  all three phases.

## 4. Why the track is being closed

1. **All three methods fail the quality gate.** No method achieves >=90% token
   agreement and >=0.999 cosine similarity at 64 steps across both prompt scenarios.
2. **The methods represent the practical zero-calibration landscape.** NF4 (per-block
   4-bit), LLM.int8() (outlier decomposition 8-bit), and torchao (per-tensor
   symmetric 8-bit) are the three main approaches available without calibration data.
3. **Calibration-based methods (GPTQ, AWQ) are out of scope** for this project phase
   — they require domain-specific calibration datasets and pre-quantized model
   artifacts.
4. **ProtGPT2 at 1.49 GB FP16 already fits on 6+ GB GPUs.** Weight quantization is
   not a practical necessity for this model size. The INT8 KV Triton path already
   solves the dynamic memory growth problem.

## 5. Final project decision

**Canonical inference path for ProtGPT2:**
- **Weights:** FP16 (no quantization)
- **KV cache:** INT8-Triton (v5 fused kernel, validated)
- **Entry point:** `python scripts/generate_protgpt2.py --kv-mode int8-triton`

**Ruled out:**
- NF4 weight quantization for ProtGPT2 (quality failure)
- bitsandbytes LLM.int8() weight quantization for ProtGPT2 (quality failure)
- torchao INT8 weight quantization for ProtGPT2 (quality + stacking failure)
- Any further zero-calibration weight quantization experiments on ProtGPT2

**Deferred (not active):**
- Calibration-based weight quantization (GPTQ, AWQ) — separate research question
- Weight quantization for larger models (3B+) — different model, separate evaluation
- ProGen2 optimization — different architecture, separate track

## 6. What remains canonical after closure

| Component | Status |
|-----------|--------|
| FP16 weights for ProtGPT2 | **Canonical** |
| INT8-Triton KV capacity path | **Canonical** |
| Fused Triton kernel (BLOCK_KV=64) | **Canonical** |
| `generate_protgpt2.py` CLI | **Canonical** |
| `int8_generate.py` API | **Canonical** |
| Dual-peak memory tracking | **Canonical** |
| ProtGPT2 validation gate | **Canonical** |

## 7. Experimental record preserved

All raw data and reports are retained in the project history:

| Artifact | Path |
|----------|------|
| Phase 1 raw | `results/raw/weight_quant_phase1_20260406T231058.json` |
| Phase 1 report | `results/summaries/weight_quant_phase1_report.md` |
| Phase 2 raw | `results/raw/weight_quant_phase2_20260407T100546.json` |
| Phase 2 report | `results/summaries/weight_quant_phase2_report.md` |
| Phase 3 raw | `results/raw/weight_quant_phase3_20260407T103724.json` |
| Phase 3 report | `results/summaries/weight_quant_phase3_report.md` |
| Planning memo | `results/summaries/weight_quantized_track_plan.md` |
| This closure memo | `results/summaries/protgpt2_weight_quant_track_closure.md` |

---
*Closure memo completed 2026-04-07. Track closed by project decision.*
