# Weight-Quantized Inference Track: Planning Memo

> **CLOSED — 2026-04-07.** All three planned weight-quantization methods (NF4,
> bitsandbytes LLM.int8(), torchao INT8) failed the quality gate for ProtGPT2.
> Track closed by project decision. See
> `results/summaries/protgpt2_weight_quant_track_closure.md` for the full closure
> memo. The canonical ProtGPT2 path remains FP16 weights + INT8-Triton KV.

**Date:** 2026-04-06
**Context:** ProtGPT2 v5 INT8 KV capacity path is validated and productized.
**Decision:** ~~Open a weight-quantized model track as the next active work.~~
Opened and completed. Track closed after three phases of evaluation.

---

## 1. What is already solved by the KV-capacity path?

The v5 INT8 KV path solves **decode-time memory growth:**

| Metric | FP16 Baseline | INT8-Triton v5 |
|--------|--------------|----------------|
| Decode growth/token (long_decode) | 0.18 MB | 0.005 MB |
| Decode growth for 512 new tokens | 92 MB | 3 MB |
| Decode peak | 1622 MB | 1558 MB |

The KV cache is no longer the memory constraint during decode. At 512 tokens, the
INT8 KV path adds only 3 MB of decode growth — negligible.

**What the KV path does NOT solve:** The model weights themselves. ProtGPT2 in FP16
occupies 1485 MB (1.45 GB) before any generation begins. On a 12 GB RTX 4070 this
is fine, but on smaller GPUs (6-8 GB) or for larger models, the weight footprint
is the dominant constraint.

---

## 2. What problem does weight quantization address?

**The model-load footprint.** Weights are the largest single VRAM consumer and are
constant throughout inference — they don't grow with sequence length, but they set
the floor for how much VRAM is available for everything else.

Current ProtGPT2 VRAM budget on RTX 4070 (12 GB):

```
FP16 weights:           1,485 MB (fixed)
Prefill overhead:          ~9 MB (transient, prompt-dependent)
INT8 KV decode growth:    ~3 MB (for 512 tokens)
Allocator/runtime:       ~100 MB (CUDA overhead)
─────────────────────────────
Total in use:          ~1,597 MB
Free VRAM:            ~10,691 MB  ← plenty of headroom
```

Now consider the same model on a 6 GB GPU:

```
FP16 weights:           1,485 MB
Prefill overhead:          ~9 MB
INT8 KV decode growth:    ~3 MB
Allocator/runtime:       ~100 MB
─────────────────────────────
Total in use:          ~1,597 MB
Free VRAM:             ~4,547 MB  ← still fits, but tight
```

With INT4 weights:

```
INT4 weights:             369 MB
Prefill overhead:          ~9 MB
INT8 KV decode growth:    ~3 MB
Allocator/runtime:       ~100 MB
─────────────────────────────
Total in use:            ~481 MB
Free VRAM:             ~5,663 MB  ← 4x more headroom
```

**Weight quantization and KV optimization are complementary layers:**
- Layer 1 (weight quantization): reduces the fixed floor
- Layer 2 (KV optimization): reduces the variable growth rate

Together they maximize the usable VRAM for generation on memory-constrained hardware.

---

## 3. What model(s) should be used for the first weight-quantized track?

**ProtGPT2 (774M params).**

Rationale:
- It's the model we have full benchmark infrastructure for
- The INT8 KV path is validated and can be combined immediately
- The weight footprint (1.45 GB FP16) is small enough that quantization is
  easy to validate end-to-end on the existing RTX 4070
- We can measure the stacked benefit: weight quantization + KV quantization
- Success criteria can reuse the existing evaluation harness

**Not a larger model yet.** A larger model (e.g., ProtGPT2-XL if it existed, or a
3B+ protein LM) would be a stronger motivation for weight quantization, but introduces
model-loading complexity, potential OOM during quantization, and new evaluation needs.
Start with ProtGPT2, validate the approach, then consider scaling.

---

## 4. Should the first step use ProtGPT2 or a larger model candidate?

**ProtGPT2 first.** The reasoning:

1. **Infrastructure reuse.** The benchmark harness, memory semantics, Triton kernel,
   and evaluation discipline are all built for ProtGPT2. Using a new model would
   require validating all of this from scratch.

2. **The stacking question is the novel contribution.** Weight quantization alone is
   well-studied. What's novel here is combining weight quantization with the existing
   INT8 KV Triton path and measuring the stacked capacity benefit. ProtGPT2 is the
   fastest path to answering this question.

3. **Feasibility is guaranteed.** ProtGPT2 at 774M params will quantize and run on
   the RTX 4070 without OOM risk. A larger model might not.

4. **The result generalizes.** If weight quant + KV quant stacks cleanly on ProtGPT2,
   the same approach applies to larger models. If it doesn't (e.g., quality collapses),
   better to discover that on a cheap model.

After validating on ProtGPT2, the next model candidate would be whichever protein LM
is the next practical user need (likely a 1-3B model that doesn't fit on 6 GB in FP16).

---

## 5. What quantization formats are realistic to test first?

Available in the current environment (torch 2.11, transformers 5.5):

| Method | Library | Status | Bits | Approach |
|--------|---------|--------|------|----------|
| `torch.ao.quantization.quantize_dynamic` | PyTorch built-in | Available | INT8 | Dynamic weight-only quantization |
| `BitsAndBytesConfig` (nf4/int8) | bitsandbytes | Installable (`pip install bitsandbytes`) | 4/8 | Weight-only, NF4 or INT8 with absmax |
| `GPTQConfig` | auto_gptq | Installable | 4 | Calibration-based weight quantization |
| `AwqConfig` | awq | Installable | 4 | Activation-aware weight quantization |
| `torchao` int8_weight_only | torchao | Installable | 8 | Weight-only INT8 with torch.compile |

**Recommended first step: `bitsandbytes` NF4 (4-bit) via HuggingFace integration.**

Rationale:
1. **Simplest integration.** `AutoModelForCausalLM.from_pretrained(..., quantization_config=BitsAndBytesConfig(load_in_4bit=True))` — one line change to model loading.
2. **HuggingFace-native.** No separate calibration step, no pre-quantized model needed.
3. **NF4 is the standard.** NF4 (4-bit NormalFloat) is the most widely used format
   for inference on consumer GPUs. It's what QLoRA uses.
4. **Maximum weight compression.** 4-bit gives ~4x weight reduction (1485 → ~370 MB),
   which is the most impactful first test.

**Recommended second step: `torch.ao.quantization.quantize_dynamic` (INT8).**

Rationale:
1. **Zero additional dependencies.** Already available in PyTorch.
2. **INT8 weights + INT8 KV** is a clean uniform-precision story.
3. **More modest compression** (1485 → ~740 MB) but potentially better quality.

**Deferred:** GPTQ and AWQ require calibration datasets and pre-quantized model files.
These are higher quality but higher setup cost. Defer until after the simpler methods
are baselined.

---

## 6. How should weight quantization and KV optimization be combined?

**Three-way comparison, stacking one layer at a time:**

| Config | Weights | KV Cache | What it tests |
|--------|---------|----------|---------------|
| A. Baseline | FP16 | FP16 | Current baseline (existing data) |
| B. Weight-only | INT4 (bnb) | FP16 | Weight quantization benefit alone |
| C. KV-only | FP16 | INT8 Triton | Current v5 path (existing data) |
| D. Stacked | INT4 (bnb) | INT8 Triton | **The new question: do they compose?** |

The key experiment is **Config D**: does the INT8 KV Triton path work correctly
when the model weights are in INT4? Potential issues:
- The Triton kernel expects FP16 Q/K/V projections from the model. If bitsandbytes
  dequantizes weights to FP16 during forward (which it does), this should work
  transparently.
- The `run_chunked_decode_step()` function calls the model's QKV projections, FFN,
  and layer norms via PyTorch. If these work with quantized weights (bitsandbytes
  handles this via custom Linear modules), the Triton kernel doesn't need changes.
- Quality may degrade more than either method alone — INT4 weights + INT8 KV is
  double quantization noise.

**Implementation approach:**
1. Modify `models.py:load_model_and_tokenizer()` to accept a `weight_quantization`
   parameter (e.g., `"fp16"`, `"bnb-nf4"`, `"bnb-int8"`)
2. When `weight_quantization="bnb-nf4"`, pass `BitsAndBytesConfig(load_in_4bit=True,
   bnb_4bit_compute_dtype=torch.float16)` to `from_pretrained`
3. Run the existing v5 generation path — no changes to `int8_generate.py`,
   `kv_int8_chunked.py`, or `triton_int8_attention.py`
4. Measure all four configs with fresh model loads

**The hypothesis:** Config D should show:
- Weight memory: ~370 MB (vs 1485 MB baseline) — 75% reduction
- Decode growth: ~3-18 MB (same as Config C) — KV optimization unchanged
- Total VRAM: ~470-490 MB (vs ~1600 MB baseline) — 70% total reduction
- Quality: some degradation from double quantization — need to measure

---

## 7. What benchmark criteria will define success?

### Memory (primary — this is a capacity track)

| Metric | Target | Rationale |
|--------|--------|-----------|
| Weight memory reduction | >=60% vs FP16 | INT4 should give ~75%; INT8 should give ~50% |
| Total VRAM (weights + decode) | <800 MB for ProtGPT2 | Fits on 4 GB GPUs with headroom |
| Decode growth unchanged | Within 10% of v5 | KV optimization must not regress |

### Quality

| Metric | Target | Rationale |
|--------|--------|-----------|
| Token agreement (64 steps) | >=90% | Double quantization will cause some drift |
| Token agreement (256 steps) | >=80% | Longer windows may diverge more |
| Cosine similarity (64 steps) | >=0.999 | Looser than v5's 0.9999 — weight quant adds noise |
| Cosine similarity (256 steps) | >=0.99 | Allows for accumulated double-quant drift |

Quality targets are intentionally looser than the KV-only path because weight
quantization introduces a larger perturbation than KV quantization. If quality
drops below these thresholds, the quantization is too aggressive.

### Speed

| Metric | Target | Rationale |
|--------|--------|-----------|
| Decode tok/s | >=40% of FP16 baseline | INT4 weights may slow down GEMMs |
| No hard speed target | — | This is a capacity track, not a speed track |

Speed may improve or degrade depending on whether quantized weight reads are
faster (less memory bandwidth) or slower (dequantization overhead). We measure
it but don't gate on it.

### Failure conditions

| Condition | Action |
|-----------|--------|
| Cosine < 0.99 at 64 steps | Quantization too aggressive; try INT8 instead of INT4 |
| Decode growth regresses >20% | Weight quantization interferes with KV path; investigate |
| Model fails to load with quantization | Library compatibility issue; try alternative method |
| Total VRAM > FP16 baseline | Quantization overhead exceeds savings; abandon method |

---

## 8. What should stay out of scope in the first iteration?

| Topic | Status | Rationale |
|-------|--------|-----------|
| GPTQ / AWQ calibration-based methods | Deferred | Requires calibration dataset; higher setup cost |
| Quantization-aware fine-tuning (QAT) | Out of scope | Research-grade; this is an inference track |
| Larger models (>1B params) | Deferred | Validate approach on ProtGPT2 first |
| ProGen2 | Deferred | Different architecture, separate validation needed |
| Weight quantization for speed | Out of scope | This is a capacity track |
| Mixed-precision weight policies (per-layer) | Deferred | Start with uniform quantization |
| Custom CUDA kernels for quantized GEMMs | Out of scope | Use library-provided kernels (bnb, torch.ao) |
| Prefill optimization | Out of scope | Prefill is already fast and not the bottleneck |
| GGUF / llama.cpp formats | Out of scope | Different ecosystem; stay in PyTorch/HF |

---

## Proposed Implementation Phases

### Phase 1 — bitsandbytes NF4 baseline (smallest useful step)

1. `pip install bitsandbytes`
2. Extend `load_model_and_tokenizer()` with `weight_quantization` parameter
3. Run Config B (INT4 weights + FP16 KV) — measure weight memory, quality, speed
4. Run Config D (INT4 weights + INT8 KV Triton) — measure stacked benefit
5. Compare all four configs in a single report

**Deliverable:** `results/summaries/weight_quant_phase1_report.md`

### Phase 2 — torch.ao INT8 weight-only (if Phase 1 quality is too low)

1. Apply `quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)`
2. Verify compatibility with `run_chunked_decode_step`
3. Compare INT8 weights vs INT4 weights on quality/memory tradeoff

### Phase 3 — User integration

1. Add `--weight-quant {fp16, bnb-nf4, bnb-int8}` to `generate_protgpt2.py`
2. Update capacity table to show weight + KV stacked savings
3. Update README and quickstart

---

## VRAM Projection Summary

| Config | Weights | KV (512 tok) | Total | vs Baseline |
|--------|---------|-------------|-------|-------------|
| FP16 + FP16 KV | 1,485 MB | 92 MB | ~1,597 MB | — |
| FP16 + INT8 KV (v5) | 1,485 MB | 3 MB | ~1,508 MB | -6% |
| INT4 + FP16 KV | ~370 MB | 92 MB | ~482 MB | -70% |
| **INT4 + INT8 KV** | **~370 MB** | **3 MB** | **~393 MB** | **-75%** |
| INT8 + INT8 KV | ~740 MB | 3 MB | ~763 MB | -52% |

The stacked INT4 + INT8 KV config could bring ProtGPT2 inference below 400 MB
of active VRAM — feasible on virtually any CUDA-capable GPU.

---
*Planning memo completed 2026-04-06. No implementation started.*
