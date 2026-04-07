# ProtGPT2 v5 Handoff Note

**Date:** 2026-04-06 (updated 2026-04-07)
**Status:** Complete — ready for use

---

## Canonical Path

**ProtGPT2 + INT8-Triton KV** is the validated capacity optimization path.

Entry point: `python scripts/generate_protgpt2.py --kv-mode int8-triton`

## What is solved

- **Decode-phase memory growth reduced by 60-97%** vs FP16 baseline on ProtGPT2
- Fused Triton kernel eliminates the Python chunked attention overhead (3.7x faster than v2 reference)
- 100% token agreement through 256 decode steps, cosine similarity 0.999986
- Memory metrics are properly separated (e2e overhead vs decode-phase growth) with
  independent peak tracking per phase
- User-facing CLI with `--compare` mode (fresh model loads, capacity table)
- Benchmark integration preserving structured JSON output

## What is intentionally out of scope

| Topic | Status | Rationale |
|-------|--------|-----------|
| **Weight quantization** | **Closed (2026-04-07)** | **Three methods tested (NF4, bnb-int8, torchao INT8) — all failed quality gate. See closure memo.** |
| Boundary-layer protection | Dropped | v3 showed no quality benefit for ProtGPT2 |
| Asymmetric K/V precision | Deferred | Quality already near-perfect; revisit only if needed |
| Per-head calibration | Deferred | No evidence of head-specific sensitivity |
| Orthogonal rotation | Deferred | Cosine already ~1.0; adds overhead |
| ProGen2 optimization | Deferred | Different bottleneck profile (non-KV dominated) |
| Batch size > 1 | Not implemented | Triton kernel is batch=1, q_len=1 only |
| Prefill INT8 | Not implemented | Prefill always FP16; INT8 is decode-only |
| INT4 / mixed-bit KV | Not implemented | INT8 provides sufficient compression for ProtGPT2 |
| GPTQ / AWQ / calibration-based weight quant | Out of scope | Requires calibration data; separate research question |

## Files a future maintainer should start from

### Core (modify with care)

| File | What it does | Caution |
|------|-------------|---------|
| `bio_inference_bench/triton_int8_attention.py` | Fused Triton kernel | Validated at Phase A (39 tests). Changes require re-running `tests/test_triton_int8_attention.py` |
| `bio_inference_bench/kv_int8_chunked.py` | INT8 KV cache + decode step dispatch | Contains `use_triton` flag, v2 Python fallback, and the ChunkedInt8KVCache class |
| `bio_inference_bench/int8_generate.py` | Generation orchestrator | ProtGPT2 validation gate lives here. Memory measurement logic (dual-peak tracking) is here |
| `bio_inference_bench/generation.py` | FP16 baseline path | **Do not add INT8 logic here.** Kept clean as the baseline reference |

### Scripts (user-facing)

| File | What it does |
|------|-------------|
| `scripts/generate_protgpt2.py` | Primary user CLI (`--kv-mode`, `--compare`, `--prompt`) |
| `scripts/benchmark_generation.py` | Benchmark CLI (`--kv-mode` with ProtGPT2 gate) |
| `scripts/eval_kv_fused_v5.py` | Experimental eval (three-way comparison, used for Phase B data) |

### Tests

| File | What it covers |
|------|---------------|
| `tests/test_triton_int8_attention.py` | 39 correctness tests: Triton vs FP16 reference, vs v2 chunked, edge cases, determinism |

### Documentation

| File | What it covers |
|------|---------------|
| `results/summaries/protgpt2_v5_quickstart.md` | User-oriented quickstart |
| `results/summaries/v5_productization_plan.md` | Implementation plan and design decisions |
| `results/summaries/protgpt2_fused_kernel_design.md` | Kernel design memo (Phase A/B/C specs) |
| `results/summaries/protgpt2_v5_sample_output.md` | Example compare-mode output |
| `results/summaries/protgpt2_weight_quant_track_closure.md` | Weight-quant track closure memo (2026-04-07) |
| `results/summaries/weight_quantized_track_plan.md` | Weight-quant planning memo (closed) |

### Key artifacts (benchmark evidence)

| File | What it contains |
|------|-----------------|
| `results/raw/kv_fused_v5_eval_20260406T214141.json` | Phase B corrected benchmark (authoritative) |
| `results/raw/kv_prototype_v2_eval_20260406T195958.json` | v2 baseline (pre-Triton) |
| `results/raw/kv_prototype_v3_eval_20260406T202004.json` | v3 boundary protection (dropped) |
| `results/raw/weight_quant_phase1_20260406T231058.json` | NF4 four-way comparison (failed quality) |
| `results/raw/weight_quant_phase2_20260407T100546.json` | bnb-int8 four-way comparison (failed quality) |
| `results/raw/weight_quant_phase3_20260407T103724.json` | torchao INT8 four-way comparison (failed quality) |

## Weight quantization: evaluated and closed

Three zero-calibration weight-quantization methods were tested for ProtGPT2 in
Phases 1-3. All failed the quality gate (>=90% token agreement, >=0.999 cosine
at 64 steps). The track was closed 2026-04-07.

| Method | Weight memory | Token agr. (best) | Cosine (best) | Why it failed |
|--------|-------------|-------------------|---------------|---------------|
| NF4 (bnb 4-bit) | 546 MB | 98.4% | 0.995 | Collapses on long prompts (28% at prompt=256) |
| LLM.int8() (bnb 8-bit) | 817 MB | 7.8% | 0.716 | Outlier decomposition catastrophically miscalibrated |
| torchao INT8 | 879 MB | 51.6% | 0.898 | Fails quality + dequant buffers break stacking |

**Do not reopen casually.** The failure is not a configuration issue — three
fundamentally different quantization algorithms all fail. Further progress would
require either calibration-based methods (GPTQ/AWQ with protein sequence data)
or a different model. Both are separate research questions.

Full details: `results/summaries/protgpt2_weight_quant_track_closure.md`

## What not to change casually

1. **Memory measurement in `int8_generate.py`:** The dual-peak tracking
   (`torch.cuda.reset_peak_memory_stats()` between prefill and decode) is what makes
   `decode_phase_growth_mb` correct. Removing the reset conflates prefill transients
   with decode growth.

2. **ProtGPT2 validation in `_validate_protgpt2()`:** Checks both model name AND
   architecture parameters. Do not relax this gate without validating the Triton kernel
   on the new architecture.

3. **`generation.py` baseline path:** This is the FP16 reference used for all
   comparisons. INT8 logic must NOT be added here.

4. **Triton kernel BLOCK_KV=64:** Confirmed optimal by Phase C tuning sweep. Changing
   this requires re-running `scripts/tune_block_kv.py` and verifying no regression.

5. **Fresh model loads in compare mode:** The `--compare` flag loads a fresh model per
   mode. This is intentional — shared-load comparisons have unreliable memory metrics.

---
*Handoff note completed 2026-04-06. Updated 2026-04-07 with weight-quantization track closure.*
