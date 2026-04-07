# bio-inference-bench: Final Project Summary

**Date:** 2026-04-07
**Status:** Finalized. No active development tracks.

---

## The original question

Can we reduce the VRAM footprint of autoregressive protein sequence generation on
consumer GPUs (12 GB RTX 4070) — and by how much, at what quality cost?

The project started as a profiling benchmark to measure where inference memory
actually goes (model weights, KV cache, activations, allocator overhead) before
making any optimization decisions. It evolved through profiling, KV cache
optimization, kernel development, productization, and weight-quantization
experiments.

## What was built

### Profiling harness (foundation)

A reproducible benchmark framework for autoregressive generation:
- Smoke benchmarks and grid evaluation for ProtGPT2 (774M) and ProGen2-small (151M)
- Operator profiling identifying per-component memory and time costs
- Dual memory semantics: end-to-end generation overhead and decode-phase growth,
  tracked independently with `torch.cuda.reset_peak_memory_stats()` between phases
- Fresh model load per configuration for clean memory measurement
- Structured JSON output for all raw results

### INT8 KV capacity optimization (the successful path)

A four-version evolution of KV cache compression for ProtGPT2:

| Version | Approach | Result |
|---------|----------|--------|
| v1 | Dequantize-on-read INT8 | Failed (memory coexistence) |
| v2 | Chunked dequantize INT8 | -47% generation overhead, 3-6x slower |
| v3 | Boundary-layer protection on v2 | No quality benefit, dropped |
| v5 | Fused Triton kernel | -60 to -97% decode growth, 64% of FP16 speed |

The v5 fused Triton kernel is the final product:
- Single-program-per-head design, BLOCK_KV=64
- Reads INT8 KV from global memory, dequantizes in SRAM
- FP32 accumulators with online softmax (Milakov & Gimelshein 2018)
- 100% token agreement through 256 decode steps, cosine similarity 0.999986
- 39 correctness tests (vs FP16 reference, vs v2 chunked, edge cases, determinism)

### Productized CLI and API

- `scripts/generate_protgpt2.py` — user CLI with `--kv-mode`, `--compare`, `--prompt`
- `bio_inference_bench/int8_generate.py` — programmatic API
- Capacity table with four sections (measured, model limit, capped VRAM, uncapped slope)
- ProtGPT2 validation gate (model name + architecture check)
- Benchmark mode with structured JSON output

## What succeeded

1. **KV cache compression:** INT8 per-token absmax quantization with fused Triton
   decode kernel. Reduces decode-phase memory growth by 60-97% with zero quality
   loss (100% token agreement, cos >= 0.999986).

2. **Memory stacking validation:** Weight quantization and KV optimization compose
   independently — confirmed across three weight-quantization methods. The KV
   optimization layer adds zero quality loss on top of any weight method.

3. **Kernel tuning:** BLOCK_KV=64 confirmed optimal by Phase C sweep. Attention
   kernel is only ~5% of per-step time — the remaining 36-42% speed gap is weight
   GEMMs, outside the scope of KV optimization.

4. **Measurement methodology:** Dual-peak tracking, fresh model loads per config,
   shared evaluation helpers — a discipline that made all results comparable across
   phases.

## What failed

**Weight quantization for ProtGPT2.** Three zero-calibration methods were tested:

| Method | Weight memory | Best quality (64 steps) | Failure mode |
|--------|-------------|------------------------|-------------|
| NF4 (bitsandbytes 4-bit) | 546 MB (63% reduction) | 98.4% token, cos=0.995 | Collapses on long prompts (28% at prompt=256) |
| LLM.int8() (bitsandbytes 8-bit) | 817 MB (45% reduction) | 7.8% token, cos=0.716 | Outlier decomposition catastrophically miscalibrated |
| torchao INT8 (per-tensor symmetric) | 879 MB (41% reduction) | 51.6% token, cos=0.898 | Quality fails + dequant buffers inflate decode growth 3-4x |

None met the quality gate (>=90% token agreement, >=0.999 cosine at 64 steps).
The track was closed after Phase 3. For ProtGPT2 specifically, standard
zero-calibration weight-quantization methods do not produce acceptable results.

## The final canonical path

**ProtGPT2 + FP16 weights + INT8-Triton KV**

```bash
python scripts/generate_protgpt2.py --prompt "MKTLLILAVL" --kv-mode int8-triton
```

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Weights | FP16 (1489 MB) | Weight quantization failed quality gate |
| KV cache | INT8-Triton | 60-97% decode growth reduction, zero quality loss |
| Prefill | FP16 | Prefill is fast, not the bottleneck |
| Decode kernel | Fused Triton (BLOCK_KV=64) | 3.7x faster than Python chunked reference |

Total VRAM for ProtGPT2 at 512 new tokens: ~1558 MB (vs ~1622 MB FP16 baseline).
The savings are in decode growth rate, not absolute peak — the weight footprint
dominates.

## What was explicitly ruled out

| Topic | Status | Why |
|-------|--------|-----|
| ProtGPT2 weight quantization | Closed | Three methods failed; track closed 2026-04-07 |
| GPTQ / AWQ / calibration-based methods | Out of scope | Requires calibration data; separate research question |
| ProGen2 optimization | Deferred | Different bottleneck (non-KV dominated); not started |
| Larger models (3B+) | Deferred | Separate evaluation needed |
| Boundary-layer KV protection | Dropped | v3 showed no quality benefit for ProtGPT2 |
| Asymmetric K/V precision | Deferred | Quality already near-perfect at INT8 |
| Batch size > 1 | Not implemented | Triton kernel is single-sequence only |
| Prefill INT8 | Not implemented | Prefill is not the bottleneck |
| Speed optimization | Out of scope | This is a capacity project |

## What a future reader should take away

1. **The KV capacity path works.** INT8 per-token absmax KV quantization with a fused
   Triton decode kernel is a validated, zero-quality-loss optimization for ProtGPT2.
   It reduces the variable cost of generation (decode-phase memory growth) by up to
   97%, freeing VRAM for longer sequences.

2. **Weight quantization did not work for this model.** Three independent
   zero-calibration methods all failed the quality gate for ProtGPT2. This is a
   finding about ProtGPT2 and these specific methods — not a general statement about
   protein language models or weight quantization. Calibration-based methods or
   different models may behave differently.

3. **The speed gap is in weight GEMMs, not attention.** The Triton kernel is only ~5%
   of per-step time. The remaining 36-42% speed gap vs FP16 baseline is from weight
   matrix multiplications. Closing this gap requires weight-level optimization (which
   failed for quality reasons) or hardware with faster INT8/FP16 GEMM throughput.

4. **Measurement discipline matters.** Dual-peak memory tracking, fresh model loads,
   and shared evaluation helpers are what made the results across seven evaluation
   phases (v1, v2, v3, v5, weight Phase 1/2/3) comparable and trustworthy. The raw
   JSON artifacts are preserved for reproducibility.

5. **Negative results are preserved.** The weight-quantization failure is documented
   in three phase reports and a closure memo. These are not hidden — they are part of
   the project record and inform future decisions.

---

## Timeline

| Date | Milestone |
|------|-----------|
| 2026-04-06 | Profiling harness, grid evaluation, operator profiling |
| 2026-04-06 | KV v1 (failed), v2 (chunked INT8, -47%), v3 (boundary, dropped) |
| 2026-04-06 | Fused Triton kernel: Phase A (39 tests), Phase B (3.7x over v2), Phase C (BLOCK_KV=64 optimal) |
| 2026-04-06 | v5 productization: CLI, API, capacity table, quickstart, public docs |
| 2026-04-06 | Weight quant Phase 1: NF4 four-way comparison (quality failed on long prompts) |
| 2026-04-07 | Weight quant Phase 2: bnb-int8 (catastrophic quality failure) |
| 2026-04-07 | Weight quant Phase 3: torchao INT8 (quality + stacking failure) |
| 2026-04-07 | Weight-quant track closed |
| 2026-04-07 | **Project finalized** |

---
*Final project summary completed 2026-04-07.*
