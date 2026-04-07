# bio-inference-bench

Profiling benchmark and capacity optimization for autoregressive protein sequence
generation on consumer GPUs (12 GB VRAM).

**Status:** Finalized (2026-04-07). See `results/summaries/final_project_summary.md`.

## What this project is

A **profiling and capacity optimization** project for protein language model inference.
Starting from a reproducible measurement harness, the project identified the KV cache
as the primary memory growth bottleneck during decode, then built a fused Triton kernel
for INT8 KV attention that reduces decode-phase memory growth by 60-97% with zero
quality loss. Weight quantization was evaluated (three methods) and ruled out for
ProtGPT2.

## Key design principles

1. **No assumed bottleneck.** The harness measures; humans interpret.
2. **Memory concepts are reported separately — never conflated:**
   - **Theoretical KV cache** — formula-based estimate: `2 × layers × batch × seq_len × kv_heads × head_dim × dtype_bytes`
   - **Observed peak allocated** — `torch.cuda.max_memory_allocated()` during generation
   - **Observed peak reserved** — `torch.cuda.max_memory_reserved()` during generation
3. **Observed peak memory is NOT KV cache.** It includes activations, allocator overhead, temporary buffers, logits, and sampling tensors.
4. **Prompt lengths are always in tokens** (`prompt_token_length`), never amino-acid characters.
5. **Manual prefill/decode loop is the primary profiling path.** `model.generate()` is a secondary end-to-end baseline.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

```bash
# Inspect model metadata and theoretical KV cache
python scripts/inspect_model.py --model protgpt2

# Run a single benchmark
python scripts/benchmark_generation.py \
  --model protgpt2 \
  --prompt-token-length 64 \
  --max-new-tokens 128

# Run the smoke benchmark pair (harness validation)
python scripts/run_smoke.py
```

## INT8 KV Capacity Mode (ProtGPT2)

The INT8 KV capacity path reduces decode-phase memory growth by 60-97% vs FP16 baseline,
letting you generate longer sequences on memory-constrained GPUs.

**Capacity optimization, not speed optimization.** The INT8 path uses less VRAM for the
KV cache, giving you more headroom. Decode speed is ~60% of FP16 — the remaining gap
is in weight computations (GEMMs), not the attention kernel. If VRAM is not your
bottleneck, use FP16.

### Quick Start

```bash
# Generate with INT8 KV (real protein prompt)
python scripts/generate_protgpt2.py --prompt "MKTLLILAVL" --kv-mode int8-triton

# Compare FP16 vs INT8-Triton (capacity table with memory breakdown)
python scripts/generate_protgpt2.py --compare --max-new-tokens 256

# Benchmark mode (structured JSON output)
python scripts/benchmark_generation.py --model protgpt2 --kv-mode int8-triton
```

Compare mode uses **fresh model loads** for each mode so the memory measurements are
independent and apples-to-apples. See `results/summaries/protgpt2_v5_quickstart.md`
for a full guide.

### How it works

1. **Prefill** uses standard FP16 attention (full prompt forward pass)
2. Prefill KV cache is transferred to INT8 with per-token absmax quantization (1.94x compression)
3. **Decode** uses a fused Triton kernel that reads INT8 KV directly, dequantizes in SRAM, and computes exact attention with online softmax — the full FP16 cache is never materialized

### Limitations

1. **ProtGPT2 only** (774M params, GPT-2 architecture, 20 heads, 36 layers, head_dim=64)
2. **Batch size = 1** (single-sequence decode)
3. **Prefill always uses FP16** — INT8 is decode-only
4. **Requires NVIDIA GPU with Triton support**
5. **Decode speed is 58-64% of FP16 baseline** — the remaining gap is weight GEMMs, not attention
6. **Quality validated through 256 decode steps** — longer sequences not yet tested
7. **This is a capacity optimization, not a speed optimization**

---

## Supported models

| Model | HuggingFace path | INT8 KV | Notes |
|-------|-----------------|---------|-------|
| ProtGPT2 | `nferruz/ProtGPT2` | Validated | Primary model, all optimizations validated |
| ProGen2-small | `hugohrban/progen2-small` (primary), `multimolecule/progen2-small` (fallback) | Not optimized | Profiling only; different bottleneck profile |

## Project structure

```
bio-inference-bench/
  bio_inference_bench/              # Core library
    triton_int8_attention.py        # Fused Triton kernel for INT8 KV attention
    kv_int8_chunked.py              # INT8 KV cache + decode step dispatch
    int8_generate.py                # INT8 generation orchestrator
    models.py                       # Model loading with candidate fallback
    generation.py                   # FP16 baseline path
    eval_helpers.py                 # Shared evaluation logic (weight-quant phases)
    profiler.py                     # CUDA memory profiling
    kv_estimator.py                 # Theoretical KV cache estimation
    report.py                       # Output formatting, capacity table
    utils.py                        # Constants, model registry, helpers
  scripts/                          # CLI entry points
    generate_protgpt2.py            # Primary user CLI (--kv-mode, --compare)
    benchmark_generation.py         # Structured benchmark
    inspect_model.py                # Model metadata inspection
    run_smoke.py                    # Harness validation
    run_grid.py                     # Full benchmark grid
  tests/
    test_triton_int8_attention.py   # 39 Triton kernel correctness tests
  results/
    raw/                            # Per-run JSON results
    summaries/                      # Reports and reproducibility artifacts
```
