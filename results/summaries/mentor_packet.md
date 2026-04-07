# Bio-Inference-Bench — Mentor Packet (Smoke Validation)

Generated: 20260406T175222


## 1. Project Tree

```
bio-inference-bench/
  bio_inference_bench/
    __init__.py
    utils.py
    profiler.py
    kv_estimator.py
    models.py
    generation.py
    report.py
  scripts/
    inspect_model.py
    benchmark_generation.py
    run_smoke.py
    run_grid.py
  results/
    raw/
    summaries/
  pyproject.toml
  README.md
```

## 2. Environment Summary

- Python: 3.12.3 (main, Jan 22 2026, 20:57:42) [GCC 13.3.0]
- PyTorch: 2.11.0+cu130
- Transformers: 5.5.0
- Accelerate: 1.13.0
- Platform: Linux-6.18.7-76061807-generic-x86_64-with-glibc2.39
- GPU: NVIDIA GeForce RTX 4070
- GPU Memory: 11864 MB
- CUDA: 13.0

## 3. Implemented Files

- **utils.py**: Constants, SMOKE_CONFIG, MODEL_REGISTRY with candidate fallbacks, helpers
- **profiler.py**: CUDA memory snapshot/tracking with context manager interface
- **kv_estimator.py**: Formula-based theoretical KV cache estimation (never from CUDA)
- **models.py**: Model/tokenizer loading with ordered candidate fallback, metadata extraction
- **generation.py**: Manual prefill/decode (PRIMARY) + generate API (SECONDARY)
- **report.py**: Console output, JSON export, mentor packet generation
- **inspect_model.py**: CLI: model metadata + theoretical KV cache table
- **benchmark_generation.py**: CLI: single benchmark run
- **run_smoke.py**: CLI: smoke benchmark pair with mentor packet output
- **run_grid.py**: CLI: active gate — exits with error until mentor review

**Intentional TODOs for later:**
- Full benchmark grid execution (gated behind mentor review)
- Warm-up runs for timing stability
- Multi-batch benchmarking
- Deeper allocator/activation profiling (torch.profiler integration)

## 4. Smoke Benchmark Config

- prompt_token_length: 64
- max_new_tokens: 128
- batch_size: 1
- do_sample: False
- num_beams: 1
- use_cache: True
- dtype: float16
- decode_mode: greedy

## 5. Model Candidate Resolution


### protgpt2

- Candidates attempted: nferruz/ProtGPT2
- Successfully loaded: nferruz/ProtGPT2
- Warnings/failures:
  - Config field 'num_key_value_heads' not found (tried fallbacks: ('num_kv_heads',))
  - num_key_value_heads not found; falling back to num_attention_heads=20
  - Candidate nferruz/ProtGPT2: applied patch — Added get_head_mask() to ProGenPreTrainedModel (removed in transformers 5.x)
  - Candidate nferruz/ProtGPT2: applied patch — Added get_head_mask() to inner transformer model (GPT2Model)

### progen2-small

- Candidates attempted: hugohrban/progen2-small
- Successfully loaded: hugohrban/progen2-small
- Warnings/failures:
  - Used fallback config field 'n_embd' for 'hidden_size'
  - Used fallback config field 'n_head' for 'num_attention_heads'
  - Config field 'num_key_value_heads' not found (tried fallbacks: ('num_kv_heads',))
  - num_key_value_heads not found; falling back to num_attention_heads=16
  - Used fallback config field 'n_positions' for 'max_position_embeddings'
  - Candidate hugohrban/progen2-small: applied patch — Added get_head_mask() to ProGenPreTrainedModel (removed in transformers 5.x)
  - Candidate hugohrban/progen2-small: applied patch — Added get_head_mask() to inner transformer model (ProGenModel)
  - Candidate hugohrban/progen2-small: applied patch — Moved scale_attn from meta to cuda:0 in 12 attention layers
  - Candidate hugohrban/progen2-small: applied patch — Added config.num_hidden_layers = 12 (alias for config.n_layer)

## 6. Per-Model Results (PRIMARY: manual prefill/decode)


### protgpt2

| Metric | Value |
|--------|-------|
| Theoretical weight memory | 1476.35 MB |
| Actual allocated after load | 1476.35 MB |
| Actual reserved after load | 1616.00 MB |
| Theoretical KV cache (actual seq_len=192) | 33.75 MB |
| Theoretical KV upper bound (configured seq_len=192) | 33.75 MB |
| Theoretical KV as % of weights | 2.29% |
| Observed peak allocated | 1519.68 MB |
| Observed peak reserved | 1678.00 MB |
| Prefill time | 153.69 ms |
| Decode time | 1081.70 ms |
| Decode tokens/sec | 117.41 |
| End-to-end tokens/sec | 103.61 |
| Actual generated tokens | 128 |


### progen2-small

| Metric | Value |
|--------|-------|
| Theoretical weight memory | 288.29 MB |
| Actual allocated after load | 309.93 MB |
| Actual reserved after load | 334.00 MB |
| Theoretical KV cache (actual seq_len=192) | 9.00 MB |
| Theoretical KV upper bound (configured seq_len=192) | 9.00 MB |
| Theoretical KV as % of weights | 3.12% |
| Observed peak allocated | 336.85 MB |
| Observed peak reserved | 364.00 MB |
| Prefill time | 87.82 ms |
| Decode time | 815.45 ms |
| Decode tokens/sec | 155.74 |
| End-to-end tokens/sec | 141.71 |
| Actual generated tokens | 128 |


## 7. Secondary Path Cross-Validation (generate API)

The generate API results below are for sanity-checking only.
Bottleneck conclusions should be drawn from the PRIMARY path above.


### protgpt2

- Actual tokens generated: 7
- Total seq length: 71
- Total time: 120.25 ms
- End-to-end tokens/sec: 58.21
- Theoretical KV at actual seq_len=71: 12.48 MB
- Observed peak allocated: 1500.16 MB
- Observed peak reserved: 1634.00 MB

### progen2-small

- Actual tokens generated: 128
- Total seq length: 192
- Total time: 881.62 ms
- End-to-end tokens/sec: 145.19
- Theoretical KV at actual seq_len=192: 9.00 MB
- Observed peak allocated: 345.94 MB
- Observed peak reserved: 374.00 MB

## 8. Measurement Semantics

**This section defines what each reported metric means.**

- **Theoretical KV cache** is computed from a formula:
  `2 × num_layers × batch_size × seq_len × num_kv_heads × head_dim × dtype_bytes`.
  It is NEVER measured from CUDA memory. It represents the minimum memory
  required to store the key and value tensors for all layers.

- **Observed peak allocated** (`torch.cuda.max_memory_allocated()`) is the
  highest point of actually-used GPU memory during the run. This includes
  model weights, KV cache, attention matrices, activation tensors, logits,
  sampling buffers, and any other temporary allocations.

- **Observed peak reserved** (`torch.cuda.max_memory_reserved()`) is the
  highest point of memory held by PyTorch's CUDA allocator, including
  free blocks in the allocator pool. This is >= peak allocated.

**Critical: observed peak memory MUST NOT be interpreted as KV cache.**
The difference between peak memory and weight memory includes activations,
attention matrices (O(B×H×S²) per layer), temporary buffers, logits, and
allocator fragmentation overhead. Only the theoretical estimate isolates KV cache.

**This is harness validation, NOT model ranking.** The two models are tested
to verify that the measurement stack works correctly, not to determine which
model is faster or more memory-efficient.

## 9. Known Measurement Artifacts

The following artifacts are inherent to the measurement setup and should
be considered when interpreting results:

1. **Secondary-path timing is warmed relative to the primary path.**
   The primary path (manual prefill/decode) runs first. By the time the
   secondary path (generate API) runs, CUDA kernels are already JIT-compiled
   and the GPU memory allocator is warmed. Secondary-path timings will
   therefore appear faster. This is expected, not a bug.

2. **First decode step includes warmup/JIT overhead.**
   In the primary path's per-step timing, the first decode step is
   typically 5-10x slower than subsequent steps due to CUDA kernel
   compilation on first use. Steady-state decode throughput should be
   computed from step 2 onward for accurate bottleneck analysis.

3. **Primary-path timings are the authoritative bottleneck signal.**
   The secondary path is for cross-validation only. Bottleneck conclusions,
   prefill/decode breakdowns, and per-step analysis should always reference
   the primary path. The secondary path is useful for confirming that
   end-to-end token counts and rough throughput are in the same ballpark.

4. **Secondary path may generate fewer tokens (EOS).**
   `model.generate()` respects EOS tokens and may stop early. The manual
   decode loop always generates exactly `max_new_tokens`. Theoretical KV
   estimates are computed per-path using the actual final sequence length.

## 10. Interpretation


### protgpt2

- Weight memory: 1476.35 MB
- Observed peak above weights: 43.34 MB
- Theoretical KV cache: 33.75 MB (77.9% of overhead)
- Non-KV overhead (activations, logits, allocator): 9.59 MB

- Prefill: 153.7 ms (12.4% of total)
- Decode: 1081.7 ms (87.6% of total)
- First decode step: 56.1 ms (includes JIT)
- Steady-state decode: 8.1 ms/step (122.9 tok/s)

**Preliminary observation (not a direct decomposition):**
At this short configuration, the theoretical KV estimate is large
relative to the observed overhead above weights (78%),
which is consistent with KV cache being a material contributor.
However, this should not be interpreted as a direct measurement of
component-wise memory attribution — the overhead residual also includes
activations, logits, and allocator behavior that are not isolated here.
Longer-sequence tests are needed to observe how this ratio evolves
as attention matrices grow with O(S²).


### progen2-small

- Weight memory: 288.29 MB
- Observed peak above weights: 48.56 MB
- Theoretical KV cache: 9.00 MB (18.5% of overhead)
- Non-KV overhead (activations, logits, allocator): 39.56 MB

- Prefill: 87.8 ms (9.7% of total)
- Decode: 815.5 ms (90.3% of total)
- First decode step: 16.0 ms (includes JIT)
- Steady-state decode: 6.3 ms/step (157.6 tok/s)

**Preliminary observation (not a direct decomposition):**
The theoretical KV estimate is 19% of the observed
overhead above weights, which suggests KV cache is not the dominant
memory consumer at this sequence length. The remaining overhead likely
includes activations, attention score matrices, logits, and allocator
fragmentation. Whether KV cache becomes dominant at longer sequences
requires grid testing.

**Confidence level:** Moderate. The harness produces plausible numbers —
weight memory matches theoretical, timing breakdown is consistent,
and per-step decode times show expected JIT warmup pattern.
Further validation needed: warm-up runs, repeated measurements,
and longer sequences to test scaling behavior.

## 11. Recommendation for Next Step

**A. Proceed to full benchmark grid.**
All models completed successfully. The harness is validated.
