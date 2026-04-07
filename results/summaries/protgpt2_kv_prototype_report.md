# ProtGPT2 INT8 KV Cache Prototype Report

Generated: 20260406T194856
Source: `results/raw/kv_prototype_eval_20260406T194723.json`

## Artifact Integrity

- Authoritative raw JSON: `results/raw/kv_prototype_eval_20260406T194723.json`
- All numbers in this report are generated from that file.
- Isolation method: fresh model load per measurement.
- decode_heavy: isolation check **PASSED** (baseline after_load=1485.47 MB, prototype after_load=1485.47 MB)
- long_decode: isolation check **PASSED** (baseline after_load=1485.47 MB, prototype after_load=1485.47 MB)

## Prototype Design

INT8 per-token absmax KV cache (`Int8KVCache` in `bio_inference_bench/kv_int8_cache.py`).
Subclasses HuggingFace `DynamicCache`. On each decode step:
1. Quantize new K/V to INT8 with per-token scale (FP16)
2. Concatenate onto INT8 cache
3. Dequantize full cache to FP16 for attention
4. Return FP16 tensors to the model's attention layers

**Goal:** Reduce generation memory growth in decode-heavy ProtGPT2 regimes.
**Non-goal:** Decode speedup.

## Result: Not Viable

The prototype **increased** memory usage and **decreased** speed.
The quantization scheme itself works (no observed behavioral drift), but the
dequantize-on-read architecture forces a full FP16 cache copy to coexist
with the INT8 storage, negating all savings.

## Memory Comparison

| Metric | decode_heavy Base | decode_heavy Proto | long_decode Base | long_decode Proto |
|--------|--------|--------|--------|--------|
| After load (MB) | 1485.47 | 1485.47 | 1485.47 | 1485.47 |
| Peak allocated (MB) | 1536.76 | 1564.21 | 1622.43 | 1695.72 |
| Peak reserved (MB) | 1712.0 | 1720.0 | 1856.0 | 1914.0 |
| Gen overhead above load (MB) | 51.29 | 78.74 | 136.96 | 210.25 |
| INT8 KV storage (MB) | N/A | 26.01 | N/A | 69.52 |
| FP16 KV equivalent (MB) | N/A | 50.45 | N/A | 134.82 |
| INT8 compression ratio | N/A | 1.94x | N/A | 1.94x |
| **Net memory change** | --- | **-27.45 MB (-53.5%)** | --- | **-73.29 MB (-53.5%)** |

**Root cause:** `update()` dequantizes the entire INT8 cache to FP16 every
decode step. GPU memory simultaneously holds: INT8 data + FP16 scales + FP16
dequantized copy. The dequantized copy alone equals the baseline cost.

## Speed Comparison

| Metric | decode_heavy Base | decode_heavy Proto | long_decode Base | long_decode Proto |
|--------|--------|--------|--------|--------|
| Decode tok/s | 124.83 | 78.68 | 124.14 | 78.18 |
| E2E tok/s | 124.2 | 77.55 | 123.91 | 78.08 |
| **Speed change** | --- | **-37.0%** | --- | **-37.0%** |

Both paths received equal warmup. Decode timing comparison is fair.
~35% slowdown is from per-step quantize + dequantize overhead.

## Behavior Sanity Check

| Metric | decode_heavy | long_decode |
|--------|--------|--------|
| Token agreement (first 64 steps) | 100.0% | 100.0% |
| Top-1 logit agreement | 100.0% | 100.0% |
| Avg logit cosine similarity | 0.999995 | 0.999993 |
| First divergence step | 64 | 64 |

No observed behavioral drift in this test window. Greedy decode produced
identical tokens across all checked steps. Logit cosine similarity is
near-unity but not exactly 1.0, indicating sub-threshold quantization noise
exists. Whether this noise accumulates over longer sequences or with
different model weights is not tested here.

## Promising for Second Iteration?

**The quantization scheme is valid; the memory architecture is not.**

What works: INT8 per-token absmax achieves 1.94x compression with no
observed behavioral drift. The precision is sufficient for ProtGPT2.

What doesn't work: dequantize-on-read forces the full FP16 copy to coexist,
negating savings. Three directions for a viable second iteration:

- **A. Fused attention kernel:** Read INT8 directly in attention, never
  materialize full FP16 cache. Best savings, requires CUDA kernel work.
- **B. Chunked dequantize:** Dequantize in small chunks, accumulate attention
  incrementally. Reduces peak to O(chunk + INT8_cache). Pure PyTorch.
- **C. Pre-allocated buffer reuse:** Fixed FP16 buffer overwritten each step.
  Peak = INT8 cache + one FP16 buffer (~1.5x baseline). Simplest.

## Caveats

1. This prototype failed its primary goal. Memory increased ~54%.
2. The no-drift finding is limited to 64 steps on one prompt per config.
3. Speed regression (~35%) is structural for dequantize-on-read designs.
4. Do not generalize to ProGen2 (different architecture, different bottleneck).
5. The ~9 MB warmup JIT residual (1485 vs 1476 MB) is consistent across
   all runs and does not affect the comparison.

---
*Prototype assessment. No production optimization implemented.*
