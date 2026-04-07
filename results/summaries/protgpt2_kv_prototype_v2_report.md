# ProtGPT2 KV Prototype v2: Chunked-Dequantize INT8

Generated: 20260406T200810
Source: `results/raw/kv_prototype_v2_eval_20260406T195958.json`
v1 reference: `results/raw/kv_prototype_eval_20260406T194723.json`

## Artifact Integrity

- Authoritative raw JSON: `results/raw/kv_prototype_v2_eval_20260406T195958.json`
- All numbers in this report are generated from that file.
- Isolation method: fresh model load per measurement.
- decode_heavy: isolation **PASSED** (baseline=1485.47 MB, v2=1485.47 MB)
- long_decode: isolation **PASSED** (baseline=1485.47 MB, v2=1485.47 MB)

## 1. What Changed from v1

**v1 (dequantize-on-read):** Stored KV in INT8, dequantized the entire cache to
FP16 every decode step. Full FP16 coexisted with INT8 storage → +54% memory.

**v2 (chunked dequantize):** Stores KV in INT8. During decode, dequantizes in
chunks of 64 positions using online softmax. The full FP16 cache is **never
materialized** during decode.

## 2. Full FP16 Materialization Avoided?

**Yes, during decode.** Peak decode memory holds: INT8 cache + FP16 scales +
one chunk (64 positions) of dequantized FP16 at a time + online softmax accumulators.

**During prefill:** Standard FP16 path runs (full prompt attention needed),
then the FP16 cache is freed before decode begins.

## 3. Memory Comparison vs Baseline

| Metric | decode_heavy Base | decode_heavy v2 | long_decode Base | long_decode v2 |
|--------|--------|--------|--------|--------|
| After load (MB) | 1485.47 | 1485.47 | 1485.47 | 1485.47 |
| Peak allocated (MB) | 1536.76 | 1512.47 | 1622.43 | 1557.64 |
| Peak reserved (MB) | 1712.00 | 1658.00 | 1856.00 | 1712.00 |
| Gen overhead (MB) | 51.29 | 27.00 | 136.96 | 72.17 |
| INT8 KV storage (MB) | N/A | 26.01 | N/A | 69.52 |
| FP16 KV equivalent (MB) | N/A | 50.45 | N/A | 134.82 |
| **Net change** | --- | **+24.29 MB (+47.4%)** | --- | **+64.79 MB (+47.3%)** |

## 4. Memory Comparison vs v1

v1 reference: `results/raw/kv_prototype_eval_20260406T194723.json`

| Metric | decode_heavy v1 | decode_heavy v2 | long_decode v1 | long_decode v2 |
|--------|--------|--------|--------|--------|
| Gen overhead (MB) | 78.74 | 27.00 | 210.25 | 72.17 |
| Peak allocated (MB) | 1564.21 | 1512.47 | 1695.72 | 1557.64 |
| vs baseline | +53.5% worse | +47.4% better | +53.5% worse | +47.3% better |

## 5. Speed Comparison

| Metric | decode_heavy Base | decode_heavy v2 | long_decode Base | long_decode v2 |
|--------|--------|--------|--------|--------|
| Decode tok/s | 125.87 | 40.41 | 124.26 | 21.64 |
| E2E tok/s | 125.21 | 40.50 | 124.04 | 21.67 |
| **Speed change** | --- | **-67.9%** | --- | **-82.6%** |

Both paths received equal warmup. The 3-6x slowdown is from the pure-PyTorch
chunked attention loop (no kernel fusion). This is a capacity prototype, not
a throughput optimization.

## 6. Behavior Sanity Check

| Metric | decode_heavy | long_decode |
|--------|--------|--------|
| Token agreement (64 steps) | 100.0% | 100.0% |
| Top-1 logit agreement | 100.0% | 100.0% |
| Avg logit cosine sim | 0.999993 | 0.999994 |
| First divergence step | 64 | 64 |

No observed behavioral drift. Cosine similarity ~0.99999 indicates
sub-threshold INT8 quantization noise that does not affect argmax selection.

## 7. Does v2 Justify Direction A (Fused Kernel)?

**Yes, conditionally.** The 47% memory reduction is architecturally real and
behaviorally clean. The speed cost (3-6x) is the sole limitation, and it is
entirely attributable to Python-level chunked attention — a fused CUDA kernel
would eliminate this overhead while preserving the memory benefit.

A fused kernel is justified if the 47% generation-overhead savings materially
enables a use case (longer sequences, larger batches, or fitting on a smaller GPU).

## 8. Recommendation

**Continue to fused-kernel work (Direction A) if capacity scaling is a priority.**

Before starting fused-kernel work, consider intermediate improvements on the
chunked path: asymmetric K/V precision, layer-aware policies, or selective
compression — these can be tested on the existing v2 framework cheaply and may
improve the quality/compression tradeoff before committing to kernel development.

## Caveats

1. 47% savings applies to generation overhead, not total VRAM (weights dominate).
2. Sanity check covers 64 steps on one prompt per config — not exhaustive.
3. Speed regression (3-6x) makes this unusable for throughput-sensitive workloads.
4. Prefill still uses full FP16 (chunked decode only).
5. ProtGPT2-specific; do not generalize to ProGen2 or other architectures.

---
*Report regenerated from `results/raw/kv_prototype_v2_eval_20260406T195958.json` on 20260406T200810.*
*Every numeric field was read directly from the raw JSON. No hand-maintained values.*
