# TurboQuant Advanced Roadmap

> **Status update — 2026-04-07:** Track 1 (ProtGPT2 Capacity Path) is complete
> through v5 (fused Triton kernel). The weight-quantization track opened after v5
> productization has been **closed** — NF4, bitsandbytes LLM.int8(), and torchao INT8
> all failed the quality gate for ProtGPT2. See
> `results/summaries/protgpt2_weight_quant_track_closure.md`. The canonical path
> remains FP16 weights + INT8-Triton KV. ProGen2 (Track 4) remains deferred.

**Date:** 2026-04-06
**Baseline:** Benchmark v1 grid + v2 chunked-dequantize prototype
**Project:** bio-inference-bench (ProtGPT2 + ProGen2-small)

## Context

The v2 chunked-dequantize INT8 KV prototype achieved a 47% reduction in generation
overhead for ProtGPT2 with no behavioral drift (100% token agreement, cosine ~0.99999).
The remaining bottleneck is speed (3-6x slower due to pure-PyTorch chunked attention).

This roadmap defines four tracks for advancing the work, plus one concrete next
experiment.

---

## Track 1 — ProtGPT2 Capacity Path

**Goal:** Reduce ProtGPT2 KV cache memory during decode-heavy generation to enable
longer sequences, larger batches, or operation on smaller GPUs.

### Completed

| Version | Approach | Memory | Speed | Behavioral | Status |
|---------|----------|--------|-------|------------|--------|
| v1 | Dequantize-on-read INT8 | +54% worse | -37% | 100% agree | Failed (coexistence) |
| v2 | Chunked dequantize INT8 | **-47% better** | -68% to -83% | 100% agree | Promising for capacity |

### Next Steps (ordered by increasing engineering cost)

**v3 — Asymmetric / selective KV compression on v2 framework**
Explore whether different precision policies for K vs V, or per-layer policies,
can improve the compression/quality tradeoff before committing to kernel work.
See the v3 proposal below.

**v4 — Optimized chunked attention (pure PyTorch)**
Before jumping to CUDA kernels, optimize the v2 Python path:
- Reduce per-chunk overhead by batching chunk operations
- Use `torch.compile()` on the chunked attention inner loop
- Tune chunk_size (current=64; try 128, 256) for speed/memory balance
- Expected: 1.5-2x speedup over v2, still slower than baseline

**v5 — Fused INT8 attention kernel (Direction A)**
Write a Triton or CUDA kernel that reads INT8 KV directly and dequantizes
per-element inside the kernel. This eliminates the Python loop and per-chunk
tensor allocation overhead. Expected: recover most of the speed regression
while keeping the ~47% memory benefit.

---

## Track 2 — TurboQuant-Plus-Inspired Extensions

**Goal:** Improve the quality/compression tradeoff of KV quantization using ideas
from TurboQuant (arXiv:2504.19874) and turboquant_plus, adapted for ProtGPT2.

### Candidate Ideas

**2a. Asymmetric K/V precision**

*Motivation:* In the turbogene project, K and V activations showed different
statistical profiles: K outlier/median ratio was 5.5-8.6x, V was 6.9-11.7x.
Dual K/V lookup tables improved attention cosine similarity by +0.014 over shared.
For ProtGPT2, K and V may also have different quantization sensitivity.

*Experiment:* Keep V in INT8 (values are typically smoother) but try INT4 for K
(keys may tolerate more aggressive compression since they only participate in
dot-product scores, not directly in the output). Or vice versa — measure which
direction helps.

*Testable on v2 framework:* Yes. Modify `quantize_to_int8` to accept a bits
parameter and use different settings for K vs V in `ChunkedInt8KVCache.update()`.

**2b. Layer-aware compression / boundary-layer protection**

*Motivation:* In transformer models, early layers (embedding-adjacent) and final
layers (output-adjacent) often have different sensitivity to quantization than
middle layers. TurboQuant's ablation showed layer 0 had the worst KIVI baseline
(0.799 cos_sim vs 0.917 mean).

*Experiment:* Keep the first and last N layers (e.g., N=2) in FP16, compress
only middle layers to INT8. This protects the most sensitive layers while still
capturing the majority of the memory benefit (32 of 36 layers compressed).

*Testable on v2 framework:* Yes. Add a `protected_layers` parameter to
`ChunkedInt8KVCache` that stores specified layers in FP16 and skips quantization.

**2c. Orthogonal rotation before quantization**

*Motivation:* TurboQuant applies a random orthogonal rotation before scalar
quantization to distribute outlier energy across dimensions. This improved
attention cosine from 0.976 to 0.990 in turbogene.

*Experiment:* Apply a fixed random rotation matrix to K/V vectors before INT8
quantization, and inverse-rotate after dequantization. This should improve
quantization quality (fewer clipped outliers) at minimal runtime cost.

*Testable on v2 framework:* Yes, but adds O(head_dim^2) matmul per quantize/
dequantize call. May worsen the speed regression. Better suited for v5 (fused
kernel) where the rotation can be fused.

**2d. Per-head precision calibration**

*Motivation:* Different attention heads may have different quantization sensitivity.
Some heads may be safely compressed to 4-bit while others need 8-bit.

*Experiment:* Run a calibration pass on a small set of prompts, measure per-head
reconstruction error at different bit widths, then assign per-head precision.

*Testable on v2:* Partially — the chunked attention can handle mixed precision
per head, but the calibration step adds complexity.

### Priority Order

As standalone ideas ranked by implementation cost:
1. **2a (asymmetric K/V)** — lowest implementation cost, directly testable
2. **2b (boundary layers)** — low cost, clear architectural motivation
3. **2c (rotation)** — moderate cost, better suited for v5

**However, v3 implements 2b (boundary layers) before 2a.** Rationale: boundary-layer
protection is the cheapest *algorithm-design* test on the validated v2 chunked path.
It answers a structural question — whether the fused kernel needs mixed-precision
layer support — before we invest in asymmetric K/V precision tuning. If boundary
protection shows no quality benefit, we can drop it and simplify the kernel design.
If it helps, the kernel must support it. Asymmetric K/V (2a) is a precision-tuning
experiment that can be layered on top of either outcome and does not affect the
kernel architecture decision. Testing 2b first resolves the larger design question
at lower cost.
4. **2d (per-head calibration)** — higher cost, deferred

---

## Track 3 — Fused-Kernel Escalation Criteria

**When to move to Direction A (fused INT8 attention kernel):**

The escalation is justified when ALL of:

1. **Memory benefit confirmed:** >=40% reduction in generation overhead, sustained
   across multiple configs and sequence lengths. *Currently met (47%).*

2. **Behavioral stability confirmed:** >=99.99% top-1 logit agreement across at
   least 256 decode steps on at least 3 different prompts. *Partially met (100%
   agreement on 64 steps, 2 configs — needs extension).*

3. **Speed regression quantified:** The pure-PyTorch path is >2x slower than
   baseline, confirming that kernel fusion is the remaining bottleneck, not the
   algorithm. *Currently met (3-6x slower).*

4. **Algorithmic improvements exhausted:** v3 and v4 have been explored and the
   remaining speed gap cannot be closed without kernel-level changes.
   *Not yet met — v3 and v4 not attempted.*

**When NOT to escalate:**

- If v3 or v4 close the speed gap to <1.5x without kernel work
- If behavioral drift appears at longer test windows
- If the memory benefit drops below 30% after incorporating quality improvements

**Expected fused-kernel target:**
- Triton kernel for INT8 dequantize-fused SDPA
- Target: <20% speed overhead vs FP16 baseline, ~47% memory savings
- Estimated development effort: 2-4 days for a basic Triton implementation

---

## Track 4 — ProGen2 Engineering Path

**Separate from ProtGPT2.** ProGen2-small's bottleneck is non-KV overhead (~67%
of generation overhead), not KV cache. KV optimization is low priority.

### Next Steps

**4a. Replace `aten::cat` KV management with pre-allocated cache**

*Motivation:* ProGen2's custom `modeling_progen.py` concatenates new K/V tensors
onto the cache at each step (`torch.cat`). This allocates new memory, copies old
data, and frees old tensors — creating the ~4% `aten::cat` overhead seen in
operator profiling and contributing to the large non-KV baseline.

*Implementation:* Pre-allocate K/V cache tensors at max sequence length during
prefill, then write new K/V into pre-allocated slots. Eliminates per-step
allocation/copy.

**4b. Investigate non-KV runtime baseline**

*Motivation:* Even after pre-allocated cache, ~20-28 MB of non-KV overhead
persists. Use `torch.cuda.memory_snapshot()` or per-layer memory checkpoints
to identify what this overhead consists of.

**4c. Apply v2-style INT8 compression (if 4a/4b show KV becomes material)**

After engineering fixes reduce the non-KV baseline, re-evaluate whether KV
compression becomes worthwhile for ProGen2.

---

## Concrete Next Experiment: v3 Proposal

### What

**v3 = Asymmetric K/V precision + boundary-layer protection on the v2 chunked framework.**

Combine ideas 2a and 2b from Track 2:
- **Keys:** Quantize to INT8 (current behavior — unchanged)
- **Values:** Quantize to INT8 (current behavior — unchanged)
- **First 2 layers and last 2 layers:** Keep in FP16 (no quantization)
- **Middle 32 layers:** INT8 for both K and V (current behavior)

This is the minimal interesting configuration. After measuring this baseline,
a follow-up can test INT4 for keys in the middle layers.

### Why Before Fused Kernels

1. **Cheap to test:** Requires only adding a `protected_layers` parameter to
   `ChunkedInt8KVCache` — the chunked attention already handles FP16 chunks.
2. **Answers a quality question:** Does protecting boundary layers change the
   cosine similarity or token agreement at longer sequences?
3. **Answers a memory question:** With 4 of 36 layers in FP16, the memory savings
   drops from ~47% to ~41% (4/36 = 11% of layers uncompressed). Is this tradeoff
   worthwhile?
4. **Informs kernel design:** If boundary-layer protection matters, the fused
   kernel needs to support mixed-precision layers. Better to know this before
   building the kernel.

### Success Metric

- **Memory:** Generation overhead reduction >=35% (down from 47% due to protected layers)
- **Behavior:** Token agreement stays at 100% for 64 steps
- **Quality signal:** If cosine similarity at step 64 improves from 0.99999 to
  closer to 1.0, boundary protection has measurable value
- **If cosine is unchanged** (still ~0.99999), boundary protection adds no value
  for ProtGPT2 and can be dropped, simplifying the kernel design

### Stop / Escalate Criteria

- **Stop v3 and go to v4 (optimized PyTorch):** If boundary protection shows no
  quality improvement and the memory cost (47% → ~41%) is not justified
- **Stop and escalate to v5 (fused kernel):** If v3 confirms the algorithmic
  design is stable and the only remaining issue is speed
- **Stop entirely:** If behavioral drift appears at 64+ steps with boundary
  protection, indicating INT8 quantization is insufficient for ProtGPT2 and a
  different approach (e.g., FP8, mixed 4/8-bit) is needed

---

*This roadmap is a planning document. No implementation has started beyond v2.*
