# v5 Productization Plan: INT8 KV Capacity Path for ProtGPT2

**Date:** 2026-04-06
**Status:** Implemented
**Decision:** TurboQuant capacity path is the primary direction for bio-inference-bench.

---

## 1. Project Decision

The fused INT8 KV attention path (v5) is the canonical capacity optimization for
ProtGPT2. It reduces decode-phase memory growth by 60-97% vs FP16 baseline, enabling
longer sequences on memory-constrained GPUs. The remaining 36-42% speed cost is
dominated by weight GEMMs (not attention) and is accepted as the capacity tradeoff.

**Not in scope:** GEMM optimization, weight quantization, boundary-layer protection,
asymmetric K/V, new research branches, ProGen2 work.

---

## 2. v5 as an Explicit Inference Mode

The INT8 KV path is exposed as a named `kv_mode` parameter:

```python
from bio_inference_bench import generate, load_model_and_tokenizer

model, tokenizer, metadata = load_model_and_tokenizer("protgpt2", device="cuda", dtype=torch.float16)
input_ids = tokenizer.encode("MKTLLILAVL", return_tensors="pt")

# FP16 baseline
result_fp16 = generate(model, tokenizer, input_ids, max_new_tokens=256, kv_mode="fp16")

# INT8-Triton capacity mode
result_int8 = generate(model, tokenizer, input_ids, max_new_tokens=256, kv_mode="int8-triton")
```

**Implementation:** `bio_inference_bench/int8_generate.py`
- `generate()` — unified dispatcher routing to FP16 or INT8-Triton based on `kv_mode`
- `generate_int8()` — full pipeline: prefill → KV transfer → INT8 decode loop
- `_validate_protgpt2()` — model name + architecture gate (n_layer=36, n_head=20, head_dim=64)

---

## 3. CLI Surface

**Primary user script:** `scripts/generate_protgpt2.py`

```bash
# Real protein prompt (primary input mode)
python scripts/generate_protgpt2.py --prompt "MKTLLILAVL" --kv-mode int8-triton

# Compare both modes (fresh model load per mode, capacity table)
python scripts/generate_protgpt2.py --compare --max-new-tokens 256

# Read prompt from file
python scripts/generate_protgpt2.py --prompt-file sequences.txt --kv-mode int8-triton

# Benchmark fallback (synthetic prompt)
python scripts/generate_protgpt2.py --prompt-token-length 64 --kv-mode int8-triton
```

**Benchmark integration:** `scripts/benchmark_generation.py --kv-mode int8-triton`
- Gated to ProtGPT2 only (fails with clear error on other models)
- Preserves established benchmark semantics (structured JSON, dual memory metrics)

---

## 4. Configuration Flags

| Parameter | User-configurable | Default | Notes |
|-----------|:-:|---------|-------|
| `--prompt` | Yes | — | Direct protein sequence (**primary input**) |
| `--prompt-file` | Yes | — | Read prompt from file |
| `--prompt-token-length` | Yes | — | Benchmark fallback: synthetic prompt |
| `--kv-mode` | Yes | `fp16` | `fp16` or `int8-triton` |
| `--max-new-tokens` | Yes | 256 | Decode length |
| `--chunk-size` | Yes (advanced) | 64 | INT8 KV block size |
| `--compare` | Yes | off | Run both modes, capacity table |
| model | No | protgpt2 | int8-triton gated to ProtGPT2 |
| batch_size | No | 1 | Always 1 |
| protected_layers | No | empty | Dropped (v3 showed no benefit) |

`--prompt`, `--prompt-file`, and `--prompt-token-length` are mutually exclusive.

---

## 5. Memory Reporting Semantics

Two overhead metrics reported for all paths, using the same definitions:

| Metric | Definition | Purpose |
|--------|-----------|---------|
| `end_to_end_generation_overhead_mb` | `max(prefill_peak, decode_peak) - before_gen` | Full high-water mark |
| `decode_phase_growth_mb` | `decode_peak - after_prefill` | **Primary capacity metric** |

Prefill and decode peaks are tracked independently via `torch.cuda.reset_peak_memory_stats()`
between phases.

---

## 6. Capacity Table

The `--compare` flag produces a capacity comparison table:

```
ProtGPT2 INT8 KV Capacity Report — NVIDIA GeForce RTX 4070 (11864 MB)
============================================================
                                 FP16 Baseline   INT8-Triton v5
------------------------------------------------------------
Decode growth/token                  0.18 MB          0.005 MB
Decode speed                         125 tok/s         79 tok/s (63%)
Cache compression ratio                1.00x            1.94x

Measured: 256 new tokens from 64-token prompt

Model context limit: 1024 tokens (max_position_embeddings)
Feasible new tokens for this prompt: 960 (= 1024 - 64)

Decode VRAM for full context (960 new tokens):
  FP16:         ~172.6 MB
  INT8-Triton:  ~  4.8 MB
  Savings:      ~167.8 MB (97%)

Slope-based VRAM projection (theoretical headroom, NOT achievable on ProtGPT2):
  1 GB free VRAM       ~5,500 tokens     ~200,000 tokens
  2 GB free VRAM       ~11,000 tokens    ~400,000 tokens
  These are uncapped slope extrapolations showing how the decode growth
  rate scales with VRAM. ProtGPT2 is limited to 1024 total positions.
============================================================
```

**Capacity table design (four sections):**
1. **Measured run summary** — decode growth/token, speed, compression from actual run
2. **Model context limit** — feasible new tokens capped at `max_position_embeddings - prompt_len`
3. **VRAM for feasible context** — decode VRAM needed for the capped token count
4. **Slope-based VRAM projection** — uncapped extrapolations, explicitly labeled as
   theoretical headroom not achievable on ProtGPT2

Key constraints:
- Feasible tokens are always capped by the model's context limit
- VRAM projection rows are explicitly uncapped and labeled as such
- `decode_growth_per_token_mb` computed from actual run, not theoretical

---

## 7. User-Facing Limitations

1. **ProtGPT2 only** (774M params, GPT-2 architecture, 20 heads, 36 layers, head_dim=64)
2. **Batch size = 1** (single-sequence decode)
3. **Prefill always uses FP16** — INT8 is decode-only
4. **Requires NVIDIA GPU with Triton support**
5. **Decode speed is 58-64% of FP16 baseline** — weight GEMMs dominate, not attention
6. **Quality validated through 256 decode steps** — longer sequences not yet tested
7. **This is a capacity optimization, not a speed optimization**

---

## 8. Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `bio_inference_bench/int8_generate.py` | Created | INT8 generation orchestrator + unified `generate()` |
| `scripts/generate_protgpt2.py` | Created | User-facing CLI with `--compare` |
| `scripts/benchmark_generation.py` | Modified | Added `--kv-mode` flag with ProtGPT2 gate |
| `bio_inference_bench/report.py` | Modified | Added `format_capacity_table()` and `format_generation_summary()` |
| `bio_inference_bench/__init__.py` | Modified | Exports `generate` and `load_model_and_tokenizer` |
| `README.md` | Modified | Added INT8 KV section + limitations |

**Unchanged:** `generation.py` (baseline stays clean), `kv_int8_chunked.py`, `triton_int8_attention.py`.

---

## 9. Evidence Base

| Artifact | Key Finding |
|----------|-------------|
| v2 eval (20260406T195958) | -47% generation overhead, 100% token agreement |
| v3 eval (20260406T202004) | Boundary protection unnecessary for ProtGPT2 |
| v5 Phase B (20260406T214141) | 3.68x speedup over v2, zero memory regression |
| Phase C tuning (20260406) | BLOCK_KV=64 optimal, kernel is only ~5% of step time |
| Phase A tests (39/39 passed) | Triton kernel correctness validated at all edge cases |

---
*Productization plan completed 2026-04-06.*
*All implementation verified against 11-point checklist.*
