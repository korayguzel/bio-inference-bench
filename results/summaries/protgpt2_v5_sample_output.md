# Sample Output: ProtGPT2 v5 Compare Mode

Captured on RTX 4070 (12 GB), 128 new tokens from default prompt.

```
$ python scripts/generate_protgpt2.py --compare --max-new-tokens 128
```

## Console Output

```
ProtGPT2 v5 — Capacity Comparison
GPU: NVIDIA GeForce RTX 4070 (11864 MB)

Comparing FP16 baseline vs INT8-Triton v5
Each mode uses a fresh model load for clean memory measurement.

  [1/2] FP16 baseline...
  [fp16] 128 tokens in 1.0s (129 tok/s) | peak 1513 MB | decode growth 22.7 MB

  [2/2] INT8-Triton v5...
  [int8-triton] 128 tokens in 1.8s (70 tok/s) | peak 1502 MB | decode growth 10.0 MB

ProtGPT2 INT8 KV Capacity Report — NVIDIA GeForce RTX 4070 (11864 MB)
============================================================
                                 FP16 Baseline   INT8-Triton v5
------------------------------------------------------------
Decode growth/token                  0.1790 MB        0.0786 MB
Decode speed                           129 tok/s        70 tok/s (54%)
Cache compression ratio                  1.00x            1.94x

Measured: 128 new tokens from 11-token prompt

Model context limit: 1024 tokens (max_position_embeddings)
Feasible new tokens for this prompt: 1013 (= 1024 - 11)

Decode VRAM for full context (1013 new tokens):
  FP16:         ~ 181.3 MB
  INT8-Triton:  ~  79.6 MB
  Savings:      ~101.7 MB (56%)

Slope-based VRAM projection (theoretical headroom, NOT achievable on ProtGPT2):
  1 GB free VRAM       ~   5,720 tokens    ~  13,027 tokens
  2 GB free VRAM       ~  11,441 tokens    ~  26,055 tokens
  4 GB free VRAM       ~  22,882 tokens    ~  52,111 tokens
  These are uncapped slope extrapolations showing how the decode growth
  rate scales with VRAM. ProtGPT2 is limited to 1024 total positions.
============================================================

Raw saved to: results/raw/capacity_compare_20260406T225314.json
```

## How to read this

**Summary lines:** Each `[mode] N tokens in Xs (Y tok/s) | peak Z MB | decode growth W MB`
shows a single run. The key comparison is `decode growth`: 22.7 MB (FP16) vs 10.0 MB
(INT8-Triton) — the INT8 path used ~56% less decode-phase VRAM.

**Capacity table:**
- `Decode growth/token` is the slope — how fast VRAM grows per generated token
- `Feasible new tokens` is capped by ProtGPT2's 1024-position context limit
- `Decode VRAM for full context` shows what filling the context window would cost
- `Slope-based projection` shows the theoretical scaling (not achievable on ProtGPT2)

**Note on growth/token values:** At short sequence lengths (128 tokens), the per-token
growth includes amortized overhead. At longer sequences (256-512 tokens), the INT8
growth/token drops further because the overhead is spread over more tokens. The
benchmark eval results at 512 tokens show ~0.005 MB/token for INT8 vs ~0.18 MB/token
for FP16.
