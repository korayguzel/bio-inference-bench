# ProtGPT2 v5 Quickstart: INT8 KV Capacity Mode

## What is v5?

v5 is a **memory-saving mode** for running ProtGPT2 on local GPUs. It stores the
KV cache (the model's memory of previously generated tokens) in INT8 instead of
FP16, using a fused Triton kernel for attention. This cuts the decode-phase memory
growth by 60-97%, letting you generate longer sequences without running out of VRAM.

**This is a capacity optimization.** It makes more generation headroom available,
not faster generation. Decode speed is ~60% of FP16 baseline — the remaining gap
is in weight computations, not the attention kernel.

## When to use each mode

| Use case | Mode | Command flag |
|----------|------|-------------|
| Normal generation, speed matters most | FP16 | `--kv-mode fp16` (default) |
| Long sequences, VRAM is tight | INT8-Triton | `--kv-mode int8-triton` |
| See the memory difference | Compare | `--compare` |

## Minimal commands

```bash
# Install
cd /home/koray/projects/bio-inference-bench
source .venv/bin/activate

# Generate with FP16 (default)
python scripts/generate_protgpt2.py --prompt "MKTLLILAVL"

# Generate with INT8-Triton (lower VRAM)
python scripts/generate_protgpt2.py --prompt "MKTLLILAVL" --kv-mode int8-triton

# Compare both modes side-by-side (fresh model load per mode)
python scripts/generate_protgpt2.py --compare --max-new-tokens 256
```

## How to read the output

### Single-mode output

```
[int8-triton] 128 tokens in 1.8s (70 tok/s) | peak 1502 MB | decode growth 10.0 MB
```

- **128 tokens** — how many new tokens were generated
- **1.8s (70 tok/s)** — total time and decode throughput
- **peak 1502 MB** — highest VRAM usage during the run
- **decode growth 10.0 MB** — how much VRAM grew during the decode phase only
  (this is the capacity metric — lower = more headroom for longer sequences)

### Compare-mode capacity table

The `--compare` output has four sections:

1. **Measured run** — actual numbers from this run (growth/token, speed, compression)
2. **Model context limit** — max tokens ProtGPT2 can handle (1024 positions)
3. **VRAM for full context** — how much decode VRAM you'd need to fill the context window
4. **Slope projection** — theoretical extrapolation showing the memory-efficiency slope
   (these numbers are NOT achievable on ProtGPT2 — they show how the rate scales)

The key number is **decode growth/token**: FP16 uses ~0.18 MB per token, INT8-Triton
uses ~0.005-0.08 MB per token (varies with sequence length). That's the capacity win.

## Limitations that matter most for local GPU users

1. **ProtGPT2 only.** The Triton kernel is validated for ProtGPT2's architecture
   (20 heads, 36 layers, head_dim=64). It will refuse to run on other models.

2. **Slower decode.** Expect ~60% of FP16 speed. If you need maximum throughput
   and have enough VRAM, use `--kv-mode fp16`.

3. **Prefill still uses FP16.** The INT8 savings only apply during the decode phase.
   For very long prompts, the prefill can temporarily use more VRAM than the final
   decode state.

4. **Needs Triton.** Requires an NVIDIA GPU with Triton support (ships with PyTorch).

5. **Context limit: 1024 tokens.** ProtGPT2 has a 1024-position context window.
   The INT8 path doesn't change this — it just makes reaching the full context
   cheaper in VRAM.

## Files to know

| File | What it does |
|------|-------------|
| `scripts/generate_protgpt2.py` | User-facing CLI (start here) |
| `bio_inference_bench/int8_generate.py` | INT8 generation orchestrator |
| `bio_inference_bench/triton_int8_attention.py` | Fused Triton kernel |
| `bio_inference_bench/kv_int8_chunked.py` | INT8 KV cache + decode step |
