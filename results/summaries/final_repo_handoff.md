# bio-inference-bench: Final Repository Handoff

**Date:** 2026-04-07
**Status:** Finalized project snapshot. No active development tracks.

---

## Canonical inference path

**ProtGPT2 + FP16 weights + INT8-Triton KV cache**

```bash
# Generate with INT8 KV (recommended for capacity)
python scripts/generate_protgpt2.py --prompt "MKTLLILAVL" --kv-mode int8-triton

# Compare FP16 vs INT8 side-by-side
python scripts/generate_protgpt2.py --compare --max-new-tokens 256

# FP16 only (when speed matters more than VRAM)
python scripts/generate_protgpt2.py --prompt "MKTLLILAVL" --kv-mode fp16
```

## Start here

### If you want to use the tool

1. Read `results/summaries/protgpt2_v5_quickstart.md`
2. Run `python scripts/generate_protgpt2.py --compare` to see the capacity table
3. Use `--kv-mode int8-triton` for capacity, `--kv-mode fp16` for speed

### If you want to understand the project

1. Read `results/summaries/final_project_summary.md` — full project arc
2. Read `results/summaries/protgpt2_v5_handoff.md` — technical handoff
3. Read `results/summaries/protgpt2_weight_quant_track_closure.md` — why weight quant failed

### If you want to benchmark

1. `python scripts/benchmark_generation.py --model protgpt2 --kv-mode int8-triton`
2. Raw JSON output goes to `results/raw/`

## Main scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/generate_protgpt2.py` | User-facing CLI (`--prompt`, `--kv-mode`, `--compare`) | **Primary entry point** |
| `scripts/benchmark_generation.py` | Structured benchmark (`--model`, `--kv-mode`) | **Active** |
| `scripts/run_smoke.py` | Harness validation (ProtGPT2 + ProGen2-small) | Active |
| `scripts/inspect_model.py` | Model metadata and theoretical KV cache | Active |
| `scripts/run_grid.py` | Full benchmark grid | Active |
| `scripts/eval_kv_fused_v5.py` | v5 three-way eval (baseline/v2/Triton) | Historical |
| `scripts/eval_weight_quant_phase1.py` | NF4 four-way comparison | Historical (track closed) |
| `scripts/eval_weight_quant_phase2.py` | bnb-int8 four-way comparison | Historical (track closed) |
| `scripts/eval_weight_quant_phase3.py` | torchao INT8 four-way comparison | Historical (track closed) |
| `scripts/tune_block_kv.py` | Triton kernel BLOCK_KV sweep | Historical (BLOCK_KV=64 confirmed) |
| `scripts/eval_kv_prototype.py` | v1 KV prototype eval | Historical |
| `scripts/eval_kv_prototype_v2.py` | v2 KV prototype eval | Historical |
| `scripts/eval_kv_prototype_v3.py` | v3 boundary protection eval | Historical (dropped) |

## Core library

| File | What it does | Modify with care? |
|------|-------------|-------------------|
| `bio_inference_bench/triton_int8_attention.py` | Fused Triton kernel | Yes — 39 tests validate it |
| `bio_inference_bench/kv_int8_chunked.py` | INT8 KV cache + decode dispatch | Yes — Triton/v2 dispatch logic |
| `bio_inference_bench/int8_generate.py` | INT8 generation orchestrator | Yes — dual-peak memory tracking |
| `bio_inference_bench/models.py` | Model loading (FP16, bnb-nf4, bnb-int8, torchao-int8) | Yes — validation gate |
| `bio_inference_bench/generation.py` | FP16 baseline path | Yes — do not add INT8 here |
| `bio_inference_bench/eval_helpers.py` | Shared eval logic (Phases 1-3) | Low risk |
| `bio_inference_bench/profiler.py` | CUDA memory profiling | Low risk |
| `bio_inference_bench/report.py` | Output formatting, capacity table | Low risk |
| `bio_inference_bench/kv_estimator.py` | Theoretical KV cache estimation | Low risk |
| `bio_inference_bench/utils.py` | Constants, model registry, helpers | Low risk |
| `bio_inference_bench/kv_int8_cache.py` | v1 KV cache (historical, failed) | Historical |
| `bio_inference_bench/progen2_compat.py` | ProGen2 compatibility shims | Historical |

## Reports — reading order

### Essential (read these)

| Report | What it covers |
|--------|---------------|
| `final_project_summary.md` | Full project arc, what succeeded, what failed, takeaways |
| `protgpt2_v5_handoff.md` | Technical handoff: canonical path, files, what not to change |
| `protgpt2_weight_quant_track_closure.md` | Why weight quantization was closed |
| `protgpt2_v5_quickstart.md` | How to use the CLI |

### Background (for deeper context)

| Report | What it covers |
|--------|---------------|
| `protgpt2_fused_kernel_design.md` | Triton kernel design decisions |
| `protgpt2_fused_kernel_v5_report.md` | Phase B benchmark (v5 vs baseline vs v2) |
| `protgpt2_fused_kernel_phase_c_report.md` | Phase C tuning (BLOCK_KV sweep) |
| `v5_productization_plan.md` | Productization design decisions |
| `weight_quantized_track_plan.md` | Weight-quant planning memo (closed) |
| `turboquant_advanced_roadmap.md` | Original four-track roadmap (partially completed) |

### Historical (negative results, earlier iterations)

| Report | What it covers |
|--------|---------------|
| `weight_quant_phase1_report.md` | NF4 results (failed quality on long prompts) |
| `weight_quant_phase2_report.md` | bnb-int8 results (catastrophic quality failure) |
| `weight_quant_phase3_report.md` | torchao INT8 results (quality + stacking failure) |
| `protgpt2_kv_prototype_report.md` | v1 KV cache (failed memory coexistence) |
| `protgpt2_kv_prototype_v2_report.md` | v2 chunked INT8 (succeeded, slow) |
| `protgpt2_kv_prototype_v3_report.md` | v3 boundary protection (no benefit, dropped) |
| `optimization_candidate_memo.md` | Initial optimization candidates analysis |
| `progen2_compatibility_report.md` | ProGen2 compatibility findings |
| `grid_report_*.md` | Benchmark grid results |
| `mentor_packet.md` | Initial mentor review packet |

## Key raw artifacts

| File | Significance |
|------|-------------|
| `kv_fused_v5_eval_20260406T214141.json` | Authoritative v5 benchmark (Phase B corrected) |
| `weight_quant_phase1_20260406T231058.json` | NF4 four-way comparison |
| `weight_quant_phase2_20260407T100546.json` | bnb-int8 four-way comparison |
| `weight_quant_phase3_20260407T103724.json` | torchao INT8 four-way comparison |
| `kv_prototype_v2_eval_20260406T195958.json` | v2 baseline (pre-Triton reference) |

All other `results/raw/*.json` files are grid/smoke/prototype run artifacts. They
are preserved for reproducibility but are not referenced by the final reports.

## What not to change casually

1. **Dual-peak memory tracking** in `int8_generate.py` — `reset_peak_memory_stats()`
   between prefill and decode is what makes `decode_phase_growth_mb` correct
2. **ProtGPT2 validation** in `_validate_protgpt2()` — checks model name + architecture
3. **`generation.py`** — FP16 reference path, no INT8 logic belongs here
4. **Triton kernel BLOCK_KV=64** — confirmed optimal by Phase C sweep
5. **Fresh model loads** in `--compare` mode — intentional for clean memory measurement

## What not to reopen casually

1. **ProtGPT2 weight quantization** — three methods failed, track closed with
   documented rationale
2. **Boundary-layer KV protection** — v3 showed no quality benefit
3. **ProGen2 optimization** — different bottleneck profile, not started, deferred

## Tests

| File | Coverage |
|------|---------|
| `tests/test_triton_int8_attention.py` | 39 tests: Triton vs FP16, vs v2, determinism, shapes, edge cases |

Run with: `cd /home/koray/projects/bio-inference-bench && .venv/bin/python -m pytest tests/ -v`

## Dependencies

Key packages (from `.venv`):
- `torch` 2.11.0+cu130
- `transformers` 5.x
- `triton` (ships with torch)
- `bitsandbytes` (for weight-quant experiments; not needed for canonical path)
- `torchao` 0.17.0 (for Phase 3 weight-quant; not needed for canonical path)

The canonical path (FP16 weights + INT8-Triton KV) requires only `torch` and
`transformers`. `bitsandbytes` and `torchao` are only needed to reproduce the
weight-quantization experiments.

---
*Final repository handoff completed 2026-04-07.*
