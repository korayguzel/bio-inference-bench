#!/usr/bin/env python3
"""Operator-level profiling for representative benchmark configurations.

Runs torch.profiler separately on prefill and decode phases for a small
set of configs chosen from Benchmark v1 grid results. Produces per-phase
operator summaries (top CUDA-time operators) and memory signals.

This is decision-support profiling, not a full benchmark rerun.

Usage:
    python scripts/profile_representative_configs.py
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoModelForCausalLM

from bio_inference_bench.models import (
    _extract_metadata,
    _set_eval_mode,
    _try_load_tokenizer,
    prepare_prompt,
)
from bio_inference_bench.progen2_compat import patch_progen2_model
from bio_inference_bench.profiler import get_gpu_info, reset_memory_tracking
from bio_inference_bench.utils import MODEL_REGISTRY, get_device, timestamp

logger = logging.getLogger(__name__)

# Representative configs chosen from Benchmark v1 grid analysis.
REPRESENTATIVE_CONFIGS = [
    # ProtGPT2 decode-heavy: runtime KV ratio ~98.5%, peak in decode
    {"model": "protgpt2", "prompt": 32, "max_new": 256, "label": "protgpt2_decode_heavy"},
    # ProtGPT2 prefill-heavy: runtime KV ratio ~71.9%, peak in prefill
    {"model": "protgpt2", "prompt": 256, "max_new": 32, "label": "protgpt2_prefill_heavy"},
    # ProtGPT2 long decode: longest config, runtime KV ratio ~98.0%, peak in decode
    {"model": "protgpt2", "prompt": 256, "max_new": 512, "label": "protgpt2_long_decode"},
    # ProGen2-small representative: runtime KV ratio ~33.4%, peak in decode
    {"model": "progen2-small", "prompt": 64, "max_new": 128, "label": "progen2_representative"},
    # ProGen2-small long decode: runtime KV ratio ~32.2%, peak in decode
    {"model": "progen2-small", "prompt": 256, "max_new": 512, "label": "progen2_long_decode"},
]

# How many decode steps to profile (not the full generation — just enough
# for operator statistics). Profiling every step is expensive.
PROFILED_DECODE_STEPS = 32


@dataclass
class PhaseProfile:
    """Summarized profiler output for one phase."""
    phase: str  # "prefill" or "decode"
    label: str
    model_name: str
    prompt_token_length: int
    max_new_tokens: int
    top_cuda_ops: list[dict]
    top_cpu_ops: list[dict]
    total_cuda_time_us: float
    total_cpu_time_us: float
    device_mem_self_bytes: int  # sum of per-op self CUDA memory allocations


def extract_top_ops(key_averages, sort_by: str, top_n: int = 15,
                    exclude_wrappers: tuple[str, ...] = ("prefill", "decode")) -> list[dict]:
    """Extract top operators from profiler key_averages.

    Filters out record_function wrappers (e.g. 'prefill', 'decode') so that
    percentages reflect real operator costs.
    """
    rows = []
    for evt in key_averages:
        if evt.key in exclude_wrappers:
            continue
        if sort_by == "cuda" and evt.device_time_total == 0:
            continue
        rows.append({
            "name": evt.key,
            "calls": evt.count,
            "cpu_time_us": round(evt.cpu_time_total, 1),
            "cuda_time_us": round(evt.device_time_total, 1),
            "cpu_time_pct": 0.0,
            "cuda_time_pct": 0.0,
        })
    if sort_by == "cuda":
        rows.sort(key=lambda x: x["cuda_time_us"], reverse=True)
    else:
        rows.sort(key=lambda x: x["cpu_time_us"], reverse=True)

    total_cuda = sum(r["cuda_time_us"] for r in rows) or 1
    total_cpu = sum(r["cpu_time_us"] for r in rows) or 1
    for r in rows:
        r["cuda_time_pct"] = round(r["cuda_time_us"] / total_cuda * 100, 1)
        r["cpu_time_pct"] = round(r["cpu_time_us"] / total_cpu * 100, 1)

    return rows[:top_n]


def profile_prefill(
    model, input_ids: torch.Tensor, label: str, model_name: str,
    prompt_token_length: int, max_new_tokens: int,
) -> PhaseProfile:
    """Profile the prefill phase (single forward pass over prompt)."""
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    # Warm-up (outside profiler)
    with torch.no_grad():
        _ = model(input_ids, use_cache=True)
    reset_memory_tracking()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with record_function("prefill"):
            with torch.no_grad():
                outputs = model(input_ids, use_cache=True)
            torch.cuda.synchronize()

    ka = prof.key_averages()
    total_cuda = sum(e.device_time_total for e in ka)
    total_cpu = sum(e.cpu_time_total for e in ka)
    mem_alloc = sum(e.self_device_memory_usage for e in ka if hasattr(e, 'self_device_memory_usage') and e.self_device_memory_usage > 0)

    return PhaseProfile(
        phase="prefill",
        label=label,
        model_name=model_name,
        prompt_token_length=prompt_token_length,
        max_new_tokens=max_new_tokens,
        top_cuda_ops=extract_top_ops(ka, "cuda"),
        top_cpu_ops=extract_top_ops(ka, "cpu"),
        total_cuda_time_us=round(total_cuda, 1),
        total_cpu_time_us=round(total_cpu, 1),
        device_mem_self_bytes=mem_alloc,
    )


def profile_decode(
    model, input_ids: torch.Tensor, label: str, model_name: str,
    prompt_token_length: int, max_new_tokens: int, steps: int = PROFILED_DECODE_STEPS,
) -> PhaseProfile:
    """Profile the decode phase (token-by-token with past_key_values)."""
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    # Run prefill first (outside profiler) to get past_key_values
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
    past_key_values = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # Warm-up one decode step (outside profiler)
    with torch.no_grad():
        outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
    past_key_values = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    reset_memory_tracking()

    actual_steps = min(steps, max_new_tokens - 2)  # subtract prefill token + warmup

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with record_function("decode"):
            with torch.no_grad():
                for _ in range(actual_steps):
                    outputs = model(
                        next_token, past_key_values=past_key_values, use_cache=True
                    )
                    past_key_values = outputs.past_key_values
                    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            torch.cuda.synchronize()

    ka = prof.key_averages()
    total_cuda = sum(e.device_time_total for e in ka)
    total_cpu = sum(e.cpu_time_total for e in ka)
    mem_alloc = sum(e.self_device_memory_usage for e in ka if hasattr(e, 'self_device_memory_usage') and e.self_device_memory_usage > 0)

    # Clean up KV cache
    del past_key_values, outputs
    reset_memory_tracking()

    return PhaseProfile(
        phase=f"decode ({actual_steps} steps)",
        label=label,
        model_name=model_name,
        prompt_token_length=prompt_token_length,
        max_new_tokens=max_new_tokens,
        top_cuda_ops=extract_top_ops(ka, "cuda"),
        top_cpu_ops=extract_top_ops(ka, "cpu"),
        total_cuda_time_us=round(total_cuda, 1),
        total_cpu_time_us=round(total_cpu, 1),
        device_mem_self_bytes=mem_alloc,
    )


def phase_profile_to_dict(pp: PhaseProfile) -> dict:
    return {
        "phase": pp.phase,
        "label": pp.label,
        "model_name": pp.model_name,
        "prompt_token_length": pp.prompt_token_length,
        "max_new_tokens": pp.max_new_tokens,
        "top_cuda_ops": pp.top_cuda_ops,
        "top_cpu_ops": pp.top_cpu_ops,
        "total_cuda_time_us": pp.total_cuda_time_us,
        "total_cpu_time_us": pp.total_cpu_time_us,
        "device_mem_self_bytes": pp.device_mem_self_bytes,
    }


def load_and_patch(model_name: str, device: torch.device):
    """Load model with candidate fallback and patches."""
    entry = MODEL_REGISTRY[model_name]
    for candidate in entry["candidates"]:
        hf_path = candidate["hf_path"]
        trust_remote_code = candidate.get("trust_remote_code", False)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                hf_path, trust_remote_code=trust_remote_code, torch_dtype=torch.float16,
            ).to(str(device))
            _set_eval_mode(model)
        except Exception:
            continue
        tokenizer, _ = _try_load_tokenizer(hf_path, trust_remote_code)
        if tokenizer is None:
            del model; torch.cuda.empty_cache()
            continue
        patch_progen2_model(model)
        metadata = _extract_metadata(model, hf_path, model_name, torch.float16)
        return model, tokenizer, metadata
    return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("  Targeted Operator Profiling (Benchmark v1 representative configs)")
    print("=" * 60)

    device = get_device()
    gpu_info = get_gpu_info()
    print(f"\nGPU: {gpu_info['name']} ({gpu_info['total_mb']:.0f} MB)")
    print(f"Configs: {len(REPRESENTATIVE_CONFIGS)}")
    print(f"Profiled decode steps: {PROFILED_DECODE_STEPS}")
    print()

    all_profiles: list[dict] = []
    loaded_models: dict = {}

    for cfg in REPRESENTATIVE_CONFIGS:
        model_name = cfg["model"]
        label = cfg["label"]
        ptl = cfg["prompt"]
        mnt = cfg["max_new"]

        print(f"\n{'─' * 60}")
        print(f"  {label}: {model_name} prompt={ptl} max_new={mnt}")
        print(f"{'─' * 60}")

        # Load model (reuse if same model)
        if model_name not in loaded_models:
            if loaded_models:
                # Unload previous model
                prev = list(loaded_models.values())[0]
                del prev
                loaded_models.clear()
                torch.cuda.empty_cache()

            result = load_and_patch(model_name, device)
            if result is None:
                print(f"  FAILED: Could not load {model_name}")
                continue
            loaded_models[model_name] = result

        model, tokenizer, metadata = loaded_models[model_name]

        # Prepare prompt
        input_ids = prepare_prompt(tokenizer, prompt_token_length=ptl)

        # Profile prefill
        print(f"  Profiling prefill...", end="", flush=True)
        prefill_profile = profile_prefill(
            model, input_ids, label, model_name, ptl, mnt
        )
        print(f" done ({prefill_profile.total_cuda_time_us:.0f} us CUDA)")

        # Profile decode
        print(f"  Profiling decode ({PROFILED_DECODE_STEPS} steps)...", end="", flush=True)
        decode_profile = profile_decode(
            model, input_ids, label, model_name, ptl, mnt
        )
        print(f" done ({decode_profile.total_cuda_time_us:.0f} us CUDA)")

        # Print top-3 CUDA ops for quick feedback
        print(f"\n  Top-3 prefill CUDA ops:")
        for op in prefill_profile.top_cuda_ops[:3]:
            print(f"    {op['cuda_time_pct']:5.1f}%  {op['name']}")
        print(f"  Top-3 decode CUDA ops:")
        for op in decode_profile.top_cuda_ops[:3]:
            print(f"    {op['cuda_time_pct']:5.1f}%  {op['name']}")

        entry = {
            "label": label,
            "model_name": model_name,
            "prompt_token_length": ptl,
            "max_new_tokens": mnt,
            "prefill": phase_profile_to_dict(prefill_profile),
            "decode": phase_profile_to_dict(decode_profile),
        }
        all_profiles.append(entry)

        # Save per-config profile
        profile_path = Path("results/profiles") / f"{label}_{timestamp()}.json"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        with open(profile_path, "w") as f:
            json.dump(entry, f, indent=2)

    # Cleanup
    for v in loaded_models.values():
        del v
    loaded_models.clear()
    torch.cuda.empty_cache()

    # Save combined profiles
    ts = timestamp()
    combined_path = Path("results/profiles") / f"all_profiles_{ts}.json"
    with open(combined_path, "w") as f:
        json.dump(all_profiles, f, indent=2)
    print(f"\n\nAll profiles saved to: {combined_path}")
    print(f"Individual profiles in: results/profiles/")

    print(f"\n{'=' * 60}")
    print(f"  Profiling complete: {len(all_profiles)} configs")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
