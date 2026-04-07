"""Shared constants, configuration, and helper functions."""

from __future__ import annotations

import datetime
from typing import Any

import torch

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# First ~120 residues of human ubiquitin (P0CG48) — a well-characterized,
# biologically real protein fragment used as a reproducible prompt source.
DEFAULT_PROMPT_SEQUENCE = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQ"
    "KESTLHLVLRLRGGIIEPSLKALASKYNCDKSVCRKCYARLPPRATNCRKRKCGHTNQLRPK"
)

# Canonical smoke benchmark configuration.
# prompt_token_length is in *tokens*, not amino-acid characters.
SMOKE_CONFIG: dict[str, Any] = {
    "prompt_token_length": 64,
    "max_new_tokens": 128,
    "batch_size": 1,
    "do_sample": False,
    "num_beams": 1,
    "use_cache": True,
}

# Model registry with ordered fallback candidates.
# For each model, candidates are tried in order until one succeeds.
# Both model and tokenizer must load for a candidate to be accepted.
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "protgpt2": {
        "candidates": [
            {"hf_path": "nferruz/ProtGPT2", "trust_remote_code": False},
        ],
    },
    "progen2-small": {
        "candidates": [
            {"hf_path": "hugohrban/progen2-small", "trust_remote_code": True},
            {"hf_path": "multimolecule/progen2-small", "trust_remote_code": True},
        ],
    },
}


def get_device() -> torch.device:
    """Return CUDA device. Raises RuntimeError if no GPU is available."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "bio-inference-bench requires a CUDA GPU. No CUDA device found."
        )
    return torch.device("cuda")


def format_bytes(n: int | float) -> str:
    """Format byte count as human-readable string (MB or GB)."""
    mb = n / (1024**2)
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.2f} MB"


def timestamp() -> str:
    """Return ISO-8601 timestamp suitable for filenames."""
    return datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
