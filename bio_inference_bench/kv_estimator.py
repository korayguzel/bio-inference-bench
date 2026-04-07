"""Theoretical KV cache estimation.

Computes formula-based KV cache size estimates. These are NEVER inferred
from observed CUDA memory — they are purely arithmetic from model config.

Formula:
    KV_bytes = 2 * num_layers * batch_size * seq_len * num_kv_heads * head_dim * dtype_bytes

The factor of 2 accounts for both K and V tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch


@dataclass
class KVEstimate:
    """Theoretical KV cache size estimate."""

    total_bytes: int
    total_mb: float
    per_token_bytes: int
    per_token_mb: float
    as_pct_of_weights: float  # filled by caller; 0.0 if unknown
    growth_per_100_tokens_mb: float


def dtype_to_bytes(dtype: torch.dtype) -> int:
    """Map torch dtype to bytes per element."""
    mapping = {
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.float32: 4,
        torch.float64: 8,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype for KV estimation: {dtype}")
    return mapping[dtype]


def estimate_kv_cache(
    num_layers: int,
    batch_size: int,
    seq_len: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,
) -> KVEstimate:
    """Compute theoretical KV cache size.

    Args:
        num_layers: Number of transformer layers.
        batch_size: Batch size.
        seq_len: Total sequence length (prompt + generated tokens).
        num_kv_heads: Number of key/value heads (may differ from query heads in GQA).
        head_dim: Dimension per attention head.
        dtype_bytes: Bytes per element (2 for fp16/bf16, 4 for fp32).
    """
    total_bytes = 2 * num_layers * batch_size * seq_len * num_kv_heads * head_dim * dtype_bytes
    per_token_bytes = 2 * num_layers * 1 * 1 * num_kv_heads * head_dim * dtype_bytes
    return KVEstimate(
        total_bytes=total_bytes,
        total_mb=total_bytes / (1024**2),
        per_token_bytes=per_token_bytes,
        per_token_mb=per_token_bytes / (1024**2),
        as_pct_of_weights=0.0,
        growth_per_100_tokens_mb=(per_token_bytes * 100) / (1024**2),
    )


def estimate_kv_table(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int,
    seq_lengths: list[int],
    batch_sizes: list[int],
) -> pd.DataFrame:
    """Generate a table of KV cache estimates across sequence lengths and batch sizes."""
    rows = []
    for bs in batch_sizes:
        for sl in seq_lengths:
            est = estimate_kv_cache(num_layers, bs, sl, num_kv_heads, head_dim, dtype_bytes)
            rows.append({
                "batch_size": bs,
                "seq_len": sl,
                "kv_cache_mb": round(est.total_mb, 2),
                "per_token_mb": round(est.per_token_mb, 4),
                "growth_per_100_tokens_mb": round(est.growth_per_100_tokens_mb, 2),
            })
    return pd.DataFrame(rows)
