"""Fused INT8 KV attention kernel (Triton) — Phase A: correctness only.

Replaces the Python chunked attention loop in ChunkedInt8KVCache with a single
fused Triton kernel that:
1. Reads INT8 KV + FP16 scales from global memory
2. Dequantizes to FP16 in SRAM
3. Computes Q·K^T scores
4. Applies online softmax (Milakov & Gimelshein 2018)
5. Accumulates weighted V output
6. Writes final attention output

Scope: batch=1, q_len=1 (single-token decode), MHA, exact attention.
No performance tuning — Phase A is correctness validation only.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _int8_kv_attention_kernel(
    # Q: (batch, heads, 1, head_dim) — but we index per (batch, head) program
    Q_ptr,
    # K_int8: (batch, heads, seq_len, head_dim) — int8
    K_ptr,
    # K_scales: (batch, heads, seq_len, 1) — fp16
    K_scales_ptr,
    # V_int8: (batch, heads, seq_len, head_dim) — int8
    V_ptr,
    # V_scales: (batch, heads, seq_len, 1) — fp16
    V_scales_ptr,
    # Output: (batch, heads, 1, head_dim) — fp16
    Out_ptr,
    # Strides for Q: (batch_stride, head_stride, seq_stride, dim_stride)
    q_stride_b, q_stride_h, q_stride_s, q_stride_d,
    # Strides for K_int8
    k_stride_b, k_stride_h, k_stride_s, k_stride_d,
    # Strides for K_scales
    ks_stride_b, ks_stride_h, ks_stride_s, ks_stride_d,
    # Strides for V_int8
    v_stride_b, v_stride_h, v_stride_s, v_stride_d,
    # Strides for V_scales
    vs_stride_b, vs_stride_h, vs_stride_s, vs_stride_d,
    # Strides for Out
    o_stride_b, o_stride_h, o_stride_s, o_stride_d,
    # Attention scaling factor (1/sqrt(head_dim))
    scaling,
    # Sequence length of KV cache
    seq_len,
    # Constants
    HEAD_DIM: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """Single-query decode attention over INT8 KV cache with online softmax."""
    # Grid = (batch * heads,). With batch=1, pid maps directly to head index.
    pid = tl.program_id(0)
    batch_idx = 0  # batch=1 assumption
    head_idx = pid

    # Base pointers for this (batch, head)
    q_base = Q_ptr + batch_idx * q_stride_b + head_idx * q_stride_h
    k_base = K_ptr + batch_idx * k_stride_b + head_idx * k_stride_h
    ks_base = K_scales_ptr + batch_idx * ks_stride_b + head_idx * ks_stride_h
    v_base = V_ptr + batch_idx * v_stride_b + head_idx * v_stride_h
    vs_base = V_scales_ptr + batch_idx * vs_stride_b + head_idx * vs_stride_h
    o_base = Out_ptr + batch_idx * o_stride_b + head_idx * o_stride_h

    # Load Q vector: (HEAD_DIM,) in FP32 for accumulation precision
    dim_offsets = tl.arange(0, HEAD_DIM)
    q = tl.load(q_base + 0 * q_stride_s + dim_offsets * q_stride_d).to(tl.float32)

    # Online softmax accumulators (FP32)
    m = tl.full([1], float("-inf"), dtype=tl.float32)  # running max
    l = tl.zeros([1], dtype=tl.float32)  # running sum of exp
    o = tl.zeros([HEAD_DIM], dtype=tl.float32)  # running weighted output

    # Loop over KV blocks
    for block_start in range(0, seq_len, BLOCK_KV):
        # Positions in this block
        block_len = tl.minimum(BLOCK_KV, seq_len - block_start)
        kv_offsets = tl.arange(0, BLOCK_KV)
        mask = kv_offsets < block_len

        # --- Load and dequantize K block ---
        # K_int8: (BLOCK_KV, HEAD_DIM)
        k_ptrs = k_base + (block_start + kv_offsets[:, None]) * k_stride_s + dim_offsets[None, :] * k_stride_d
        k_int8 = tl.load(k_ptrs, mask=mask[:, None], other=0).to(tl.float32)

        # K_scales: (BLOCK_KV, 1)
        ks_ptrs = ks_base + (block_start + kv_offsets) * ks_stride_s
        k_scales = tl.load(ks_ptrs, mask=mask, other=0.0).to(tl.float32)

        # Dequantize: k_fp = k_int8 * k_scales
        k_fp = k_int8 * k_scales[:, None]

        # --- Compute scores: Q @ K^T → (BLOCK_KV,) ---
        # q is (HEAD_DIM,), k_fp is (BLOCK_KV, HEAD_DIM)
        # scores[i] = sum_d(q[d] * k_fp[i, d]) * scaling
        scores = tl.sum(q[None, :] * k_fp, axis=1) * scaling

        # Mask out-of-bounds positions
        scores = tl.where(mask, scores, float("-inf"))

        # --- Online softmax update ---
        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(m, block_max)

        # Rescale previous accumulators
        exp_diff = tl.exp(m - new_max)
        l = l * exp_diff

        # Current block exp scores
        exp_scores = tl.exp(scores - new_max)
        exp_scores = tl.where(mask, exp_scores, 0.0)
        block_sum = tl.sum(exp_scores, axis=0)
        l = l + block_sum

        # Rescale running output
        o = o * exp_diff

        # --- Load and dequantize V block ---
        v_ptrs = v_base + (block_start + kv_offsets[:, None]) * v_stride_s + dim_offsets[None, :] * v_stride_d
        v_int8 = tl.load(v_ptrs, mask=mask[:, None], other=0).to(tl.float32)

        vs_ptrs = vs_base + (block_start + kv_offsets) * vs_stride_s
        v_scales = tl.load(vs_ptrs, mask=mask, other=0.0).to(tl.float32)

        v_fp = v_int8 * v_scales[:, None]

        # --- Accumulate weighted V: o += exp_scores @ V ---
        # exp_scores is (BLOCK_KV,), v_fp is (BLOCK_KV, HEAD_DIM)
        o = o + tl.sum(exp_scores[:, None] * v_fp, axis=0)

        m = new_max

    # Final normalization
    o = o / l

    # Store output as FP16
    tl.store(o_base + 0 * o_stride_s + dim_offsets * o_stride_d, o.to(tl.float16))


def triton_int8_attention(
    query: torch.Tensor,
    k_int8: torch.Tensor,
    k_scales: torch.Tensor,
    v_int8: torch.Tensor,
    v_scales: torch.Tensor,
    scaling: float,
    block_kv: int = 64,
) -> torch.Tensor:
    """Python wrapper for the Triton INT8 KV attention kernel.

    Args:
        query: (batch, heads, 1, head_dim) FP16
        k_int8: (batch, heads, seq_len, head_dim) INT8
        k_scales: (batch, heads, seq_len, 1) FP16
        v_int8: (batch, heads, seq_len, head_dim) INT8
        v_scales: (batch, heads, seq_len, 1) FP16
        scaling: 1/sqrt(head_dim)
        block_kv: KV block size for chunked processing (must be power of 2)

    Returns:
        output: (batch, heads, 1, head_dim) FP16
    """
    batch, heads, q_len, head_dim = query.shape
    assert q_len == 1, f"Triton kernel supports q_len=1 only, got {q_len}"
    assert batch == 1, f"Triton kernel supports batch=1 only, got {batch}"

    seq_len = k_int8.shape[2]
    assert k_int8.shape == (batch, heads, seq_len, head_dim)
    assert k_scales.shape == (batch, heads, seq_len, 1)
    assert v_int8.shape == (batch, heads, seq_len, head_dim)
    assert v_scales.shape == (batch, heads, seq_len, 1)

    # Ensure contiguous
    query = query.contiguous()
    k_int8 = k_int8.contiguous()
    k_scales = k_scales.contiguous()
    v_int8 = v_int8.contiguous()
    v_scales = v_scales.contiguous()

    output = torch.empty(batch, heads, 1, head_dim, dtype=torch.float16, device=query.device)

    # Grid: one program per (batch, head)
    grid = (batch * heads,)

    BLOCK_KV = block_kv

    _int8_kv_attention_kernel[grid](
        query,
        k_int8,
        k_scales,
        v_int8,
        v_scales,
        output,
        # Q strides
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        # K strides
        k_int8.stride(0), k_int8.stride(1), k_int8.stride(2), k_int8.stride(3),
        # K_scales strides
        k_scales.stride(0), k_scales.stride(1), k_scales.stride(2), k_scales.stride(3),
        # V strides
        v_int8.stride(0), v_int8.stride(1), v_int8.stride(2), v_int8.stride(3),
        # V_scales strides
        v_scales.stride(0), v_scales.stride(1), v_scales.stride(2), v_scales.stride(3),
        # Output strides
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        # Runtime args
        scaling=scaling,
        seq_len=seq_len,
        # Constexpr
        HEAD_DIM=head_dim,
        BLOCK_KV=BLOCK_KV,
    )

    return output
