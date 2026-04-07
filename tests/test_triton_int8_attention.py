"""Correctness tests for the Triton INT8 KV attention kernel (Phase A).

Tests compare the Triton kernel output against two reference implementations:
1. Standard FP16 attention on dequantized KV (ground truth)
2. v2 ChunkedInt8KVCache.chunked_attention() (the Python reference path)

The kernel should match both references within FP16 precision tolerances.
"""

from __future__ import annotations

import math

import pytest
import torch

from bio_inference_bench.kv_int8_cache import dequantize_from_int8, quantize_to_int8
from bio_inference_bench.kv_int8_chunked import ChunkedInt8KVCache
from bio_inference_bench.triton_int8_attention import triton_int8_attention

# ProtGPT2 parameters
HEADS = 20
HEAD_DIM = 64
SCALING = 1.0 / math.sqrt(HEAD_DIM)

# Test seq_len values: edge cases around BLOCK_KV=64 boundaries
SEQ_LENS = [1, 63, 64, 65, 128, 256, 512, 768, 1024]

DEVICE = "cuda"


def _make_random_int8_kv(batch, heads, seq_len, head_dim, device=DEVICE):
    """Generate random FP16 KV, quantize to INT8, return (int8, scales) pairs."""
    K_fp16 = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
    V_fp16 = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
    K_int8, K_scales = quantize_to_int8(K_fp16)
    V_int8, V_scales = quantize_to_int8(V_fp16)
    return K_int8, K_scales, V_int8, V_scales


def _reference_fp16_attention(query, k_int8, k_scales, v_int8, v_scales, scaling):
    """Standard softmax attention on dequantized FP16 KV — ground truth."""
    K = dequantize_from_int8(k_int8, k_scales).float()
    V = dequantize_from_int8(v_int8, v_scales).float()
    Q = query.float()
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scaling
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, V).to(torch.float16)


def _reference_v2_chunked(query, k_int8, k_scales, v_int8, v_scales, scaling,
                           chunk_size=64):
    """v2 ChunkedInt8KVCache.chunked_attention() reference."""
    cache = ChunkedInt8KVCache(chunk_size=chunk_size)
    layer_idx = 0
    # Populate cache internal state directly
    while len(cache._key_int8) <= layer_idx:
        cache._key_int8.append(None)
        cache._key_scales.append(None)
        cache._value_int8.append(None)
        cache._value_scales.append(None)
        cache._key_fp16.append(None)
        cache._value_fp16.append(None)
    cache._key_int8[layer_idx] = k_int8
    cache._key_scales[layer_idx] = k_scales
    cache._value_int8[layer_idx] = v_int8
    cache._value_scales[layer_idx] = v_scales
    return cache.chunked_attention(query, layer_idx, scaling)


# ---------------------------------------------------------------------------
# Test: Triton vs standard FP16 attention (ground truth)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seq_len", SEQ_LENS)
def test_triton_vs_fp16_reference(seq_len):
    """Triton output matches standard FP16 attention on dequantized KV."""
    torch.manual_seed(42 + seq_len)
    Q = torch.randn(1, HEADS, 1, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    K_int8, K_scales, V_int8, V_scales = _make_random_int8_kv(1, HEADS, seq_len, HEAD_DIM)

    ref = _reference_fp16_attention(Q, K_int8, K_scales, V_int8, V_scales, SCALING)
    tri = triton_int8_attention(Q, K_int8, K_scales, V_int8, V_scales, SCALING)

    max_diff = (ref.float() - tri.float()).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        ref.float().reshape(-1), tri.float().reshape(-1), dim=0
    ).item()

    assert max_diff < 0.01, f"seq_len={seq_len}: max_diff={max_diff}"
    assert cos_sim > 0.9999, f"seq_len={seq_len}: cos_sim={cos_sim}"


# ---------------------------------------------------------------------------
# Test: Triton vs v2 chunked attention (the Python reference path)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seq_len", SEQ_LENS)
def test_triton_vs_v2_chunked(seq_len):
    """Triton output matches v2 ChunkedInt8KVCache.chunked_attention()."""
    torch.manual_seed(42 + seq_len)
    Q = torch.randn(1, HEADS, 1, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    K_int8, K_scales, V_int8, V_scales = _make_random_int8_kv(1, HEADS, seq_len, HEAD_DIM)

    v2_ref = _reference_v2_chunked(Q, K_int8, K_scales, V_int8, V_scales, SCALING)
    tri = triton_int8_attention(Q, K_int8, K_scales, V_int8, V_scales, SCALING)

    max_diff = (v2_ref.float() - tri.float()).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        v2_ref.float().reshape(-1), tri.float().reshape(-1), dim=0
    ).item()

    assert max_diff < 0.01, f"seq_len={seq_len}: max_diff={max_diff}"
    assert cos_sim > 0.9999, f"seq_len={seq_len}: cos_sim={cos_sim}"


# ---------------------------------------------------------------------------
# Test: Determinism — same inputs produce same outputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seq_len", [64, 256, 512])
def test_determinism(seq_len):
    """Repeated calls with same input produce identical output."""
    torch.manual_seed(123)
    Q = torch.randn(1, HEADS, 1, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    K_int8, K_scales, V_int8, V_scales = _make_random_int8_kv(1, HEADS, seq_len, HEAD_DIM)

    out1 = triton_int8_attention(Q, K_int8, K_scales, V_int8, V_scales, SCALING)
    out2 = triton_int8_attention(Q, K_int8, K_scales, V_int8, V_scales, SCALING)

    assert torch.equal(out1, out2), f"Non-deterministic output at seq_len={seq_len}"


# ---------------------------------------------------------------------------
# Test: Output shape and dtype
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seq_len", [1, 64, 256])
def test_output_shape_dtype(seq_len):
    """Output has correct shape and dtype."""
    Q = torch.randn(1, HEADS, 1, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    K_int8, K_scales, V_int8, V_scales = _make_random_int8_kv(1, HEADS, seq_len, HEAD_DIM)

    out = triton_int8_attention(Q, K_int8, K_scales, V_int8, V_scales, SCALING)

    assert out.shape == (1, HEADS, 1, HEAD_DIM)
    assert out.dtype == torch.float16
    assert out.device.type == "cuda"


# ---------------------------------------------------------------------------
# Test: No NaN/Inf in output
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seq_len", SEQ_LENS)
def test_no_nan_inf(seq_len):
    """Output contains no NaN or Inf values."""
    torch.manual_seed(99 + seq_len)
    Q = torch.randn(1, HEADS, 1, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    K_int8, K_scales, V_int8, V_scales = _make_random_int8_kv(1, HEADS, seq_len, HEAD_DIM)

    out = triton_int8_attention(Q, K_int8, K_scales, V_int8, V_scales, SCALING)

    assert not torch.isnan(out).any(), f"NaN in output at seq_len={seq_len}"
    assert not torch.isinf(out).any(), f"Inf in output at seq_len={seq_len}"


# ---------------------------------------------------------------------------
# Test: Uniform KV (all same vector) — output should be that vector
# ---------------------------------------------------------------------------

def test_uniform_values():
    """When all V vectors are identical, attention output should equal that vector."""
    seq_len = 128
    Q = torch.randn(1, HEADS, 1, HEAD_DIM, device=DEVICE, dtype=torch.float16)

    # All K vectors identical (attention weights don't matter for V output)
    K_single = torch.randn(1, HEADS, 1, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    K_fp16 = K_single.expand(1, HEADS, seq_len, HEAD_DIM).contiguous()
    V_single = torch.randn(1, HEADS, 1, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    V_fp16 = V_single.expand(1, HEADS, seq_len, HEAD_DIM).contiguous()

    K_int8, K_scales = quantize_to_int8(K_fp16)
    V_int8, V_scales = quantize_to_int8(V_fp16)

    out = triton_int8_attention(Q, K_int8, K_scales, V_int8, V_scales, SCALING)

    # The output should be close to the dequantized V_single
    V_deq = dequantize_from_int8(V_int8[:, :, :1, :], V_scales[:, :, :1, :])
    max_diff = (out.float() - V_deq.float()).abs().max().item()
    assert max_diff < 0.01, f"Uniform V test: max_diff={max_diff}"


# ---------------------------------------------------------------------------
# Test: Different number of heads (robustness)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("heads", [1, 4, 12, 20])
def test_different_head_counts(heads):
    """Kernel works with different number of heads."""
    seq_len = 128
    Q = torch.randn(1, heads, 1, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    K_int8, K_scales, V_int8, V_scales = _make_random_int8_kv(1, heads, seq_len, HEAD_DIM)

    ref = _reference_fp16_attention(Q, K_int8, K_scales, V_int8, V_scales, SCALING)
    tri = triton_int8_attention(Q, K_int8, K_scales, V_int8, V_scales, SCALING)

    max_diff = (ref.float() - tri.float()).abs().max().item()
    assert max_diff < 0.01, f"heads={heads}: max_diff={max_diff}"


# ---------------------------------------------------------------------------
# Aggregate: collect max error across all seq_lens for Phase A note
# ---------------------------------------------------------------------------

def test_aggregate_error_report(capsys):
    """Print aggregate error stats for Phase A completion note."""
    max_diffs_fp16 = []
    max_diffs_v2 = []
    cosines_fp16 = []
    cosines_v2 = []

    for seq_len in SEQ_LENS:
        torch.manual_seed(42 + seq_len)
        Q = torch.randn(1, HEADS, 1, HEAD_DIM, device=DEVICE, dtype=torch.float16)
        K_int8, K_scales, V_int8, V_scales = _make_random_int8_kv(1, HEADS, seq_len, HEAD_DIM)

        ref_fp16 = _reference_fp16_attention(Q, K_int8, K_scales, V_int8, V_scales, SCALING)
        ref_v2 = _reference_v2_chunked(Q, K_int8, K_scales, V_int8, V_scales, SCALING)
        tri = triton_int8_attention(Q, K_int8, K_scales, V_int8, V_scales, SCALING)

        d_fp16 = (ref_fp16.float() - tri.float()).abs().max().item()
        d_v2 = (ref_v2.float() - tri.float()).abs().max().item()
        c_fp16 = torch.nn.functional.cosine_similarity(
            ref_fp16.float().reshape(-1), tri.float().reshape(-1), dim=0
        ).item()
        c_v2 = torch.nn.functional.cosine_similarity(
            ref_v2.float().reshape(-1), tri.float().reshape(-1), dim=0
        ).item()

        max_diffs_fp16.append(d_fp16)
        max_diffs_v2.append(d_v2)
        cosines_fp16.append(c_fp16)
        cosines_v2.append(c_v2)

    with capsys.disabled():
        print("\n" + "=" * 60)
        print("  Phase A: Aggregate Error Report")
        print("=" * 60)
        print(f"\n  vs FP16 reference:")
        print(f"    Max abs diff (worst): {max(max_diffs_fp16):.6f}")
        print(f"    Max abs diff (best):  {min(max_diffs_fp16):.6f}")
        print(f"    Min cosine sim:       {min(cosines_fp16):.8f}")
        print(f"\n  vs v2 chunked reference:")
        print(f"    Max abs diff (worst): {max(max_diffs_v2):.6f}")
        print(f"    Max abs diff (best):  {min(max_diffs_v2):.6f}")
        print(f"    Min cosine sim:       {min(cosines_v2):.8f}")
        print(f"\n  Per seq_len detail:")
        for i, sl in enumerate(SEQ_LENS):
            print(f"    seq_len={sl:>4}: fp16_diff={max_diffs_fp16[i]:.6f} "
                  f"v2_diff={max_diffs_v2[i]:.6f} "
                  f"fp16_cos={cosines_fp16[i]:.8f} "
                  f"v2_cos={cosines_v2[i]:.8f}")
        print("=" * 60)
