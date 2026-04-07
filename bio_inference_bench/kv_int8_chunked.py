"""Chunked-dequantize INT8 KV cache prototype (v2).

Unlike v1 (dequantize-on-read), this prototype never materializes the full
FP16 KV cache. Instead, it:
1. Stores K/V in INT8 with per-token scales (same as v1)
2. During attention, dequantizes K/V in chunks of `chunk_size` positions
3. Accumulates attention output using the online softmax algorithm
4. Peak memory = INT8 cache + O(chunk_size * heads * head_dim) FP16 per chunk

The online softmax (Milakov & Gimelshein 2018) correctly combines partial
attention scores across chunks without requiring the full softmax denominator
upfront.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from bio_inference_bench.kv_int8_cache import dequantize_from_int8, quantize_to_int8

# Default chunk size for KV dequantization during attention.
# Smaller = less peak memory, more overhead. 64 is a reasonable balance.
DEFAULT_CHUNK_SIZE = 64


class ChunkedInt8KVCache(DynamicCache):
    """INT8 KV cache with chunked dequantization for attention.

    The `update()` method stores INT8 data and returns lightweight handles
    instead of full FP16 tensors. A separate `chunked_attention()` method
    performs attention by dequantizing in chunks.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        protected_layers: set[int] | None = None,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.protected_layers = protected_layers or set()
        self._key_int8: list[torch.Tensor] = []
        self._key_scales: list[torch.Tensor] = []
        self._value_int8: list[torch.Tensor] = []
        self._value_scales: list[torch.Tensor] = []
        # FP16 cache for protected layers
        self._key_fp16: list[torch.Tensor | None] = []
        self._value_fp16: list[torch.Tensor | None] = []

    def is_protected(self, layer_idx: int) -> bool:
        return layer_idx in self.protected_layers

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Store new KV. Protected layers keep FP16; others use INT8."""

        # Ensure lists are long enough
        while len(self._key_int8) <= layer_idx:
            self._key_int8.append(None)
            self._key_scales.append(None)
            self._value_int8.append(None)
            self._value_scales.append(None)
            self._key_fp16.append(None)
            self._value_fp16.append(None)

        if self.is_protected(layer_idx):
            # Store in FP16 directly
            if self._key_fp16[layer_idx] is None:
                self._key_fp16[layer_idx] = key_states
                self._value_fp16[layer_idx] = value_states
            else:
                self._key_fp16[layer_idx] = torch.cat(
                    [self._key_fp16[layer_idx], key_states], dim=2
                )
                self._value_fp16[layer_idx] = torch.cat(
                    [self._value_fp16[layer_idx], value_states], dim=2
                )
            full_key = self._key_fp16[layer_idx]
            full_value = self._value_fp16[layer_idx]
        else:
            # Store in INT8
            k_q, k_s = quantize_to_int8(key_states)
            v_q, v_s = quantize_to_int8(value_states)

            if self._key_int8[layer_idx] is None:
                self._key_int8[layer_idx] = k_q
                self._key_scales[layer_idx] = k_s
                self._value_int8[layer_idx] = v_q
                self._value_scales[layer_idx] = v_s
            else:
                self._key_int8[layer_idx] = torch.cat(
                    [self._key_int8[layer_idx], k_q], dim=2
                )
                self._key_scales[layer_idx] = torch.cat(
                    [self._key_scales[layer_idx], k_s], dim=2
                )
                self._value_int8[layer_idx] = torch.cat(
                    [self._value_int8[layer_idx], v_q], dim=2
                )
                self._value_scales[layer_idx] = torch.cat(
                    [self._value_scales[layer_idx], v_s], dim=2
                )
            # Return only new slice dequantized
            full_key = dequantize_from_int8(k_q, k_s)
            full_value = dequantize_from_int8(v_q, v_s)

        # Keep parent layers in sync for seq_length tracking
        if layer_idx >= len(self.layers):
            super().update(key_states, value_states, layer_idx, *args, **kwargs)
            self.layers[layer_idx].keys = full_key
            self.layers[layer_idx].values = full_value
        else:
            self.layers[layer_idx].keys = full_key
            self.layers[layer_idx].values = full_value

        # For protected layers, return full FP16 cache (needed for attention).
        # For INT8 layers, return only the new slice (chunked attention handles full cache).
        if self.is_protected(layer_idx):
            return self._key_fp16[layer_idx], self._value_fp16[layer_idx]
        else:
            return full_key, full_value

    def chunked_attention(
        self,
        query: torch.Tensor,
        layer_idx: int,
        scaling: float,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute attention over INT8 KV cache in chunks.

        Uses online softmax to combine partial attention across KV chunks
        without materializing the full FP16 cache.

        Args:
            query: (batch, heads, q_len, head_dim) — typically q_len=1 for decode
            layer_idx: which layer's KV to attend over
            scaling: attention scaling factor (1/sqrt(head_dim))
            causal_mask: optional attention mask

        Returns:
            attention output: (batch, heads, q_len, head_dim)
        """
        k_int8 = self._key_int8[layer_idx]
        k_scales = self._key_scales[layer_idx]
        v_int8 = self._value_int8[layer_idx]
        v_scales = self._value_scales[layer_idx]

        seq_len = k_int8.shape[2]
        batch, heads, q_len, head_dim = query.shape
        chunk = self.chunk_size

        # Online softmax accumulators
        # M = running max of attention logits (for numerical stability)
        # L = running sum of exp(logits - M)
        # O = running weighted sum of values
        device = query.device
        dtype = query.dtype
        M = torch.full((batch, heads, q_len, 1), float("-inf"), device=device, dtype=dtype)
        L = torch.zeros((batch, heads, q_len, 1), device=device, dtype=dtype)
        O = torch.zeros((batch, heads, q_len, head_dim), device=device, dtype=dtype)

        for start in range(0, seq_len, chunk):
            end = min(start + chunk, seq_len)

            # Dequantize this chunk only
            k_chunk = dequantize_from_int8(
                k_int8[:, :, start:end, :], k_scales[:, :, start:end, :]
            )
            v_chunk = dequantize_from_int8(
                v_int8[:, :, start:end, :], v_scales[:, :, start:end, :]
            )

            # Compute attention scores for this chunk: (B, H, Q, chunk_len)
            scores = torch.matmul(query, k_chunk.transpose(-2, -1)) * scaling

            # Apply causal mask if provided
            if causal_mask is not None:
                chunk_mask = causal_mask[:, :, :, start:end]
                scores = scores + chunk_mask

            # Online softmax update
            chunk_max = scores.amax(dim=-1, keepdim=True)  # (B, H, Q, 1)
            new_max = torch.maximum(M, chunk_max)

            # Rescale previous accumulator
            exp_diff_old = torch.exp(M - new_max)
            # Current chunk's exp
            exp_scores = torch.exp(scores - new_max)

            # Update running sum
            chunk_sum = exp_scores.sum(dim=-1, keepdim=True)  # (B, H, Q, 1)
            L = L * exp_diff_old + chunk_sum

            # Update running weighted value sum
            O = O * exp_diff_old + torch.matmul(exp_scores, v_chunk)

            M = new_max

            # Free chunk tensors
            del k_chunk, v_chunk, scores, exp_scores

        # Final normalization
        output = O / L
        return output

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx < len(self._key_fp16) and self._key_fp16[layer_idx] is not None:
            return self._key_fp16[layer_idx].shape[2]
        if layer_idx < len(self._key_int8) and self._key_int8[layer_idx] is not None:
            return self._key_int8[layer_idx].shape[2]
        return 0

    def int8_memory_bytes(self) -> int:
        """Memory used by INT8 quantized layers (not protected layers)."""
        total = 0
        for i in range(len(self._key_int8)):
            if self._key_int8[i] is None:
                continue
            total += self._key_int8[i].nelement() * self._key_int8[i].element_size()
            total += self._key_scales[i].nelement() * self._key_scales[i].element_size()
            total += self._value_int8[i].nelement() * self._value_int8[i].element_size()
            total += self._value_scales[i].nelement() * self._value_scales[i].element_size()
        return total

    def fp16_protected_bytes(self) -> int:
        """Memory used by FP16 protected layers."""
        total = 0
        for i in range(len(self._key_fp16)):
            if self._key_fp16[i] is None:
                continue
            total += self._key_fp16[i].nelement() * self._key_fp16[i].element_size()
            total += self._value_fp16[i].nelement() * self._value_fp16[i].element_size()
        return total

    def total_cache_bytes(self) -> int:
        """Total cache memory: INT8 layers + FP16 protected layers."""
        return self.int8_memory_bytes() + self.fp16_protected_bytes()

    def fp16_equivalent_bytes(self) -> int:
        """What the same cache would cost if ALL layers were FP16."""
        total = 0
        for i in range(len(self._key_int8)):
            if self._key_int8[i] is not None:
                total += self._key_int8[i].nelement() * 2  # K as fp16
                total += self._value_int8[i].nelement() * 2  # V as fp16
        for i in range(len(self._key_fp16)):
            if self._key_fp16[i] is not None:
                total += self._key_fp16[i].nelement() * self._key_fp16[i].element_size()
                total += self._value_fp16[i].nelement() * self._value_fp16[i].element_size()
        return total

    def compression_summary(self) -> dict:
        """Summary of compression stats."""
        n_int8 = sum(1 for x in self._key_int8 if x is not None)
        n_fp16 = sum(1 for x in self._key_fp16 if x is not None)
        return {
            "int8_layers": n_int8,
            "fp16_protected_layers": n_fp16,
            "int8_storage_mb": round(self.int8_memory_bytes() / (1024**2), 2),
            "fp16_protected_mb": round(self.fp16_protected_bytes() / (1024**2), 2),
            "total_cache_mb": round(self.total_cache_bytes() / (1024**2), 2),
            "fp16_equivalent_mb": round(self.fp16_equivalent_bytes() / (1024**2), 2),
        }


def run_chunked_decode_step(
    model,
    next_token: torch.Tensor,
    cache: ChunkedInt8KVCache,
    use_triton: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one decode step using INT8 KV attention.

    This bypasses the model's normal forward path for attention and uses
    either the Python chunked attention (v2) or the fused Triton kernel (v5).
    We still use the model's QKV projections, output projections, FFN, and
    layer norms.

    Args:
        use_triton: If True, use the fused Triton INT8 attention kernel
            instead of the Python chunked attention loop.

    Returns logits tensor.
    """
    device = next(model.parameters()).device
    config = model.config

    # Get the transformer (inner model for CausalLM wrappers)
    if hasattr(model, "transformer"):
        transformer = model.transformer
    else:
        transformer = model

    # Embed the token
    hidden = transformer.wte(next_token)
    if hasattr(transformer, "wpe"):
        # Position = number of tokens already in cache (0-indexed, so the
        # next token after N cached tokens is at position N).
        past_seen = cache.get_seq_length()
        position_ids = torch.tensor([[past_seen]], device=device)
        hidden = hidden + transformer.wpe(position_ids)

    if hasattr(transformer, "drop"):
        hidden = transformer.drop(hidden)

    # Process through transformer layers
    for layer_idx, block in enumerate(transformer.h):
        residual = hidden
        hidden = block.ln_1(hidden)

        # QKV projection
        qkv = block.attn.c_attn(hidden)
        query, key, value = qkv.split(config.n_embd, dim=-1)

        # Reshape to (batch, heads, seq=1, head_dim)
        n_heads = config.n_head
        head_dim = config.n_embd // n_heads
        batch = query.shape[0]

        query = query.view(batch, 1, n_heads, head_dim).transpose(1, 2)
        key = key.view(batch, 1, n_heads, head_dim).transpose(1, 2)
        value = value.view(batch, 1, n_heads, head_dim).transpose(1, 2)

        # Update cache with new K/V
        full_key_ret, full_value_ret = cache.update(key, value, layer_idx)

        # Attention dispatch
        scaling = 1.0 / math.sqrt(head_dim)
        if cache.is_protected(layer_idx):
            # Protected layer: full FP16 cache returned by update(), use SDPA
            attn_output = F.scaled_dot_product_attention(
                query, full_key_ret, full_value_ret,
                is_causal=False,  # causal mask not needed for single-token decode
                scale=scaling,
            )
        elif use_triton:
            # Fused Triton kernel: reads INT8 KV directly
            from bio_inference_bench.triton_int8_attention import triton_int8_attention
            attn_output = triton_int8_attention(
                query,
                cache._key_int8[layer_idx],
                cache._key_scales[layer_idx],
                cache._value_int8[layer_idx],
                cache._value_scales[layer_idx],
                scaling,
            )
        else:
            # Python chunked dequantize attention (v2 reference path)
            attn_output = cache.chunked_attention(query, layer_idx, scaling)

        # Output projection
        attn_output = attn_output.transpose(1, 2).reshape(batch, 1, config.n_embd)
        attn_output = block.attn.c_proj(attn_output)
        attn_output = block.attn.resid_dropout(attn_output)

        hidden = residual + attn_output

        # FFN
        residual = hidden
        hidden = block.ln_2(hidden)
        hidden = block.mlp(hidden)
        hidden = residual + hidden

    hidden = transformer.ln_f(hidden)

    # LM head
    if hasattr(model, "lm_head"):
        logits = model.lm_head(hidden)
    else:
        logits = hidden @ transformer.wte.weight.T

    return logits
