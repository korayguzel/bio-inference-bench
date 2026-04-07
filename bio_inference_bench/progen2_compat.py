"""Compatibility patches for hugohrban/progen2-small on transformers >= 5.0.

Root cause: The custom modeling_progen.py from hugohrban/progen2-small was
written for transformers 4.x. Two breaking changes in transformers 5.x:

1. PreTrainedModel.get_head_mask() was removed from the base class.
   The custom ProGenModel.forward() calls self.get_head_mask() at line 441.
   Fix: Inject a get_head_mask method into ProGenPreTrainedModel.

2. ProGenConfig uses 'n_layer' instead of the standard 'num_hidden_layers'.
   Some transformers internals expect 'num_hidden_layers'.
   Fix: Add num_hidden_layers as a property alias on ProGenConfig.

This module patches these issues after the custom model code is loaded.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_PATCHED = False


def _get_head_mask(self, head_mask, num_hidden_layers: int, is_attention_chunked: bool = False):
    """Replacement for the removed PreTrainedModel.get_head_mask().

    Prepares the head mask for attention head pruning. In practice,
    head_mask is almost always None during inference, so this returns
    [None] * num_hidden_layers.
    """
    if head_mask is not None:
        head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
        if is_attention_chunked:
            head_mask = head_mask.unsqueeze(-1)
    else:
        head_mask = [None] * num_hidden_layers
    return head_mask


def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers: int):
    """Convert a head mask to 5D format."""
    if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.dim() == 2:
        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    return head_mask


def _is_progen_model(model) -> bool:
    """Check if a model is from the ProGen family (custom HF code)."""
    class_name = type(model).__name__.lower()
    config_type = getattr(model.config, "model_type", "").lower()
    return "progen" in class_name or "progen" in config_type


def patch_progen2_model(model) -> list[str]:
    """Apply compatibility patches to a loaded ProGen2 model instance.

    Only applies patches to ProGen-family models. Returns immediately
    with an empty list for non-ProGen models (e.g., GPT2).
    """
    if not _is_progen_model(model):
        return []

    patches_applied: list[str] = []
    model_class = type(model)
    config = model.config

    # Patch 1: Add get_head_mask if missing
    if not hasattr(model, "get_head_mask"):
        model_class.get_head_mask = _get_head_mask
        model_class._convert_head_mask_to_5d = _convert_head_mask_to_5d
        patches_applied.append(
            "Added get_head_mask() to ProGenPreTrainedModel "
            "(removed in transformers 5.x)"
        )
        logger.info("Patched: get_head_mask added to %s", model_class.__name__)

    # Also patch the inner transformer model if it's a CausalLM wrapper
    if hasattr(model, "transformer") and not hasattr(model.transformer, "get_head_mask"):
        inner_class = type(model.transformer)
        inner_class.get_head_mask = _get_head_mask
        inner_class._convert_head_mask_to_5d = _convert_head_mask_to_5d
        patches_applied.append(
            "Added get_head_mask() to inner transformer model "
            f"({inner_class.__name__})"
        )
        logger.info("Patched: get_head_mask added to %s", inner_class.__name__)

    # Patch 2: Fix scale_attn tensors stranded on meta device.
    # The custom ProGen attention creates scale_attn as a plain tensor in __init__.
    # With transformers 5.x meta-device initialization, this tensor stays on 'meta'
    # because .to(device) only moves registered parameters and buffers.
    device = next(model.parameters()).device
    fixed_scale_attn = 0
    for module in model.modules():
        if hasattr(module, "scale_attn") and isinstance(module.scale_attn, torch.Tensor):
            if module.scale_attn.device.type == "meta":
                head_dim = getattr(module, "head_dim", 64)
                module.scale_attn = torch.sqrt(
                    torch.tensor(head_dim, dtype=torch.float32)
                ).to(device=device, dtype=model.dtype)
                fixed_scale_attn += 1
    if fixed_scale_attn > 0:
        patches_applied.append(
            f"Moved scale_attn from meta to {device} in {fixed_scale_attn} attention layers"
        )
        logger.info("Patched: scale_attn moved to %s in %d layers", device, fixed_scale_attn)

    # Patch 3: Add num_hidden_layers alias if missing
    if not hasattr(config, "num_hidden_layers"):
        n_layer = getattr(config, "n_layer", None)
        if n_layer is not None:
            config.num_hidden_layers = n_layer
            patches_applied.append(
                f"Added config.num_hidden_layers = {n_layer} "
                "(alias for config.n_layer)"
            )
            logger.info("Patched: num_hidden_layers = %d on config", n_layer)

    return patches_applied
