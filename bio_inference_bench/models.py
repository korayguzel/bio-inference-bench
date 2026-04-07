"""Model and tokenizer loading with candidate fallback.

Loads HuggingFace causal LMs, extracts architecture metadata, and prepares
prompt tensors. Handles ProGen2 compatibility issues by iterating through
candidate repos until one succeeds (both model AND tokenizer must load).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from transformers.pytorch_utils import Conv1D

from bio_inference_bench.utils import DEFAULT_PROMPT_SEQUENCE, MODEL_REGISTRY

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Architecture metadata extracted from a loaded model."""

    name: str
    hf_path: str
    hf_path_attempted: list[str] = field(default_factory=list)
    hf_path_loaded: str | None = None
    param_count: int = 0
    param_count_str: str = ""
    dtype: str = "float16"
    weight_memory_mb: float = 0.0
    hidden_size: int = 0
    num_layers: int = 0
    num_attention_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    max_position_embeddings: int = 0
    vocab_size: int = 0
    architecture: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "hf_path": self.hf_path,
            "hf_path_attempted": self.hf_path_attempted,
            "hf_path_loaded": self.hf_path_loaded,
            "param_count": self.param_count,
            "param_count_str": self.param_count_str,
            "dtype": self.dtype,
            "weight_memory_mb": round(self.weight_memory_mb, 2),
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "max_position_embeddings": self.max_position_embeddings,
            "vocab_size": self.vocab_size,
            "architecture": self.architecture,
            "warnings": self.warnings,
        }


def _extract_metadata(
    model: PreTrainedModel, hf_path: str, name: str, dtype: torch.dtype
) -> ModelMetadata:
    """Extract architecture metadata from a loaded model."""
    config = model.config
    warnings: list[str] = []

    def _get(primary: str, *fallbacks: str) -> int:
        val = getattr(config, primary, None)
        if val is not None:
            return int(val)
        for fb in fallbacks:
            val = getattr(config, fb, None)
            if val is not None:
                warnings.append(f"Used fallback config field '{fb}' for '{primary}'")
                return int(val)
        warnings.append(f"Config field '{primary}' not found (tried fallbacks: {fallbacks})")
        return 0

    hidden_size = _get("hidden_size", "n_embd", "d_model")
    num_layers = _get("num_hidden_layers", "n_layer", "num_layers")
    num_attention_heads = _get("num_attention_heads", "n_head", "num_heads")
    num_kv_heads = _get("num_key_value_heads", "num_kv_heads")
    if num_kv_heads == 0:
        num_kv_heads = num_attention_heads
        if num_attention_heads > 0:
            warnings.append(
                f"num_key_value_heads not found; falling back to num_attention_heads={num_attention_heads}"
            )
    head_dim = hidden_size // num_attention_heads if num_attention_heads > 0 else 0
    max_pos = _get("max_position_embeddings", "n_positions", "n_ctx")
    vocab_size = _get("vocab_size")

    param_count = sum(p.numel() for p in model.parameters())
    weight_memory_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    arch_list = getattr(config, "architectures", None)
    architecture = arch_list[0] if arch_list else type(model).__name__

    count_m = param_count / 1e6
    param_count_str = f"{count_m:.1f}M" if count_m < 1000 else f"{count_m / 1000:.2f}B"

    return ModelMetadata(
        name=name,
        hf_path=hf_path,
        param_count=param_count,
        param_count_str=param_count_str,
        dtype=str(dtype).replace("torch.", ""),
        weight_memory_mb=weight_memory_bytes / (1024**2),
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_position_embeddings=max_pos,
        vocab_size=vocab_size,
        architecture=architecture,
        warnings=warnings,
    )


def _try_load_tokenizer(
    hf_path: str, trust_remote_code: bool
) -> tuple[PreTrainedTokenizerBase | None, str | None]:
    """Attempt tokenizer loading with fast/slow fallback. Returns (tokenizer, error)."""
    try:
        tok = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=trust_remote_code)
        return tok, None
    except Exception as e1:
        first_error = str(e1)

    try:
        tok = AutoTokenizer.from_pretrained(
            hf_path, trust_remote_code=trust_remote_code, use_fast=False
        )
        return tok, None
    except Exception as e2:
        return None, f"Fast tokenizer: {first_error} | Slow tokenizer: {e2}"


def _convert_conv1d_to_linear(model: PreTrainedModel) -> int:
    """Replace all Conv1D modules with nn.Linear (required for torchao quantization).

    GPT-2 uses Conv1D (weight shape [in, out]) instead of nn.Linear ([out, in]).
    Most quantization libraries only target nn.Linear, so conversion is needed.
    Returns the number of modules converted.
    """
    count = 0
    for name, module in model.named_children():
        if isinstance(module, Conv1D):
            linear = nn.Linear(
                module.weight.shape[0], module.weight.shape[1],
                bias=module.bias is not None,
                device=module.weight.device, dtype=module.weight.dtype,
            )
            linear.weight = nn.Parameter(module.weight.T)
            if module.bias is not None:
                linear.bias = module.bias
            setattr(model, name, linear)
            count += 1
        else:
            count += _convert_conv1d_to_linear(module)
    return count


def _set_eval_mode(model: PreTrainedModel) -> None:
    """Set model to evaluation mode with no gradients."""
    # PyTorch's .eval() sets dropout/batchnorm to inference mode
    model.train(False)
    model.requires_grad_(False)


def load_model_and_tokenizer(
    model_name: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    weight_quantization: str = "fp16",
) -> tuple[PreTrainedModel | None, PreTrainedTokenizerBase | None, ModelMetadata]:
    """Load a model and tokenizer by registry name with candidate fallback.

    Iterates through candidate HF paths until both model AND tokenizer load
    successfully. If a tokenizer fails, the entire candidate is rejected.

    Args:
        weight_quantization: "fp16" (default) or "bnb-nf4" (4-bit NormalFloat
            via bitsandbytes). When "bnb-nf4", the model is loaded with
            BitsAndBytesConfig(load_in_4bit=True) and device_map="auto".

    Returns:
        (model, tokenizer, metadata) — model/tokenizer are None if all candidates fail.
        metadata.warnings contains per-candidate failure details.
    """
    if model_name not in MODEL_REGISTRY:
        meta = ModelMetadata(name=model_name, hf_path="unknown")
        meta.warnings.append(f"Model '{model_name}' not in MODEL_REGISTRY")
        return None, None, meta

    entry = MODEL_REGISTRY[model_name]
    candidates = entry["candidates"]
    all_attempted: list[str] = []
    all_warnings: list[str] = []

    for i, candidate in enumerate(candidates):
        hf_path = candidate["hf_path"]
        trust_remote_code = candidate.get("trust_remote_code", False)
        all_attempted.append(hf_path)

        # Try loading model
        try:
            logger.info(f"Trying candidate {i + 1}/{len(candidates)}: {hf_path}")
            load_kwargs = {
                "trust_remote_code": trust_remote_code,
                "torch_dtype": dtype,
            }

            if weight_quantization == "bnb-nf4":
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_quant_type="nf4",
                )
                load_kwargs["device_map"] = "auto"
                model = AutoModelForCausalLM.from_pretrained(hf_path, **load_kwargs)
            elif weight_quantization == "bnb-int8":
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                load_kwargs["device_map"] = "auto"
                model = AutoModelForCausalLM.from_pretrained(hf_path, **load_kwargs)
            elif weight_quantization == "torchao-int8":
                import gc
                from torchao.quantization import Int8WeightOnlyConfig, quantize_
                model = AutoModelForCausalLM.from_pretrained(
                    hf_path, **load_kwargs
                ).to(device)
                converted = _convert_conv1d_to_linear(model)
                if converted > 0:
                    logger.info(f"Converted {converted} Conv1D → nn.Linear for torchao")
                quantize_(model, Int8WeightOnlyConfig())
                gc.collect()
                torch.cuda.empty_cache()
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    hf_path, **load_kwargs
                ).to(device)
            _set_eval_mode(model)
        except Exception as e:
            msg = f"Candidate {hf_path}: model load failed — {e}"
            logger.warning(msg)
            all_warnings.append(msg)
            continue

        # Try loading tokenizer
        tokenizer, tok_error = _try_load_tokenizer(hf_path, trust_remote_code)
        if tokenizer is None:
            msg = f"Candidate {hf_path}: tokenizer failed — {tok_error}"
            logger.warning(msg)
            all_warnings.append(msg)
            # Unload the model since the candidate is unusable
            del model
            torch.cuda.empty_cache()
            continue

        # Both succeeded
        metadata = _extract_metadata(model, hf_path, model_name, dtype)
        metadata.hf_path_attempted = all_attempted
        metadata.hf_path_loaded = hf_path
        metadata.warnings.extend(all_warnings)
        if i > 0:
            metadata.warnings.append(f"Loaded from fallback candidate #{i + 1}: {hf_path}")
        return model, tokenizer, metadata

    # All candidates failed
    meta = ModelMetadata(name=model_name, hf_path=candidates[0]["hf_path"])
    meta.hf_path_attempted = all_attempted
    meta.hf_path_loaded = None
    meta.warnings = all_warnings + [
        f"ALL {len(candidates)} candidates failed for '{model_name}'"
    ]
    return None, None, meta


def prepare_prompt(
    tokenizer: PreTrainedTokenizerBase,
    prompt_token_length: int = 64,
) -> torch.Tensor:
    """Encode a protein sequence and truncate/repeat to exact token length.

    Args:
        tokenizer: A successfully loaded tokenizer.
        prompt_token_length: Desired number of tokens (not amino-acid characters).

    Returns:
        input_ids tensor of shape (1, prompt_token_length) on CPU.

    Raises:
        ValueError: If tokenizer is None (candidate should have been rejected).
    """
    if tokenizer is None:
        raise ValueError(
            "Cannot prepare prompt without a tokenizer. "
            "The model candidate should have been rejected during loading."
        )

    encoded = tokenizer.encode(DEFAULT_PROMPT_SEQUENCE, add_special_tokens=False)

    # Repeat if too short
    while len(encoded) < prompt_token_length:
        encoded = encoded + encoded

    # Truncate to exact length
    encoded = encoded[:prompt_token_length]

    return torch.tensor([encoded], dtype=torch.long)
