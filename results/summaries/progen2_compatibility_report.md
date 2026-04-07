# ProGen2-small Compatibility Report

**Date:** 2026-04-06
**Transformers version:** 5.5.0
**PyTorch version:** 2.11.0+cu130

## Summary

ProGen2-small from `hugohrban/progen2-small` is now **runnable** with a local
compatibility patch (`bio_inference_bench/progen2_compat.py`). No version pins
or package downgrades were needed.

## Root Cause Analysis

Three distinct issues were identified, all caused by the custom model code
(`modeling_progen.py`, `configuration_progen.py`) being written for
transformers 4.x and loaded into transformers 5.5.0.

### Issue 1: `get_head_mask()` removed from `PreTrainedModel`

- **Error:** `'ProGenModel' object has no attribute 'get_head_mask'`
- **Location:** `modeling_progen.py` line 441
- **Cause:** `PreTrainedModel.get_head_mask()` was a utility method in
  transformers 4.x for preparing attention head masks during pruning. It was
  removed in transformers 5.x. The custom `ProGenModel.forward()` calls
  `self.get_head_mask(head_mask, self.config.n_layer)` which inherits from
  `PreTrainedModel` — but the method no longer exists.
- **Fix:** Inject a replacement `get_head_mask()` into `ProGenPreTrainedModel`
  and the inner `ProGenModel`. In practice, `head_mask` is always `None` during
  inference, so the method returns `[None] * num_layers`.

### Issue 2: `scale_attn` tensor stranded on meta device

- **Error:** `Tensor on device meta is not on the expected device cuda:0!`
- **Location:** `modeling_progen.py` line 146 (`attn_weights / self.scale_attn`)
- **Cause:** The custom attention layer creates `self.scale_attn = torch.sqrt(
  torch.tensor(head_dim))` as a plain tensor (not a registered buffer or
  parameter) during `__init__`. Transformers 5.x uses meta-device initialization
  for memory efficiency — the model is first constructed on the `meta` device,
  then weights are loaded onto the target device. `.to(device)` only moves
  registered parameters and buffers. Plain tensors created in `__init__` remain
  on `meta`.
- **Fix:** After model load, iterate through all modules and move any
  `scale_attn` tensors from `meta` to the model's device. Applied to all 12
  attention layers.

### Issue 3: `num_hidden_layers` missing from `ProGenConfig`

- **Error:** `'ProGenConfig' object has no attribute 'num_hidden_layers'`
- **Location:** Internal transformers code during `model.generate()`
- **Cause:** `ProGenConfig` uses `n_layer` (GPT-2 convention) instead of the
  standard `num_hidden_layers` expected by transformers internals.
- **Fix:** Add `config.num_hidden_layers = config.n_layer` as an alias.

## Candidates Attempted

| Candidate | Load | Tokenizer | Generation | Status |
|-----------|------|-----------|------------|--------|
| `hugohrban/progen2-small` | OK | OK | OK (with patch) | **Working** |
| `multimolecule/progen2-small` | Failed | N/A | N/A | Unregistered `progen2` model type |

The `multimolecule/progen2-small` checkpoint uses a registered model type
`progen2` that does not exist in transformers 5.5.0. It does not include custom
model code (`trust_remote_code` has nothing to load), so it cannot work without
either transformers adding native `progen2` support or converting the checkpoint
to use custom code.

## What Was Tried

1. **Local patch (successful):** Created `progen2_compat.py` with three targeted
   fixes applied at runtime after model load. No changes to the HuggingFace
   cache or transformers source. Patch is applied automatically in all CLI
   scripts.

2. **Version pinning (not attempted):** Not needed since the local patch works.
   Downgrading transformers would risk breaking other models (ProtGPT2 loads
   cleanly on 5.5.0).

3. **Alternative candidates:** `multimolecule/progen2-small` was tried but
   failed at model load (unregistered architecture type). No other public
   ProGen2-small checkpoints were found with standard HuggingFace model format.

## Smoke Benchmark Results (post-patch)

Both models now complete the smoke benchmark successfully:

| Model | Params | Weight MB | Prefill ms | Decode tok/s | Theo KV MB | Peak Alloc MB |
|-------|--------|-----------|------------|--------------|------------|---------------|
| ProtGPT2 | 774M | 1476 | 154 | 117 | 33.75 | 1520 |
| ProGen2-small | 151M | 288 | 88 | 156 | 9.00 | 337 |

## Recommendation

**Option 2: Fixed with a local patch.**

The patch is minimal (3 targeted fixes, ~50 lines), documented, and does not
modify any external files. It is applied in all CLI scripts (`run_smoke.py`,
`run_grid.py`, `benchmark_generation.py`, `inspect_model.py`) and is safe for
both models (the `get_head_mask` patch is a no-op on models that already have it).

ProGen2-small should be **included in the next benchmark round**.
