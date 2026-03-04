"""Compatibility shims for functions missing in trl 0.9.6."""

import torch
import torch.nn.functional as F


def flush_left(attention_mask, *tensors):
    """Left-flush tensors: move non-padding tokens to the left."""
    # Sort by attention_mask descending so 1s come before 0s per row
    sorted_indices = attention_mask.argsort(dim=1, descending=True, stable=True)
    results = [attention_mask.gather(1, sorted_indices)]
    for t in tensors:
        results.append(t.gather(1, sorted_indices))
    return tuple(results)


def flush_right(attention_mask, *tensors):
    """Right-flush tensors: move non-padding tokens to the right."""
    sorted_indices = attention_mask.argsort(dim=1, descending=False, stable=True)
    results = [attention_mask.gather(1, sorted_indices)]
    for t in tensors:
        results.append(t.gather(1, sorted_indices))
    return tuple(results)


def maybe_extract_prompt(example):
    """Extract prompt from a DPO example if not already present."""
    if "prompt" in example:
        return example

    # Try to extract from chosen/rejected conversations
    for key in ("chosen", "rejected"):
        val = example.get(key)
        if isinstance(val, list) and len(val) >= 1:
            # Conversational format: list of dicts with "role" and "content"
            # Prompt = all messages except the last (assistant) one
            prompt = val[:-1]
            example["prompt"] = prompt
            example["chosen"] = example["chosen"][-1:]
            example["rejected"] = example["rejected"][-1:]
            return example

    return example


def maybe_apply_chat_template(example, tokenizer, tools=None):
    """Apply chat template to conversational examples."""
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return example

    for key in ("prompt", "chosen", "rejected"):
        val = example.get(key)
        if isinstance(val, list):
            example[key] = tokenizer.apply_chat_template(val, tokenize=False, add_generation_prompt=(key == "prompt"))

    return example


def selective_log_softmax(logits, index):
    """Compute log softmax only for selected indices (memory-efficient)."""
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=index.unsqueeze(-1)).squeeze(-1)


def is_conversational(example):
    """Check if a dataset example uses conversational format."""
    for key in ("prompt", "chosen", "rejected"):
        val = example.get(key)
        if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
            return True
    return False
