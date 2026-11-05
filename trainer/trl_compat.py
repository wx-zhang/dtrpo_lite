"""Preference-data formatting and tensor helpers for the pinned TRL version."""

from itertools import takewhile
import torch
import torch.nn.functional as F
from typing import Any, Literal, Optional, Sequence, Union


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


def extract_prompt(example: dict[str, Sequence]) -> dict[str, Sequence]:
    """Extract the shared prefix from chosen/rejected preference completions."""
    idx = 0
    for idx in range(min(len(example["chosen"]), len(example["rejected"]))):
        if example["chosen"][idx] != example["rejected"][idx]:
            if idx > 0 and example["chosen"][idx - 1] == " ":
                idx -= 1
            break
    else:
        idx = min(len(example["chosen"]), len(example["rejected"]))

    return {
        "prompt": example["chosen"][:idx],
        "chosen": example["chosen"][idx:],
        "rejected": example["rejected"][idx:],
    }


def maybe_extract_prompt(example):
    """Extract implicit DPO prompts using the same semantics as TRL.

    Some preference datasets include a plain string ``prompt`` column while
    ``chosen``/``rejected`` are full conversations. In that case the string
    prompt is redundant and the completions must be split so loss is applied
    only to the assistant answer, not to the system/user prompt tokens.
    """
    if "chosen" not in example or "rejected" not in example:
        return example

    if "prompt" in example:
        chosen_conv = is_conversational({"chosen": example["chosen"]})
        prompt_conv = is_conversational({"prompt": example["prompt"]})
        if (chosen_conv and prompt_conv) or (not chosen_conv and not prompt_conv):
            return example

    extracted = extract_prompt(
        {"chosen": example["chosen"], "rejected": example["rejected"]}
    )
    return {**example, **extracted}


def _chat_template_kwargs(example: dict[str, Any]) -> dict[str, Any]:
    kwargs = example.get("chat_template_kwargs", {})
    return kwargs if isinstance(kwargs, dict) else {}


def apply_chat_template(example, tokenizer, tools=None, **template_kwargs):
    """Apply chat templates while keeping completions assistant-only."""
    output = dict(example)
    kwargs = {**_chat_template_kwargs(example), **template_kwargs}
    output["_is_chat_template_applied"] = True

    if "messages" in example:
        output["text"] = tokenizer.apply_chat_template(
            example["messages"],
            tools=tools,
            tokenize=False,
            **kwargs,
        )

    if "prompt" in example:
        last_role = example["prompt"][-1]["role"]
        if last_role in ["user", "tool"]:
            add_generation_prompt = True
            continue_final_message = False
        elif last_role == "assistant":
            add_generation_prompt = False
            continue_final_message = True
        else:
            raise ValueError(f"Invalid role in the last prompt message: {last_role}")

        prompt = tokenizer.apply_chat_template(
            example["prompt"],
            tools=tools,
            continue_final_message=continue_final_message,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

        if "chosen" in example:
            prompt_chosen = tokenizer.apply_chat_template(
                example["prompt"] + example["chosen"],
                tools=tools,
                tokenize=False,
                **kwargs,
            )
            prompt = "".join(
                x
                for x, _ in takewhile(
                    lambda pair: pair[0] == pair[1],
                    zip(prompt, prompt_chosen, strict=False),
                )
            )
            output["chosen"] = prompt_chosen[len(prompt) :]

        if "rejected" in example:
            prompt_rejected = tokenizer.apply_chat_template(
                example["prompt"] + example["rejected"],
                tools=tools,
                tokenize=False,
                **kwargs,
            )
            prompt = "".join(
                x
                for x, _ in takewhile(
                    lambda pair: pair[0] == pair[1],
                    zip(prompt, prompt_rejected, strict=False),
                )
            )
            output["rejected"] = prompt_rejected[len(prompt) :]

        if "completion" in example:
            prompt_completion = tokenizer.apply_chat_template(
                example["prompt"] + example["completion"],
                tools=tools,
                tokenize=False,
                **kwargs,
            )
            prompt = "".join(
                x
                for x, _ in takewhile(
                    lambda pair: pair[0] == pair[1],
                    zip(prompt, prompt_completion, strict=False),
                )
            )
            output["completion"] = prompt_completion[len(prompt) :]

        output["prompt"] = prompt
    else:
        if "chosen" in example:
            output["chosen"] = tokenizer.apply_chat_template(
                example["chosen"],
                tools=tools,
                tokenize=False,
                **kwargs,
            )
        if "rejected" in example:
            output["rejected"] = tokenizer.apply_chat_template(
                example["rejected"],
                tools=tools,
                tokenize=False,
                **kwargs,
            )

    return output


def maybe_apply_chat_template(example, tokenizer, tools=None):
    """Apply chat template to conversational examples."""
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return example
    if is_conversational(example):
        return apply_chat_template(example, tokenizer, tools)
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

def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
            ],
            dim=dim,
        )
