"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""



def parse_nemotron_dialog(example, **kwargs):
    messages = []
    valid = True
    for turn in example["dialog"]:
        role = "user" if turn["source"] == "user" else "assistant"
        content = turn["body"].strip()
        messages.append({"role": role, "content": content})
        if role == "assistant":
            if len(content.split()) < 5:
                valid = False
    return {"messages": messages, "valid": valid}


def tulu_parsing_function(example, **kwargs):
    dialog = example["dialog"]

    for turn in reversed(dialog):
        if turn.get("source") == "user":
            prompt = turn.get("body", "")
            break
    # chosen/rejected
    choices = example["choices"]["choices"]
    chosen = choices.get("chosen_0", [{}])[0].get("body", "")
    rejected = choices.get("rejected_0", [{}])[0].get("body", "")
    example = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
    if len(chosen.split()) > 5 and len(rejected.split()) > 5:
        example["valid"] = True
    else:
        example["valid"] = False
    return example


def parse_dpo_sample(example, dataset_id: str, **kwargs):
    """
    Parse different dataset formats into a unified DPO format.

    Args:
        example (dict): One sample from the dataset.
        dataset_id (str): Identifier of the dataset.

    Returns:
        dict: A dict with fields `prompt`, `chosen`, and `rejected`.
    """

    # format: instruction, chosen_response, rejected_response
    return {
        "prompt": example["instruction"],
        "chosen": [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["chosen_response"]},
        ],
        "rejected": [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["rejected_response"]},
        ],
        "valid": True,
    }


def parse_dpo_code_sample(example, **kwargs):
    """
    Parse a DPO code sample with 'input', 'accepted', and 'rejected' fields into a unified format.

    Args:
        example (dict): One sample from the dataset.

    Returns:
        dict: A dict with fields `prompt`, `chosen`, `rejected`, and `valid`.
    """
    prompt = example.get("input", "")
    chosen = example.get("accepted", "")
    rejected = example.get("rejected", "")

    # Optionally, you can set a validity check (e.g., minimum length)
    valid = (
        isinstance(prompt, str)
        and isinstance(chosen, str)
        and isinstance(rejected, str)
        and len(chosen.split()) > 5
        and len(rejected.split()) > 5
    )

    return {
        "prompt": prompt,
        "chosen": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen},
        ],
        "rejected": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected},
        ],
        "valid": valid,
    }


def parse_open_reasoning_sample(example, **kwargs):
    """
    Parse a DPO code sample with 'input', 'accepted', and 'rejected' fields into a unified format.

    Args:
        example (dict): One sample from the dataset.

    Returns:
        dict: A dict with fields `prompt`, `chosen`, `rejected`, and `valid`.
    """
    prompt = example.get("prompt", "")
    chosen = example.get("chosen", "")
    rejected = example.get("rejected", "")

    # Optionally, you can set a validity check (e.g., minimum length)
    valid = (
        isinstance(prompt, str)
        and isinstance(chosen, str)
        and isinstance(rejected, str)
        and len(chosen.split()) > 5
        and len(rejected.split()) > 5
    )

    return {
        "prompt": prompt,
        "chosen": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen},
        ],
        "rejected": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected},
        ],
        "valid": valid,
    }
