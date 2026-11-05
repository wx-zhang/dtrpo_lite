"""GSM8K prompt and strict final-box scorer."""

import re

from eval.lm_tasks._boxed import _verify_boxed, last_boxed_content


GSM_GOLD_RE = re.compile(r"####\s*(-?[0-9.,]+)")


def gsm_doc_to_text(doc: dict) -> str:
    return (
        f"Question: {doc['question']}\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    )


def process_gsm_results(doc: dict, results: list[str]) -> dict[str, int]:
    gold_match = GSM_GOLD_RE.search(doc["answer"])
    if gold_match is None:
        raise ValueError(f"GSM8K answer has no #### target: {doc['answer']!r}")
    gold = gold_match.group(1).replace(",", "")
    prediction = last_boxed_content(results[0])
    return {"math_verify": _verify_boxed(gold, prediction)}
