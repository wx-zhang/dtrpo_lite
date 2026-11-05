"""MATH prompt and strict final-box scorer."""

from eval.lm_tasks._boxed import _verify_boxed, last_boxed_content


def math_doc_to_text(doc: dict) -> str:
    return (
        f"Problem:\n{doc['problem']}\n\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    )


def process_math_results(doc: dict, results: list[str]) -> dict[str, int]:
    gold = last_boxed_content(doc["solution"])
    if gold is None:
        raise ValueError("MATH reference solution has no boxed answer")
    prediction = last_boxed_content(results[0])
    return {"math_verify": _verify_boxed(gold, prediction)}
