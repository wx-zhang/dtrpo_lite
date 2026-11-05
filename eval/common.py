"""Benchmark identifiers and output helpers."""
import json
from pathlib import Path

TASKS = {
    "gsm8k": "dtrpo_gsm8k_boxed_prompt_math_verify",
    "math": "dtrpo_minerva_math_boxed_prompt_math_verify",
    "gpqa": "dtrpo_gpqa_diamond_no_box",
    "ifeval": "ifeval",
}
SIZES = {"gsm8k": 1319, "math": 5000, "gpqa": 198, "ifeval": 541,
         "humaneval_plus": 164, "mbpp_plus": 378, "lcb": 1055}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)
