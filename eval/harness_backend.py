"""GSM8K, MATH, GPQA Diamond and IFEval through lm-evaluation-harness."""
import json
from pathlib import Path

from eval.common import TASKS, write_json


def run(generator, job, directory):
    from lm_eval import evaluator
    from lm_eval.tasks import TaskManager
    from eval.lm_eval_generator import DLLMLM

    c = job["config"]
    model = DLLMLM(
        model_path=job["model"], output_dir=str(directory), generator=generator,
        **{k: c[k] for k in ("batch_size", "threshold", "repetition_penalty", "bd_size",
                            "small_block_size", "use_prefix_cache", "use_block_cache", "seed")},
        generation_overrides={k: c[k] for k in ("temperature", "top_p", "max_new_tokens")},
    )
    result = evaluator.simple_evaluate(
        model=model, tasks=[TASKS[job["task"]]], num_fewshot=0,
        batch_size=c["batch_size"], limit=job["limit"], bootstrap_iters=0,
        log_samples=True, apply_chat_template=True, fewshot_as_multiturn=True,
        task_manager=TaskManager(include_path=str(Path(__file__).parent / "lm_tasks")),
        random_seed=c["seed"], numpy_random_seed=c["seed"],
        torch_random_seed=c["seed"], fewshot_random_seed=c["seed"],
    )
    if result is None:
        raise RuntimeError("lm-eval returned no results")
    samples = result.pop("samples")
    write_json(directory / "results.json", result)
    for task, rows in samples.items():
        with (directory / f"samples_{task}.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
    rows = [r for records in samples.values() for r in records]
    metric = ("math_verify" if job["task"] in ("math", "gsm8k") else
              "exact_match" if job["task"] == "gpqa" else "prompt_level_strict_acc")
    return dict(score=sum(float(r[metric]) for r in rows) / len(rows), metric=metric, n=len(rows))
