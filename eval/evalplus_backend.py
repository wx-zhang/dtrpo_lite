"""HumanEval+ and MBPP+ generation and scoring (internal backend)."""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from eval.common import write_json

INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the following "
    "problem in a markdown code block:"
)
RESPONSE_PREFIX = (
    "Below is a Python script with a self-contained function that solves the "
    "problem and passes corresponding tests:"
)
DATASET_VERSIONS = {
    "humaneval_plus": "v0.1.10",
    "mbpp_plus": "v0.2.0",
}


def _json_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _generate_dataset(
    generator,
    dataset_name: str,
    problems: dict,
    output_dir: Path,
    *,
    model_path: str,
    batch_size: int,
    threshold: float,
    max_new_tokens: int,
    seed: int,
    repetition_penalty: float,
    temperature: float,
    top_p: float,
    bd_size: int,
    small_block_size: int,
    make_raw_chat_prompt,
    sanitize,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model_path": model_path,
        "dataset": dataset_name,
        "dataset_version": DATASET_VERSIONS[dataset_name],
        "batch_size": batch_size,
        "threshold": threshold,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "bd_size": bd_size,
        "small_block_size": small_block_size,
        "seed": seed,
        "repetition_penalty": repetition_penalty,
        "cache_mode": "prefix_block",
        "prompt_style": "evalplus_chat",
    }
    config_hash = _json_hash(config)
    write_json(output_dir / "run_config.json", {**config, "config_hash": config_hash})

    raw_path = output_dir / "raw_generations.jsonl"
    completed = {}
    requests = []
    for task_id, problem in problems.items():
        task_prompt = problem["prompt"].strip() + "\n"
        prompt = make_raw_chat_prompt(
            task_prompt,
            INSTRUCTION_PREFIX,
            RESPONSE_PREFIX,
            generator.tokenizer,
        )
        token_ids = generator.tokenizer([prompt]).input_ids[0]
        requests.append((len(token_ids), task_id, prompt, problem["entry_point"]))
    requests.sort(key=lambda item: (item[0], item[1]))

    with raw_path.open("w", encoding="utf-8") as raw_handle:
        for start in range(0, len(requests), batch_size):
            chunk = requests[start : start + batch_size]
            generations = generator.batch_generate(
                [item[2] for item in chunk],
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                threshold=threshold,
                use_prefix_cache=True,
                use_block_cache=True,
                bd_size=bd_size,
                small_block_size=small_block_size,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            for (_, task_id, prompt, entry_point), generation in zip(
                chunk, generations
            ):
                solution = sanitize(generation, entrypoint=entry_point)
                item = {
                    "task_id": task_id,
                    "config_hash": config_hash,
                    "prompt": prompt,
                    "generation": generation,
                    "solution": solution,
                }
                completed[task_id] = item
                raw_handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                raw_handle.flush()
            print(
                f"[{dataset_name}] {min(start + len(chunk), len(requests))}/"
                f"{len(requests)} new; {len(completed)}/{len(problems)} total",
                flush=True,
            )

    missing = [task_id for task_id in problems if task_id not in completed]
    if missing:
        raise RuntimeError(f"Missing {dataset_name} generations: {missing[:10]}")

    samples_path = output_dir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as handle:
        for task_id in problems:
            handle.write(
                json.dumps(
                    {"task_id": task_id, "solution": completed[task_id]["solution"]},
                    ensure_ascii=False,
                )
                + "\n"
            )

    debug_path = output_dir / "input_debug.jsonl"
    with debug_path.open("w", encoding="utf-8") as handle:
        for task_id in list(problems)[:2]:
            item = completed[task_id]
            token_ids = generator.tokenizer([item["prompt"]]).input_ids[0]
            handle.write(
                json.dumps(
                    {
                        "task_id": task_id,
                        "prompt": item["prompt"],
                        "token_ids": token_ids,
                        "decoded_input": generator.tokenizer.decode(token_ids),
                        "generation": item["generation"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return samples_path


def _score_dataset(
    dataset_name: str,
    problems: dict,
    samples_path: Path,
    output_dir: Path,
    workers: int,
) -> dict:
    from evalplus.data import (
        get_human_eval_plus_hash,
        get_mbpp_plus_hash,
        load_solutions,
    )
    from evalplus.eval import PASS
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    from evalplus.evaluate import check_correctness, get_groundtruth
    from evalplus.data.utils import CACHE_DIR
    from filelock import FileLock

    if dataset_name == "humaneval_plus":
        dataset_hash = get_human_eval_plus_hash(version=DATASET_VERSIONS[dataset_name])
        oracle_tasks = []
    else:
        dataset_hash = get_mbpp_plus_hash(version=DATASET_VERSIONS[dataset_name])
        oracle_tasks = MBPP_OUTPUT_NOT_NONE_TASKS

    # A limited run must not populate the full-dataset oracle cache.
    oracle_hash = _json_hash({"dataset_hash": dataset_hash, "task_ids": sorted(problems)})
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    with FileLock(str(Path(CACHE_DIR) / f"{oracle_hash}.lock")):
        expected = get_groundtruth(problems, oracle_hash, oracle_tasks)
    # EvalPlus uses these names to select benchmark-specific special oracles.
    oracle_dataset = {"humaneval_plus": "humaneval", "mbpp_plus": "mbpp"}[dataset_name]

    solutions = {item["task_id"]: item["solution"] for item in load_solutions(str(samples_path))}
    if set(solutions) != set(problems):
        raise RuntimeError(
            f"{dataset_name} sample IDs do not match dataset: "
            f"samples={len(solutions)} problems={len(problems)}"
        )

    details = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_task = {
            executor.submit(
                check_correctness,
                oracle_dataset,
                0,
                problem,
                solutions[task_id],
                expected[task_id],
                False,
                True,
                task_id,
            ): task_id
            for task_id, problem in problems.items()
        }
        for future in as_completed(future_to_task):
            result = future.result()
            details.append(
                {
                    "task_id": result["task_id"],
                    "base_status": result["base"][0],
                    "plus_status": result["plus"][0],
                }
            )

    details.sort(key=lambda item: item["task_id"])
    base_correct = sum(item["base_status"] == PASS for item in details)
    plus_correct = sum(
        item["base_status"] == PASS and item["plus_status"] == PASS
        for item in details
    )
    summary = {
        "dataset": dataset_name,
        "dataset_version": DATASET_VERSIONS[dataset_name],
        "dataset_hash": dataset_hash,
        "num_tasks": len(details),
        "base_correct": base_correct,
        "plus_correct": plus_correct,
        "base_pass_at_1": base_correct / len(details),
        "plus_pass_at_1": plus_correct / len(details),
    }
    write_json(output_dir / "score_summary.json", summary)
    with (output_dir / "score_details.jsonl").open("w", encoding="utf-8") as handle:
        for item in details:
            handle.write(json.dumps(item) + "\n")
    (output_dir / "complete").touch()
    print(json.dumps(summary, indent=2))
    return summary


def run(generator, job, directory):
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    from evalplus.data.utils import CACHE_DIR
    from evalplus.provider.utility import make_raw_chat_prompt
    from evalplus.sanitize import sanitize
    from filelock import FileLock

    task = job["task"]
    loader = get_human_eval_plus if task == "humaneval_plus" else get_mbpp_plus
    # Serialize the upstream downloader when separate benchmark processes share a cache.
    cache = Path(CACHE_DIR)
    cache.mkdir(parents=True, exist_ok=True)
    with FileLock(str(cache / "dataset-download.lock")):
        problems = loader(version=DATASET_VERSIONS[task])
    if job["limit"] is not None:
        problems = dict(list(problems.items())[:job["limit"]])
    c = job["config"]
    samples = _generate_dataset(
        generator, task, problems, directory, model_path=job["model"],
        **{k: c[k] for k in ("batch_size", "threshold", "max_new_tokens", "seed",
                            "repetition_penalty", "temperature", "top_p",
                            "bd_size", "small_block_size")},
        make_raw_chat_prompt=make_raw_chat_prompt, sanitize=sanitize,
    )
    summary = _score_dataset(task, problems, samples, directory, workers=job["score_workers"])
    return dict(score=summary["plus_pass_at_1"], metric="plus_pass@1", n=summary["num_tasks"])
