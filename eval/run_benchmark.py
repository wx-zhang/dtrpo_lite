"""Generate and score one of the seven supported Fast-dLLM benchmarks."""
import argparse
import json
from pathlib import Path
import time

from eval.common import SIZES, TASKS, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=list(SIZES))
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--repetition_penalty", type=float, required=True)
    parser.add_argument("--max_new_tokens", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--bd_size", type=int, choices=[32], default=32)
    parser.add_argument("--small_block_size", type=int, choices=[8], default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--score_workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test limit (per subject for MATH)")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1 or args.score_workers < 1:
        parser.error("batch_size, max_new_tokens and score_workers must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if not 0 <= args.threshold <= 1 or not 0 < args.top_p <= 1:
        parser.error("threshold must be in [0, 1] and top_p in (0, 1]")
    if args.temperature < 0 or args.repetition_penalty <= 0:
        parser.error("temperature must be nonnegative and repetition_penalty positive")

    directory = Path(args.output_dir).resolve()
    model = str(Path(args.model_path).resolve()) if Path(args.model_path).is_dir() else args.model_path
    config = {key: getattr(args, key) for key in (
        "threshold", "temperature", "repetition_penalty", "max_new_tokens",
        "batch_size", "top_p", "bd_size", "small_block_size", "seed")}
    config.update(use_prefix_cache=True, use_block_cache=True)
    job = dict(task=args.task, config=config, model=model, directory=str(directory),
               stage="full" if args.limit is None else "smoke", limit=args.limit,
               score_workers=args.score_workers)
    job_path = directory / "job.json"
    outcome_path = directory / "outcome.json"
    if job_path.exists():
        if json.loads(job_path.read_text()) != job:
            raise ValueError("Output directory belongs to different settings; choose a fresh --output_dir")
        if outcome_path.exists():
            result = json.loads(outcome_path.read_text())
            if result["status"] == "complete":
                print(json.dumps(result, indent=2))
                return
        raise RuntimeError("Incomplete run; choose a fresh --output_dir to preserve seeded sampling")
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Output directory is not empty; choose a fresh --output_dir")

    if args.task in TASKS:
        from eval.harness_backend import run
    elif args.task == "lcb":
        from eval.livecodebench_backend import run
    else:
        from eval.evalplus_backend import run

    import torch
    from eval.generate import FastdLLMGenerator
    torch.set_num_threads(4)
    generator = FastdLLMGenerator(model_path=model)
    # Model loading must not consume the generation random stream.
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    write_json(job_path, job)
    start = time.monotonic()
    result = run(generator, job, directory)
    if args.limit is None and result["n"] != SIZES[args.task]:
        raise RuntimeError(f"Wrong full-set size: {result['n']} != {SIZES[args.task]}")
    result.update(job, status="complete", seconds=time.monotonic() - start,
                  peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9)
    write_json(outcome_path, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
