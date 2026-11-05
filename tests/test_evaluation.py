"""CPU regression checks for the unified evaluation pipeline."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from eval import evalplus_backend, harness_backend, livecodebench_backend


CONFIG = dict(threshold=1.0, temperature=0.2, repetition_penalty=1.0,
              max_new_tokens=1024, batch_size=32, top_p=0.95, bd_size=32,
              small_block_size=8, seed=0, use_prefix_cache=True, use_block_cache=True)


class Tokenizer:
    def __call__(self, text):
        return SimpleNamespace(input_ids=[list(range(len(s))) for s in text]
                               if isinstance(text, list) else list(range(len(text))))

    def decode(self, tokens):
        return "decoded prompt"

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]


class Generator:
    tokenizer = Tokenizer()

    def __init__(self, response):
        self.response = response
        self.calls = []

    def batch_generate(self, prompts, **kwargs):
        self.calls.append((prompts, kwargs))
        return [self.response for _ in prompts]


class EvaluationTests(unittest.TestCase):
    def test_cli_help_and_fixed_structure(self):
        command = [sys.executable, "-m", "eval.run_benchmark"]
        result = subprocess.run(command + ["--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run(command + ["--task", "gsm8k", "--model_path", "unused",
            "--output_dir", "unused", "--threshold", "1", "--temperature", "0",
            "--repetition_penalty", "1.1", "--max_new_tokens", "64",
            "--small_block_size", "16"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid choice", result.stderr)

    def test_evalplus_sampling_parameters_and_metadata(self):
        generator = Generator("def answer(): return 42")
        problems = {"HumanEval/0": dict(prompt="def answer():", entry_point="answer")}
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            path = evalplus_backend._generate_dataset(
                generator, "humaneval_plus", problems, directory, model_path="owner/model",
                **{k: v for k, v in CONFIG.items() if not k.startswith("use_")},
                make_raw_chat_prompt=lambda prompt, *args: prompt,
                sanitize=lambda code, **kwargs: code,
            )
            self.assertEqual(len(path.read_text().splitlines()), 1)
            recorded = json.loads((directory / "run_config.json").read_text())
            self.assertEqual(recorded["model_path"], "owner/model")
            for key in ("temperature", "top_p", "bd_size", "small_block_size", "batch_size"):
                self.assertEqual(recorded[key], CONFIG[key])
                self.assertEqual(generator.calls[0][1][key], CONFIG[key])

    def test_find_zero_reference_passes_and_wrong_solution_fails(self):
        problem = dict(task_id="HumanEval/32", entry_point="find_zero", atol=1e-6,
            prompt="def find_zero(xs):\n", canonical_solution="    return -xs[0] / xs[1]\n",
            base_input=[([-2.0, 1.0],)], plus_input=[([-6.0, 2.0],)])
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            cache = directory / "cache"
            cache.mkdir()
            with patch("evalplus.data.get_human_eval_plus_hash", return_value="fixture-v1"), \
                 patch("evalplus.data.utils.CACHE_DIR", str(cache)), \
                 patch("evalplus.evaluate.CACHE_DIR", str(cache)):
                for solution, score in [(problem["prompt"] + problem["canonical_solution"], 1.0),
                                        ("def find_zero(xs): return 0.0", 0.0)]:
                    samples = directory / "samples.jsonl"
                    samples.write_text(json.dumps(dict(task_id="HumanEval/32", solution=solution)) + "\n")
                    result = evalplus_backend._score_dataset(
                        "humaneval_plus", {"HumanEval/32": problem}, samples, directory, workers=1)
                    self.assertEqual(result["plus_pass_at_1"], score)

    def test_lcb_generation_and_official_scoring(self):
        sample = {"input_output": json.dumps(dict(inputs=["2 3\n"], outputs=["5\n"]))}
        problem = SimpleNamespace(question_id="fixture", question_content="Add two integers.",
                                  starter_code="", get_evaluation_sample=lambda: sample)
        generator = Generator("```python\na, b = map(int, input().split())\nprint(a + b)\n```")
        job = dict(task="lcb", config=CONFIG, model="owner/model", limit=None, score_workers=1)
        with tempfile.TemporaryDirectory() as d, \
             patch("lcb_runner.benchmarks.code_generation.load_code_generation_dataset",
                   return_value=[problem]) as loader:
            result = livecodebench_backend.run(generator, job, Path(d))
            loader.assert_called_once_with("release_v6")
            self.assertEqual(result["score"], 1.0)
            self.assertEqual(result["n"], 1)
            self.assertEqual(generator.calls[0][1]["temperature"], 0.2)

    def test_harness_sampling_override_and_score(self):
        result = dict(results={}, samples={"fixture": [{"math_verify": 1}, {"math_verify": 0}]})
        job = dict(task="gsm8k", config=CONFIG, model="owner/model", limit=2)
        with tempfile.TemporaryDirectory() as d, \
             patch("eval.lm_eval_generator.DLLMLM") as adapter, \
             patch("lm_eval.tasks.TaskManager"), \
             patch("lm_eval.evaluator.simple_evaluate", return_value=result) as evaluate:
            summary = harness_backend.run(Generator("unused"), job, Path(d))
            self.assertEqual(summary["score"], 0.5)
            self.assertEqual(adapter.call_args.kwargs["generation_overrides"],
                             dict(temperature=0.2, top_p=0.95, max_new_tokens=1024))
            self.assertEqual(evaluate.call_args.kwargs["limit"], 2)


if __name__ == "__main__":
    unittest.main()
