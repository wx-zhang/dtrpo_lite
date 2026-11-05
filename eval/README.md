# Evaluation

Seven benchmarks share one entry point: `python -m eval.run_benchmark`.
It generates responses with `FastdLLMGenerator.batch_generate()` and scores
those responses. The shell script `scripts/eval.sh` contains one explicit
command per dataset, including the generation parameters.

## Install

Run from the repository root. Use Python 3.10, Git, and an NVIDIA GPU with a
CUDA-compatible driver. Install into a clean environment to avoid mixing
incompatible PyTorch, torchvision, or torchaudio installations:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-full-eval.txt
python -m eval.run_benchmark --help
```

`requirements-full-eval.txt` is self-contained for evaluation. Training uses
`requirements.txt`. The EvalPlus dependency is pinned to upstream commit
`6eb1e199c2e370518e7bf3e8eee6322c2b23c89c`, which fixes success bookkeeping
for HumanEval/32 (`find_zero`). No manual edits to installed packages are
required. This package revision is separate from the benchmark data versions:
HumanEval+ v0.1.10 and MBPP+ v0.2.0.

Models and datasets download automatically when missing. Hugging Face uses
its normal cache (`HF_HOME` can override it); EvalPlus uses the user cache
(`XDG_CACHE_HOME` can override it). The scripts do not enable offline mode.
If your shell previously enabled it, run:

```bash
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE
```

For GPQA, obtain access to [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa)
and log in with an authorized Hugging Face account:

```bash
hf auth login
```

## Run

Activate the environment and run from the repository root:

```bash
bash scripts/eval.sh
```

Edit `MODEL_PATH` in that script to select a Hugging Face model ID or a local
checkpoint directory. `OUTPUT_DIR` defaults to `./results` and can be overridden
as an environment variable. The commands run sequentially using the CUDA
visibility configured by the caller; the script does not schedule GPUs.

For a single benchmark or a small smoke test:

```bash
python -m eval.run_benchmark --task gsm8k \
  --model_path wxzhang/Fast-dLLM-v2-7B-dTRPO-Lite \
  --output_dir results/gsm8k-smoke \
  --threshold 1.0 --temperature 0.0 --repetition_penalty 1.1 \
  --max_new_tokens 2048 --batch_size 32 --top_p 0.95 --limit 2
```

Omit `--limit` for the full dataset. For MATH, the limit applies to each of
its seven subjects. `--score_workers` controls CPU workers for code tests
(default: 8). Code evaluation executes generated Python; use an isolated
execution environment when evaluating untrusted models.

## Protocol

All commands use BF16, batch size 32, top-p 0.95, prefix cache and block cache.
Block size is fixed at 32 and sub-block size at 8. The default seed is 0;
use `--seed 1` or `--seed 2` and distinct output directories for additional
runs. Changing batch size can change BF16 outputs.

| Task | Dataset / split | Examples | Metric | Threshold | Temperature | Repetition penalty | Max new tokens |
|---|---|---:|---|---:|---:|---:|---:|
| `gsm8k` | GSM8K main / test, CoT | 1319 | strict boxed math_verify | 1.0 | 0.0 | 1.1 | 2048 |
| `math` | MATH test, all seven subjects, CoT | 5000 | strict boxed math_verify | 1.0 | 0.6 | 1.1 | 2048 |
| `gpqa` | GPQA Diamond / train | 198 | flexible exact match | 0.8 | 0.0 | 1.0 | 2048 |
| `lcb` | LiveCodeBench release_v6 | 1055 | pass@1 | 1.0 | 0.2 | 1.0 | 2048 |
| `mbpp_plus` | MBPP+ v0.2.0 | 378 | Plus pass@1 | 1.0 | 0.2 | 1.0 | 1024 |
| `humaneval_plus` | HumanEval+ v0.1.10 | 164 | Plus pass@1 | 1.0 | 0.2 | 1.0 | 1024 |
| `ifeval` | IFEval train | 541 | prompt-level strict accuracy | 1.0 | 0.2 | 1.0 | 1280 |

GSM8K and MATH score only the last complete `\boxed{}` answer; missing boxes
score zero. GPQA keeps its worked example and `The answer is (D).` format.
LCB uses the Lighteval-style chat prompt and official code tests. Plus scores
require passing both base tests and additional tests. Limited EvalPlus runs
use separate reference-result cache keys from full runs.

## Outputs and reruns

Each output directory contains `job.json` (model and decoding settings),
`outcome.json` (score, metric, example count and elapsed time), generated
responses, and benchmark-specific scoring details. A successful full run
checks the expected example count. Scores in JSON are fractions in [0, 1].

Repeating a completed command with identical settings returns its saved result.
Different settings or an incomplete run require a fresh output directory;
partial stochastic generations are not silently reused with a reset RNG.

## Code layout

- `run_benchmark.py`: the only evaluation CLI.
- `common.py`: benchmark identifiers and JSON output helpers.
- `harness_backend.py`: GSM8K, MATH, GPQA and IFEval.
- `evalplus_backend.py`: HumanEval+/MBPP+ download, generation and scoring.
- `livecodebench_backend.py`: LCB prompting, generation and scoring.
- `lm_eval_generator.py`: diffusion-model adapter for lm-evaluation-harness.
- `generate.py`: diffusion decoding implementation.
- `lm_tasks/`: harness task definitions, prompts and answer checking.

`model/` supplies the model implementation; `lcb_runner/` supplies LCB data
and test execution. LCB prompting and code extraction are kept in its backend.
