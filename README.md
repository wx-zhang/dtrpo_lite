# dTRPO Lite: Distributed Training and Evaluation with a Single Node

[![Paper](https://img.shields.io/badge/Notion.svg)](https://app.notion.com/p/dTRPO-Lite-Distributed-Monte-Carlo-Estimation-40f383d2887a80c19a2cf2a23f186951?source=copy_link)
[![Model](https://img.shields.io/badge/Hugging%20Face-dTRPO--Lite-yellow.svg)](https://huggingface.co/wxzhang/Fast-dLLM-v2-7B-dTRPO-Lite)

This repository is a follow-up to the original **dTRPO: Trajectory Reduction
in Policy Optimization of Diffusion Large Language Models** implementation.
It adds two practical components:

1. **Distributed MC calculation in Fast-dLLM training:** dTRPO preference optimization with
   DDP/FSDP, Monte Carlo scoring, gradient accumulation, and a frozen reference.
2. **Seven-benchmark evaluation:** batched diffusion decoding, dataset-specific
   generation settings, scoring code, and the released
   [Fast-dLLM-v2-7B-dTRPO-Lite checkpoint](https://huggingface.co/wxzhang/Fast-dLLM-v2-7B-dTRPO-Lite).


## Install

Use **Python 3.10**, Git, and an NVIDIA GPU with a driver compatible with
PyTorch 2.10.0. Run commands from the repository root.

```bash
git clone https://github.com/wx-zhang/dllm_dtrpo.git
cd dllm_dtrpo
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Training
python -m pip install -r requirements.txt

# Evaluation (also usable by itself in a separate environment)
python -m pip install -r requirements-full-eval.txt
```
## Layout and checks

```text
train.py                              Training entry point
trainer/distributed_dtrpo_trainer.py   Distributed dTRPO training
trainer/dpo_*                         Standard trainers and shared scoring
trainer/trl_compat.py                  Preference formatting and tensor helpers
model/                                Fast-dLLM model implementation
data/ and configs/                    Preference dataset loading and mixtures
eval/run_benchmark.py                 Unified benchmark entry point
eval/*_backend.py                     Benchmark generation and scoring
lcb_runner/                           LCB data loader and code-test execution
scripts/                              Training and evaluation commands
tests/                                CPU regression tests
```

After installing both requirements files, run the CPU checks with:

```bash
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -v
```

Code benchmarks execute generated Python; use an isolated evaluation
environment. GPU training and full model evaluation require separate runs.


## Distributed Fast-dLLM training


### dTRPO objective and surrogate gradient

For a preference pair $(x_i,y_i^+,y_i^-)$, let the reduced trajectory score
for MC sample $m$ be

$$
s_{\theta,im}^{\pm}
= K\sum_b\sum_{t\in S_{imb}^{\pm}}
\log p_\theta(y_{i,t}^{\pm}\mid x_i),
$$

where $K$ is `block_size`,
and $S$ contains the policy-selected unmasking positions within each block.
The frozen reference computes
$s_{\mathrm{ref},im}^{\pm}$ on the same noisy inputs and selected positions.
For $B$ preference pairs and $M$ MC samples, the dTRPO objective is

$$
\Delta_i=\frac{1}{M}\sum_{m=1}^{M}
\left[(s_{\theta,im}^{+}-s_{\mathrm{ref},im}^{+})
-(s_{\theta,im}^{-}-s_{\mathrm{ref},im}^{-})\right],
\qquad
\mathcal L_{\mathrm{dTRPO}}
=-\frac{1}{B}\sum_{i=1}^{B}\log\sigma(\beta\Delta_i).
$$

Define a weight with **stop-gradient** and a surrogate loss:

$$
w_i=\operatorname{stopgrad}\!\left[\beta\sigma(-\beta\Delta_i)\right],
\qquad
\mathcal L_{\mathrm{sur}}
=\frac{1}{BM}\sum_{i=1}^{B}\sum_{m=1}^{M}
w_i\cdot(s_{\theta,im}^{-}-s_{\theta,im}^{+}).
$$

For fixed sampled states and selected positions, with a frozen reference,
they have the same gradient at the current parameters:

$$
\nabla_\theta\mathcal L_{\mathrm{dTRPO}}
=-\frac{1}{B}\sum_i w_i\nabla_\theta\Delta_i
=\nabla_\theta\mathcal L_{\mathrm{sur}}.
$$

The surrogate's differentiable contributions are **summed outside the
sigmoid**. The sigmoid only determines the detached coefficient $w_i$.
Consequently, MC microbatches can contribute gradients independently through
ordinary gradient accumulation, followed by the existing DDP/FSDP gradient
synchronization. Their summed gradients are the dTRPO gradient, so the trainer
can reuse the existing distributed training machinery.



### Implementation

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 bash scripts/fastdllm.sh
```

The script starts `train.py` with `--trainer_type distributed_dtrpo`, FSDP,
BF16, and the mixture in `configs/smoltalk2_dpo.yaml`. It uses the 7B base
model, learning rate `5e-7`, beta `0.01`, four MC samples, and an MC microbatch size of one. Currently implemention is for 4 H100 GPUs. 
Adjust batch size and accumulation to fit the available memory.


`MODEL_PATH`, `OUTPUT_DIR`, `DATA_CONFIG`, and `NPROC_PER_NODE` can be set in
the environment. Additional arguments are forwarded to `train.py`.
For direct use, launch `torchrun --nproc_per_node=... -m train` with the
arguments in the script. `--trainer_type standard` selects the original
trainer. The existing `scripts/llada.sh` uses the standard LLaDA trainer.

## Released model and evaluation

The following results are from the completed evaluation of the Lite checkpoint.
They are separate from the original paper's benchmark figures.

| Dataset | Split / setting | Result |
|---|---|---:|
| GSM8K | CoT · test | 88.02% |
| MATH | CoT · full test  | 62.33% |
| GPQA | Diamond | 31.82% |
| LiveCodeBench | release_v6 · pass@1 | 17.06% |
| MBPP+ | v0.2.0 · Plus pass@1 | 57.50% |
| HumanEval+ | v0.1.10 · Plus pass@1 | 69.11% |
| IFEval | Strict · prompt-level | 70.49% |

MATH, LiveCodeBench, MBPP+, HumanEval+, and IFEval report the mean over seeds
0, 1, and 2. GSM8K and GPQA use greedy decoding.

### Run the benchmarks

GPQA requires access to [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa).
Request access there and authenticate with an authorized account:

```bash
hf auth login
CUDA_VISIBLE_DEVICES=0 bash scripts/eval.sh
```

[`scripts/eval.sh`](scripts/eval.sh) contains seven explicit commands with
their final generation settings. All use the single entry point
`python -m eval.run_benchmark`. The script runs one seed (`0`) sequentially;
use `--seed 1` and `--seed 2` with separate output directories for the other
runs in the reported means.


## Fixes included

- Policy and reference score the same noisy input and selected DTRPO positions.
  The reference stays frozen while using Fast-dLLM's training forward path.
- Diffusion log-probabilities are calculated in FP32; token IDs remain integer
  tensors, including empty/padded inputs.




## Citation and license

See [the dTRPO paper](https://arxiv.org/abs/2603.18806) for the trajectory
reduction derivation. This follow-up retains the original project license and
attribution. See [LICENSE](LICENSE), [CONTRIBUTING.md](CONTRIBUTING.md), and
[the LiveCodeBench attribution](lcb_runner/README.md).

```bibtex
@article{zhang2026dtrpo,
  title={dTRPO: Trajectory Reduction in Policy Optimization of Diffusion Large Language Models},
  author={Wenxuan Zhang and Lemeng Wu and Changsheng Zhao and Ernie Chang and Mingchen Zhuge and Zechun Liu and Andy Su and Hanxian Huang and Jun Chen and Chong Zhou and Raghuraman Krishnamoorthi and Vikas Chandra and Mohamed Elhoseiny and Wei Wen},
  year={2026},
  eprint={2603.18806},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2603.18806}
}
```

Thanks to [TRL](https://github.com/huggingface/trl),
[Fast-dLLM](https://github.com/NVlabs/Fast-dLLM),
[LLaDA](https://github.com/GSAI-ML/LLaDA),
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness),
[EvalPlus](https://github.com/evalplus/evalplus), and
[LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench).
