"""
DPO (Direct Preference Optimization) training for diffusion language models.

Adapts DPO to discrete diffusion LLMs by replacing autoregressive log-prob
computation with diffusion-based log-likelihood estimation. 

"""

import argparse
import datetime
import importlib
import logging
import os

os.environ["TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE"] = "0"
os.environ["TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE"] = "0"
os.environ["TORCHINDUCTOR_AUTOTUNE_LOCAL_CACHE"] = "1"

import torch
import torch._inductor.config as inductor_config

inductor_config.fx_graph_remote_cache = False
inductor_config.autotune_remote_cache = False
inductor_config.autotune_local_cache = True

from dataclasses import dataclass, field
from typing import Optional

import trl
from data.data_mixture import get_dataset, ScriptArguments
from model.modeling_fastdllm import Fast_dLLM_QwenForCausalLM
from torch import distributed as dist
from trainer.distributed_dtrpo_trainer import FastDLLMDistributedDTRPOTrainer
from trainer.dpo_fast_dllm_trainer import FastDLLMDPOTrainer
from trainer.dpo_base import DLDPOConfig
from trainer.dpo_llada_trainer import DPOLLaDATrainer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from trl import TrlParser

try:
    from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE as SIMPLE_CHAT_TEMPLATE
except ImportError:
    SIMPLE_CHAT_TEMPLATE = (
        "{% for message in messages %}"
        "{{message['role'] + ': ' + message['content'] + '\n\n'}}"
        "{% endfor %}{{ eos_token }}"
    )

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig(trl.ModelConfig):
    """Model-related arguments."""
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "Model architecture: 'fastdllm' or 'llada'"},
    )
    mlp_only: bool = field(
        default=False,
        metadata={"help": "Only update MLP layers (freeze all other parameters)"},
    )
    dtype: Optional[str] = field(
        default=None,
        metadata={"help": "Model loading dtype (e.g. 'bfloat16', 'float32', or 'auto')."},
    )
    trainer_type: str = field(
        default="standard",
        metadata={"help": "Trainer implementation: 'standard' or 'distributed_dtrpo'."},
    )



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(script_args, training_args, model_args):
    model_args.model_type = (model_args.model_type or "").lower()
    # ── Dataset ──────────────────────────────────────────────────────────
    dataset = get_dataset(script_args)

    # ── Model & Tokenizer ────────────────────────────────────────────────
    torch_dtype = (
        model_args.dtype
        if model_args.dtype in ["auto", None]
        else getattr(torch, model_args.dtype)
    )

    if model_args.model_type == "fastdllm":
        model_kwargs = dict(
            torch_dtype=torch_dtype,
            use_cache=not training_args.gradient_checkpointing,
        )
        model = Fast_dLLM_QwenForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs,
        )
        ref_model = Fast_dLLM_QwenForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs,
        )
    elif model_args.model_type == "llada":
        model_kwargs = dict(
            revision=model_args.model_revision,
            attn_implementation=model_args.attn_implementation,
            torch_dtype=torch_dtype,
            use_cache=not (
                training_args.gradient_checkpointing
                or model_args.model_type == "llada"
            ),
            device_map=None,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs,
        )
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_args.model_type}")

    # LLaDA activation checkpointing
    if model_args.model_type == "llada" and training_args.gradient_checkpointing:
        cfg = AutoConfig.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=True
        )
        cfg_module = importlib.import_module(cfg.__class__.__module__)
        ActivationCheckpointingStrategy = getattr(
            cfg_module, "ActivationCheckpointingStrategy"
        )
        model.model.set_activation_checkpointing(
            ActivationCheckpointingStrategy.whole_layer
        )
        model.model.gradient_checkpointing = True
        training_args.gradient_checkpointing = False

    # Reference model (for DPO KL divergence)
    ref_model.train(model_args.model_type == "fastdllm")
    ref_model.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        fix_mistral_regex=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    if script_args.ignore_bias_buffers:
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # Optional: freeze everything except MLP layers
    if model_args.mlp_only and model_args.model_type == "fastdllm":
        log.info("Freezing all parameters except MLP layers")
        num_frozen = num_trainable = 0
        for name, param in model.named_parameters():
            if "mlp" in name.lower():
                param.requires_grad = True
                num_trainable += 1
            else:
                param.requires_grad = False
                num_frozen += 1
        log.info(f"Frozen {num_frozen} params, {num_trainable} MLP params trainable")

    # ── Training ─────────────────────────────────────────────────────────
    if model_args.model_type == "llada":
        if model_args.trainer_type != "standard":
            raise ValueError("--trainer_type distributed_dtrpo requires --model_type fastdllm")
        trainer_cls = DPOLLaDATrainer
    elif model_args.model_type == "fastdllm":
        if model_args.trainer_type == "standard":
            trainer_cls = FastDLLMDPOTrainer
        elif model_args.trainer_type == "distributed_dtrpo":
            trainer_cls = FastDLLMDistributedDTRPOTrainer
        else:
            raise ValueError(
                f"Unknown trainer_type: {model_args.trainer_type!r}. "
                "Expected 'standard' or 'distributed_dtrpo'."
            )
    else:
        raise ValueError(f"Unknown model_type: {model_args.model_type}")

    trainer = trainer_cls(
        model,
        ref_model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(
            dataset[script_args.dataset_test_split]
            if training_args.eval_strategy != "no"
            else None
        ),
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    log.info("Saving model")
    trainer.save_model(training_args.output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, DLDPOConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser(
            "dpo",
            help="DPO training for diffusion language models",
            dataclass_types=dataclass_types,
        )
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_and_config()
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl", timeout=datetime.timedelta(hours=8)
        )
    try:
        main(script_args, training_args, model_args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
