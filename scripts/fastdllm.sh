#!/bin/sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# DPO training script for Fast-dLLM


set -e

# Wandb configuration
export WANDB_PROJECT="${WANDB_PROJECT:-dllm}"
export WANDB_LOG_MODEL="false"

MODEL_PATH="Efficient-Large-Model/Fast_dLLM_v2_1.5B"
data="configs/smoltalk2_dpo.yaml"
output="fastdllm_smoltalk2_dtrpo"

torchrun --nproc_per_node=4 --master_port=29502 -m train \
--model_name_or_path "$MODEL_PATH" \
--model_type fastdllm \
--dataset_mixture_yaml ${data} \
--output_dir ${output} \
--run_name ${output} \
--dtype "bfloat16" \
--bf16 True \
--dataset_streaming False \
--learning_rate 1.0e-7 \
--num_train_epochs 1 \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 8 \
--max_steps 5 \
--warmup_ratio 0.1 \
--lr_scheduler_type "cosine_with_min_lr" \
--lr_scheduler_kwargs '{"min_lr_rate": 0.1}' \
--logging_steps 1 \
--logging_strategy steps \
--log_level info \
--eval_strategy no \
--save_strategy steps \
--save_steps 1000 \
--save_total_limit 2 \
--use_liger_kernel True \
--no_remove_unused_columns \
--beta 0.05 \
--loss_type sigmoid \
--logp_type dtrpo \
--mc_samples 1 \
--mc_batch_size 1 \
--block_size 32 \
--max_length 2048 \
--max_grad_norm 0.2 \
--f_divergence_type "reverse_kl" \
--truncation_mode keep_start \
--gradient_checkpointing False \
--trust_remote_code True \
--report_to wandb \
--fsdp "full_shard auto_wrap" \
--fsdp_config '{"activation_checkpointing": true, "activation_checkpointing_policy": "auto", "mixed_precision.param_dtype": "bfloat16"}'
