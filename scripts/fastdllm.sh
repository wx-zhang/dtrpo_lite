#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Efficient-Large-Model/Fast_dLLM_v2_7B}"
OUTPUT_DIR="${OUTPUT_DIR:-output/fastdllm-dtrpo}"
DATA_CONFIG="${DATA_CONFIG:-configs/smoltalk2_dpo.yaml}"
export TOKENIZERS_PARALLELISM=false

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" -m train \
  --model_name_or_path "$MODEL_PATH" --model_type fastdllm \
  --trainer_type distributed_dtrpo \
  --dataset_mixture_yaml "$DATA_CONFIG" --dataset_streaming False \
  --output_dir "$OUTPUT_DIR" \
  --dtype bfloat16 --bf16 True \
  --learning_rate 5e-7 --num_train_epochs 1 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 64 \
  --warmup_ratio 0.1 --lr_scheduler_type cosine_with_min_lr \
  --lr_scheduler_kwargs '{"min_lr_rate": 0.1}' \
  --logging_steps 1 --eval_strategy no \
  --save_strategy steps --save_steps 100 --save_total_limit 2 \
  --no_remove_unused_columns --beta 0.01 --loss_type sigmoid \
  --logp_type dtrpo --mc_samples 4 --mc_batch_size 1 \
  --block_size 32 --max_length 2048 --max_grad_norm 0.2 \
  --f_divergence_type reverse_kl --truncation_mode keep_start \
  --disable_dropout True --gradient_checkpointing False \
  --trust_remote_code True --report_to none \
  --fsdp 'full_shard auto_wrap' --fsdp_config configs/fsdp_fastdllm.json \
  "$@"
