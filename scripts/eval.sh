#!/usr/bin/env bash
set -euo pipefail
OUTPUT_DIR="${OUTPUT_DIR:-./results}"
MODEL_PATH=wxzhang/Fast-dLLM-v2-7B-dTRPO-Lite
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1

python -m eval.run_benchmark --task math \
  --model_path "$MODEL_PATH" --output_dir "$OUTPUT_DIR/math" \
  --threshold 1.0 --temperature 0.6 --repetition_penalty 1.1 --max_new_tokens 2048 \
  --batch_size 32 --top_p 0.95

python -m eval.run_benchmark --task gsm8k \
  --model_path "$MODEL_PATH" --output_dir "$OUTPUT_DIR/gsm8k" \
  --threshold 1.0 --temperature 0.0 --repetition_penalty 1.1 --max_new_tokens 2048 \
  --batch_size 32 --top_p 0.95

python -m eval.run_benchmark --task gpqa \
  --model_path "$MODEL_PATH" --output_dir "$OUTPUT_DIR/gpqa" \
  --threshold 0.8 --temperature 0.0 --repetition_penalty 1.0 --max_new_tokens 2048 \
  --batch_size 32 --top_p 0.95

python -m eval.run_benchmark --task lcb \
  --model_path "$MODEL_PATH" --output_dir "$OUTPUT_DIR/lcb" \
  --threshold 1.0 --temperature 0.2 --repetition_penalty 1.0 --max_new_tokens 2048 \
  --batch_size 32 --top_p 0.95

python -m eval.run_benchmark --task mbpp_plus \
  --model_path "$MODEL_PATH" --output_dir "$OUTPUT_DIR/mbpp_plus" \
  --threshold 1.0 --temperature 0.2 --repetition_penalty 1.0 --max_new_tokens 1024 \
  --batch_size 32 --top_p 0.95

python -m eval.run_benchmark --task humaneval_plus \
  --model_path "$MODEL_PATH" --output_dir "$OUTPUT_DIR/humaneval_plus" \
  --threshold 1.0 --temperature 0.2 --repetition_penalty 1.0 --max_new_tokens 1024 \
  --batch_size 32 --top_p 0.95

python -m eval.run_benchmark --task ifeval \
  --model_path "$MODEL_PATH" --output_dir "$OUTPUT_DIR/ifeval" \
  --threshold 1.0 --temperature 0.2 --repetition_penalty 1.0 --max_new_tokens 1280 \
  --batch_size 32 --top_p 0.95
