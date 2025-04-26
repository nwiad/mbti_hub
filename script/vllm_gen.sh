#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3



MODEL_PATH="/yeesuanAI08/LLM/Qwen2.5-7B-Instruct"
TEST_FILE="../test_dataset/mbti_Q_144_v3.parquet"

TEMPERATURE=0.7
TOP_P=0.7
MAX_TOKENS=1024
DTYPE="bf16"
BATCH_SIZE=4
TP_SIZE=4

python vllm_infer_multi.py \
     --model "$MODEL_PATH" \
     --test_file "$TEST_FILE" \
     --temperature "$TEMPERATURE" \
     --top_p "$TOP_P" \
     --max_tokens "$MAX_TOKENS" \
     --dtype "$DTYPE" \
     --batch_size "$BATCH_SIZE" \
     --tp_size "$TP_SIZE"\
     --target_type "$TYPE"\
     --prompt_type "characteristics" 