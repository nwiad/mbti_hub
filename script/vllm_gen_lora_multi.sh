#!/bin/bash

timestamp=$(date +"%m%d_%H%M")

model_name="Qwen2.5-7B-Instruct"
exp_name="mbti-sft_${model_name}_${timestamp}"
OUTPUT_PATH="../output/mbti_Q_144_bf16/all_sft/Qwen2.5-7B-Instruct/${TYPE}_${timestamp}.parquet"
MODEL_PATH="/mnt/hdfs/length_scaling/experiments/mbti/multi_profile/${exp_name}"
TEST_FILE="../test_dataset/mbti_Q_144_v3.parquet"

TEMPERATURE=0.7
TOP_P=0.7
MAX_TOKENS=128
DTYPE="bf16"
BATCH_SIZE=8
TP_SIZE=8

BASE_MODEL_PATH="hdfs://harunava/home/byte_data_seed_azure/alphaseed/daiweinan/models/Qwen2.5-7B-Instruct"

MERGED_MODEL_SAVE_PATH="../ckpt/use_use" # todo
python vllm_infer_lora_multi.py \
    --base_model "$BASE_MODEL_PATH" \
    --model "$MODEL_PATH" \
    --test_file "$TEST_FILE" \
    --output_path "$OUTPUT_PATH" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --max_tokens "$MAX_TOKENS" \
    --dtype "$DTYPE" \
    --batch_size "$BATCH_SIZE" \
    --tp_size "$TP_SIZE" \
    --merged_model_save_path "$MERGED_MODEL_SAVE_PATH" \