#!/bin/bash

timestamp=$(date +"%m%d_%H%M")
model_name="Qwen2.5-7B-Instruct"

BASE_MODEL_PATH="hdfs://harunava/home/byte_data_seed_azure/alphaseed/daiweinan/models/Qwen2.5-7B-Instruct"
MODEL_PATH="/mnt/hdfs/length_scaling/experiments/mbti/multi_profile/mbti-sft_Qwen2.5-7B-Instruct_0427_1806/global_step_2262"

TEST_FILE="/opt/tiger/dwn-opensource-verl/mbti_data/mbti_Q_144_nooption.parquet"
OUTPUT_PATH="/opt/tiger/dwn-opensource-verl/mbti_hub/output/answer/${TYPE}_${timestamp}.parquet"

TEMPERATURE=0.7
TOP_P=0.7
MAX_TOKENS=128
DTYPE="bf16"
BATCH_SIZE=1024
TP_SIZE=4
GPU_MEMORY_UTILIZATION=0.95

MERGED_MODEL_SAVE_PATH="/opt/tiger/dwn-opensource-verl/mbti_hub/ckpt"

python /opt/tiger/dwn-opensource-verl/mbti_hub/script/vllm_infer_lora_multi.py \
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
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --merged_model_save_path "$MERGED_MODEL_SAVE_PATH" \