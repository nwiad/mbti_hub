#!/bin/bash


timestamp=$(date +"%m%d_%H%M")

MODEL_NAME="Qwen2.5-7B-Instruct"
MODEL_PATH="/opt/tiger/dwn-opensource-verl/mbti_hub/down0506/global_step_62"

OUTPUT_PATH="/opt/tiger/dwn-opensource-verl/mbti_hub/output_final/$MODEL_NAME/tomi/basemodel/output.parquet"


TEST_FILE='/opt/tiger/dwn-opensource-verl/ToMi/data/test.txt'

TEMPERATURE=0.7
TOP_P=0.7
MAX_TOKENS=1024
DTYPE="bf16"
BATCH_SIZE=6000
TP_SIZE=4

# Qwen2.5-7B-Instruct
BASE_MODEL_PATH="hdfs://harunava/home/byte_data_seed_azure/alphaseed/daiweinan/models/Qwen2.5-7B-Instruct"

MERGED_MODEL_SAVE_PATH="/opt/tiger/dwn-opensource-verl/mbti_hub/ckpt"

python /opt/tiger/dwn-opensource-verl/mbti_hub/script/vllm_infer_lora_multi_tomi.py \
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