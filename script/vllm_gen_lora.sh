#!/bin/bash

typelist=(
    "ENFJ" 
    "ENFP" 
    "ISTP"
)

export CUDA_VISIBLE_DEVICES=4,5,6,7

for type in ${typelist[@]}; do
    echo "Processing type: ${type}"


    timestamp=$(date +"%m%d_%H%M")
    TYPE=$type
    OUTPUT_PATH="../output/mbti_Q_144_bf16/all_sft/Qwen2.5-7B-Instruct/${TYPE}_${timestamp}.parquet"
    MODEL_PATH="../ckpt/all_sft/Qwen2.5-7B-Instruct/${TYPE}/global_step_141"
    TEST_FILE="../test_dataset/mbti_Q_144_v3.parquet"

    TEMPERATURE=0.7
    TOP_P=0.7
    MAX_TOKENS=1024
    DTYPE="bf16"
    BATCH_SIZE=4
    TP_SIZE=4

    BASE_MODEL_PATH="/yeesuanAI08/LLM/Qwen2.5-7B-Instruct"
    MERGED_MODEL_SAVE_PATH="../ckpt/use_use"
    python vllm_infer_lora.py \
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
        --type "$TYPE"
done