#!/bin/bash


export CUDA_VISIBLE_DEVICES=4,5

timestamp=$(date +"%m%d_%H%M")

MODEL_NAME="Qwen2.5-7B-Instruct"
MODEL_PATH="../ckpt/multi_words_1k/global_step_62"
OUTPUT_PATH="../output_final/$MODEL_NAME/tomi/multi_profile/words_1000_1024/output.parquet"


TEST_FILE='/yeesuanAI08/lijiaqi/paraagent/data/ToMi/data/test.txt'

TEMPERATURE=0.7
TOP_P=0.7
MAX_TOKENS=1024
DTYPE="bf16"
BATCH_SIZE=144
TP_SIZE=2

# Qwen2.5-7B-Instruct
BASE_MODEL_PATH="/yeesuanAI08/LLM/$MODEL_NAME"
MERGED_MODEL_SAVE_PATH="../ckpt/use_use"
python vllm_infer_lora_multi_tomi.py \
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


TEST_FILE="../test_dataset/mbti_Q_144_v3.parquet"

OUTPUT_PATH="../output_final/$MODEL_NAME/mbti_Q_144_bf16/multi_profile/words_1000_1024/output.parquet"


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