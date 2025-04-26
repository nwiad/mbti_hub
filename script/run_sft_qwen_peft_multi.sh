#!/usr/bin/env bash
set -euxo pipefail

timestamp=$(date +"%m%d_%H%M")
nproc_per_node=8
model_name="Qwen2.5-7B-Instruct"
model_path="hdfs://harunava/home/byte_data_seed_azure/alphaseed/daiweinan/models/Qwen2.5-7B-Instruct"
project_name="dwn-verl"
exp_name="mbti-sft_${model_name}_${timestamp}"
epochnum=2
save_path="/mnt/hdfs/length_scaling/experiments/mbti/multi_profile/${exp_name}"

echo "Usage: sft_run_qwen_peft.sh" # 打印参数
echo "Parameters:"
echo "nproc_per_node: ${nproc_per_node}"
echo "model_path: ${model_path}"
echo "save_path: ${save_path}"


train_files="/opt/tiger/dwn-opensource-verl/mbti/mbti_data/train_with_token.parquet"
val_files="/opt/tiger/dwn-opensource-verl/mbti/mbti_data/test_with_token.parquet"

torchrun --standalone --nnodes=1 --nproc_per_node=${nproc_per_node} \
     -m verl.trainer.fsdp_sft_trainer_multi \
     data.train_files=${train_files} \
     data.val_files=${val_files} \
     data.prompt_key="instruction" \
     data.response_key="output" \
     data.prompt_dict_keys=null \
     data.response_dict_keys=null \
     optim.lr=1e-4 \
     data.micro_batch_size_per_gpu=4 \
     model.partial_pretrain=${model_path} \
     trainer.default_local_dir=${save_path} \
     trainer.project_name=${project_name} \
     trainer.experiment_name=${exp_name} \
     trainer.logger=['console','wandb'] \
     trainer.total_epochs=${epochnum} \
     trainer.default_hdfs_dir=null \
     model.lora_rank=8 \
     model.lora_alpha=16 \
     model.target_modules=all-linear



