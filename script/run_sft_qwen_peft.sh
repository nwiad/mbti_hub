# Tested with 2 & 4 GPUs

# set -x


#!/usr/bin/env bash
set -euxo pipefail

timestamp=$(date +"%m%d_%H%M")
nproc_per_node=4
save_path="../ckpt"
model_name="Qwen2.5-7B-Instruct"
model_path="/yeesuanAI08/LLM/${model_name}"
project_name="dwn-verl"
exp_name="mbti-sft_${model_name}_${timestamp}"
epochnum=2

echo "Usage: sft_run_qwen_peft.sh" # 打印参数
echo "Parameters:"
echo "nproc_per_node: ${nproc_per_node}"
echo "model_path: ${model_path}"
echo "save_path: ${save_path}"

typelist=(
     "ENFJ"
     "ENFP"
     "ISTP"  
     "ESTP" 
     "ISFJ"
     "ISFP" 
     "ESFJ" 
     "ESFP"
     "INTP" 
     "ENTP" 
     "INFJ" 
     "INFP"
     "ENTJ" 
     "INTJ" 
     "ESTJ" 
     "ISTJ"
)

export CUDA_VISIBLE_DEVICES=0,1,2,3

for type in ${typelist[@]}; do
     echo "Processing type: ${type}"

     train_files="../data/Machine_Mindset_MBTI_dataset/en/sft_all/${type}_train_data.parquet"
     val_files="../data/Machine_Mindset_MBTI_dataset/en/sft_all/${type}_test_data.parquet"

     torchrun --standalone --nnodes=1 --nproc_per_node=${nproc_per_node} \
          -m verl.trainer.fsdp_sft_trainer \
          data.train_files=${train_files} \
          data.val_files=${val_files} \
          data.prompt_key="instruction" \
          data.response_key="output" \
          +data.prompt_dict_keys=null \
          +data.response_dict_keys=null \
          optim.lr=1e-4 \
          data.micro_batch_size_per_gpu=2 \
          model.partial_pretrain=${model_path} \
          trainer.default_local_dir=${save_path}/all_sft/${epochnum}epoch/${model_name}/${type} \
          trainer.project_name=${project_name} \
          trainer.experiment_name=${exp_name} \
          trainer.logger=['console'] \
          trainer.total_epochs=${epochnum} \
          trainer.default_hdfs_dir=null \
          model.lora_rank=8 \
          model.lora_alpha=16 \
          model.target_modules=all-linear
done



