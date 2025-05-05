from peft import PeftModel
from transformers import AutoModelForCausalLM
import pandas as pd
import torch
from vllm import SamplingParams, LLM
from transformers import AutoTokenizer
import argparse
import time
from datetime import datetime
from sklearn.metrics import accuracy_score

import os


typelist=[
     "ENFJ",
     "ENFP",
     "ISTP", 
     "ESTP",
     "ISFJ",
     "ISFP",
     "ESFJ",
     "ESFP",
     "INTP",
     "ENTP",
     "INFJ",
     "INFP",
     "ENTJ",
     "INTJ",
     "ESTJ",
     "ISTJ"
]



words = {
    "ENTJ": "Extraversion, Intuition, Thinking, Judging",
    "ESTJ": "Extraversion, Sensing, Thinking, Judging",
    "ISTJ": "Introversion, Sensing, Thinking, Judging",
    "INTJ": "Introversion, Intuition, Thinking, Judging",
    "INTP": "Introversion, Intuition, Thinking, Perceiving",
    "ISTP": "Introversion, Sensing, Thinking, Perceiving",
    "ESTP": "Extraversion, Sensing, Thinking, Perceiving",
    "ENTP": "Extraversion, Intuition, Thinking, Perceiving",
    "ENFP": "Extraversion, Intuition, Feeling, Perceiving",
    "ESFP": "Extraversion, Sensing, Feeling, Perceiving",
    "INFP": "Introversion, Intuition, Feeling, Perceiving",
    "ISFP": "Introversion, Sensing, Feeling, Perceiving",
    "ISFJ": "Introversion, Sensing, Feeling, Judging",
    "INFJ": "Introversion, Intuition, Feeling, Judging",
    "ENFJ": "Extraversion, Intuition, Feeling, Judging",
    "ESFJ": "Extraversion, Sensing, Feeling, Judging"
}


def load_tomi_txt(file_path):
    examples = []
    with open(file_path, "r") as f:
        lines = f.read().strip().split("\n")
    
    story = []
    for line in lines:
        if line.strip() == "":
            continue
        if "\t" in line:
            q_line, answer, label = line.split("\t")
            question = q_line.split(" ", 1)[1]
            context = " ".join([s.split(" ", 1)[1] for s in story])
            examples.append({
                "context": context,
                "question": question,
                "answer": answer.strip(),
                "label": int(label.strip())  # 可用于true/false belief分析
            })
            story = []
        else:
            story.append(line)
    return examples

def build_prompt(context, question):
    return f"Story:\n{context}\n\nQuestion: {question}\nAnswer:"

def main(args):
    print(f"Sampling params: temperature={args.temperature}, top_p={args.top_p}, max_tokens={args.max_tokens}")
    print("-"*100)

    if args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Invalid dtype: {args.dtype}")

    print(f"Floating point precision: {dtype}")
    print("-"*100)

    output_path = args.output_path
    print(f"Output path: {output_path}")
    output_dir = "/".join(output_path.split("/")[:-1])
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print("-"*100)

    test_file = args.test_file
    print(f"Test file: {test_file}")

    tp_size = args.tp_size
    batch_size = args.batch_size
    
    df = load_tomi_txt(test_file)


    # LORA！！！！
    base_model_path = args.base_model
    lora_adapter_path = args.model
    merged_model_save_path = args.merged_model_save_path


    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True)
    model = PeftModel.from_pretrained(
        base_model, 
        lora_adapter_path, 
        trust_remote_code=True)

    merged_model = model.merge_and_unload()

    # 保存合并后的模型
    merged_model.save_pretrained(merged_model_save_path, safe_serialization=True)

    # ✅ 必须显式保存 tokenizer！
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    llm = LLM(
        model=merged_model_save_path,
        dtype=dtype,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=0.5,
        trust_remote_code=True
    )

    print(f"Model loaded from {model} using {tp_size} GPUs")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=42
    )

    for target_type in typelist:
        print("Processing type:", target_type)
        print("-"*100)
        timestamp=time.strftime("%m%d_%H%M")
        output_path=f"{output_dir}/multi_{target_type}_{timestamp}.parquet"
        print(f"Output path: {output_path}")
        print("-"*100)

        # msg_tmp =f"Please act as {target_type} in mbti.\n"
        msg_tmp = f"Please act as below personalities: \n {words[target_type]}\n"

        output_list = []
        test_prompt_list = []
        for i in range(0, len(df), batch_size):
            # print(f"Processing [{i * batch_size + min((i+1) * batch_size, len(df))}/{len(df)}]")
            print(f"Processing [{min(i + batch_size, len(df))}/{len(df)}]")

            batch = df[i:i+batch_size]
            prompts = [build_prompt(e["context"], e["question"]) for e in batch]
            # prompts = batch["prompt"].tolist()

            prompts_with_chat_template = [tokenizer.apply_chat_template(
                [{"role": "user", "content": msg_tmp + prompt}], 
                tokenize=False, 
                add_generation_prompt=True) for prompt in prompts]
            outputs = llm.generate(prompts_with_chat_template, sampling_params)
            test_prompt_list.extend(prompt_with_chat_template for prompt_with_chat_template in prompts_with_chat_template)
            output_list.extend([output.outputs[0].text for output in outputs])
        df['test_prompt'] = test_prompt_list
        df["output"] = output_list

        gold = [e["answer"].lower() for e in df]
        correct = [int(p == g) for p, g in zip(output_list, gold)]

        accuracy = sum(correct) / len(correct)
        print(f"now MBTI type is {target_type}")
        print(f"ToMi QA Accuracy: {accuracy:.2%}")

        df.to_parquet(output_path)
        

if __name__ == "__main__":
    # parse args: --model, --test_file, --output_path, --temperature, --top_p, --max_tokens, --dtype, --batch_size, --tp_size
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--tp_size", type=int, default=4)
    parser.add_argument("--merged_model_save_path", type=str, required=True)


    args = parser.parse_args()

    start_time = time.time()
    main(args)
    end_time = time.time()
    print(f"main函数运行时间: {end_time - start_time} 秒")