import pandas as pd
import os
import json
import math
import re
from collections import Counter
from openai import OpenAI

import argparse
import time

API_KEY = "sk-92a191ff740d4771953c4049083d6720"
URL = "https://api.deepseek.com"


TEMPLATE_DS = """
   Please determine whether the meaning of text is closer to option A or option B. 
   [text]: {text}
   [option A]: {option_a}
   [option B]: {option_b}
   Return directly to the specific option A or B. Note the answer should be in the format of 'A' or 'B'.""" 

def get_response(prompt, client):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            # {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )

    return response.choices[0].message.content

def find_opposite_type(type):
    if type == "E":
        return "I"
    elif type == "I":
        return "E"
    elif type == "S":
        return "N"
    elif type == "N":
        return "S"
    elif type == "T":
        return "F"
    elif type == "F":
        return "T"
    elif type == "J":
        return "P"
    elif type == "P":
        return "J"

def search_letter(text, optiona, optionb, client):
    prompt_ds = TEMPLATE_DS.format(
        text=text,
        option_a=optiona,
        option_b=optionb
    )
    res = get_response(prompt_ds, client)
    print(f"prompt: {prompt_ds}")
    print(f"response: {res}")

    if res[0] == "A" or res[0] == "B":
        return res[0]
    elif res[-1] == "A" or res[-1] == "B":
        return res[-1]
    else:
        return res.strip()
    

def extract_letter(text, optiona, optionb):
    # x是一个字符串
    # 希望从字符串里面找到字母A或B 格式是[]

    print(f"{text.lower()=} {optiona.lower()=} {optionb.lower()=}")

    match = re.search(r'\[(.*?)\]', text)
    if match:
        return match.group(1)
    else:
        text_lower = text.lower()
        if optiona.lower() in text_lower:
            return 'A'
        elif optionb[:-1].lower() in text_lower:
            return 'B'
        else:
            option_a_words = re.findall(r'[a-zA-Z]+', optiona)
            option_b_words = re.findall(r'[a-zA-Z]+', optionb)
            num_hit_a = sum([1 for word in option_a_words if word.lower() in text_lower])
            num_hit_b = sum([1 for word in option_b_words if word.lower() in text_lower])
            return 'A' if num_hit_a > num_hit_b else 'B'
    print("没有找到")
    return None


def read_json(file_path):
    """
    读取json文件
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

class MBTITest:
    def __init__(self):
        # 定义各维度对应的题目ID
        self.J_P_IDs = [1,4,8,11,15,18,22,25,30,35,42,52,57,63,74,76,79,82,85,88,91,93]
        self.S_N_IDs = [2,5,9,12,16,19,23,27,29,31,33,38,40,43,45,49,51,53,55,60,62,64,66,70,81,87]
        self.E_I_IDs = [3,6,10,13,17,20,24,26,36,41,47,58,68,72,75,77,80,83,86,89,92]
        self.F_T_IDs = [7,14,21,28,32,34,37,39,44,46,48,51,54,56,59,61,65,67,69,71,73,78,84,90]

        self.label_data = read_json('../test_dataset/mbti_Q_144.json')
    
    def determine_type(self, score_a, score_b, type_a, type_b):
        """
        根据得分确定类型
        """
        # print("Score A: ", score_a)
        # print("Score B: ", score_b)
        return type_a if score_a >= score_b else type_b
    
    def get_mbti_result(self, target_type, prompt_type, answers):
        """
        计算MBTI测试结果
        params:
            target_type: str, 目标类型
            prompt_type: str, 提示类型
            answers: list, 每个元素为字符串'A'或'B', 长度为93
        return: 
            dict, 包含原始答案、各维度得分和MBTI类型
        """
        # 初始化计数器
        mbti_counter = Counter()
        # 初始化 避免不存在改类型
        mbti_counter['E'] = 0
        mbti_counter['I'] = 0
        mbti_counter['S'] = 0
        mbti_counter['N'] = 0
        mbti_counter['F'] = 0
        mbti_counter['T'] = 0
        mbti_counter['J'] = 0
        mbti_counter['P'] = 0

        # 遍历答案与题目
        for answer, q in zip(answers, self.label_data):
            if answer == 'A':
                mbti_counter[q["option_a_type"]] += 1
            elif answer == 'B':
                mbti_counter[q["option_b_type"]] += 1
            else:
                print(f"未知答案类型: {answer}")

        print(mbti_counter)

        num_TF = 39

        # 计算每个维度的得分
        mbti_score = {}
        for key in mbti_counter.keys():
            if key in ['T', 'F']:
                mbti_score[key] = mbti_counter[key] / num_TF
            else:
                mbti_score[key] = mbti_counter[key] / 35

        print(mbti_score)
        # target_type是四个字母
        # 计算target_type_score
        # 找到对应字母的得分
        target_type_score_plus = (mbti_score[target_type[0]] + mbti_score[target_type[1]] + mbti_score[target_type[2]] + mbti_score[target_type[3]])/4
        
        target_type_score_multiply = (mbti_score[target_type[0]] * mbti_score[target_type[1]] * mbti_score[target_type[2]] * mbti_score[target_type[3]])
        # 取负对数
        target_type_score_multiply = -math.log(target_type_score_multiply)

        print(target_type_score_plus, target_type_score_multiply)
        
        # 确定MBTI类型
        ei_type = self.determine_type(mbti_score['E'], mbti_score['I'], 'E', 'I')
        sn_type = self.determine_type(mbti_score['S'], mbti_score['N'], 'S', 'N')
        ft_type = self.determine_type(mbti_score['F'], mbti_score['T'], 'F', 'T')
        jp_type = self.determine_type(mbti_score['J'], mbti_score['P'], 'J', 'P')
        
        mbti_type = ei_type + sn_type + ft_type + jp_type
        # print("MBTI Type: ", mbti_type)
        return {
            'target_type': target_type,
            'prompt_type': prompt_type,
            'true_mbti_type': mbti_type,
            'target_type_score_by_plus': target_type_score_plus,
            'target_type_score_by_multiply': target_type_score_multiply,
            'original_answers': answers,
            'original_percentage': {
                'J': mbti_score['J'],
                'P': mbti_score['P'],
                'S': mbti_score['S'],
                'N': mbti_score['N'],
                'E': mbti_score['E'],
                'I': mbti_score['I'],
                'F': mbti_score['F'],
                'T': mbti_score['T']
            },
        }

def main(args):
    prompts_results_plus = {}
    prompts_results_multiply = {}
    type_results_plus = {}
    type_results_multiply = {}

    model_name = "Qwen2.5-7B-Instruct"
    train_type = args.train_type
    
    file_folder = f"../output/mbti_Q_144_nooption_bf16/{train_type}/Qwen2.5-7B-Instruct/nook"
    output_folder = f"../scores/mbti_Q_144_nooption_bf16/{train_type}/Qwen2.5-7B-Instruct"
    label_data = read_json('../test_dataset/mbti_Q_144.json')
    

    for file in os.listdir(file_folder):
        df = pd.read_parquet(os.path.join(file_folder, file))
        df.head()
        file_name = file.replace(".parquet", "")
        print(file_name)
        if train_type[0] == "p":
            # prompt类型
            target_type = file_name.split("_")[1]
        else:
            # 训练的模型
            target_type = file_name.split("_")[0]
        prompt_type = train_type


        # 获得选项A或B的回答 
        # answers = []
        # for i, row in df.iterrows():
        #     x = row["output"]
        #     if x[0] == "A" or x[0] == "B":
        #         answers.append(x[0])
        #     else:
        #         answers.append(extract_letter(x, label_data[i]["option_a"], label_data[i]["option_b"]))
        # print(answers)


        print("get answer!!!!!!!!!!")
        # 处理直接回答问题的情况
        client = OpenAI(api_key=API_KEY, base_url=URL)
        answers = []
        for i, row in df.iterrows():
            x = row["output"]
            answers.append(search_letter(x, label_data[i]["option_a"], label_data[i]["option_b"],client))
        print(f"{answers=}")

        # 计算得分
        mbti_test = MBTITest()
        score = mbti_test.get_mbti_result(target_type, prompt_type, answers)
        # print(score)

        os.makedirs(output_folder, exist_ok=True)

        with open(f"{output_folder}/mbti_score_new.jsonl", "a") as fout:
            json.dump(score, fout)
            fout.write("\n")



    # 从mbti_score_new.jsonl读取 
    # jsonl的格式 
    # {"target_type": "INTJ", 
    # "prompt_type": "all_sft", 
    # "true_mbti_type": "INTJ", 
    # "target_type_score_by_plus": 0.6459706959706959, 
    # "target_type_score_by_multiply": 1.7486991205130713, 
    # "original_answers": ["B", "B", "B", "A", "A", "B", "A", "B", "B", "A", "A", "A", "A", "A", "B", "B", "B", "A", "A", "A", "B", "A", "B", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B", "B", "A", "A", "B", "A", "B", "B", "A", "B", "B", "B", "B", "A", "B", "B", "A", "B", "B", "B", "B", "B", "B", "B", "A", "A", "A", "A", "A", "B", "B", "A", "B", "B", "B", "B", "B", "B", "B", "B", "B", "A", "A", "A", "A", "B", "B", "A", "A", "A", "B", "B", "A", "A", "B", "A", "A", "B", "A", "A", "B", "B", "A", "B", "A", "A", "A", "B", "B", "A", "A", "B", "A", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "A", "A", "B", "B", "A", "B", "A", "B", "A", "A", "A", "A", "B", "A", "B", "A", "B", "B", "A"], "original_percentage": {"J": 0.6571428571428571, "P": 0.34285714285714286, "S": 0.34285714285714286, "N": 0.6571428571428571, "E": 0.37142857142857144, "I": 0.6285714285714286, "F": 0.358974358974359, "T": 0.6410256410256411}}
    
    # data_new = []
    # with open(f"{output_folder}/mbti_score_new.jsonl", "r") as fout:

    # prompts_results_plus[prompt_type].append(score['target_type_score_by_plus'])
    # prompts_results_multiply[prompt_type].append(score['target_type_score_by_multiply'])
    # type_results_plus[target_type].append(score['target_type_score_by_plus'])
    # type_results_multiply[target_type].append(score['target_type_score_by_multiply'])


    with open(f"{output_folder}/mbti_score_new.jsonl", "r") as fout:

        for line in fout:
            score = json.loads(line)

            prompt_type = score['prompt_type']
            target_type = score['target_type']

            if prompts_results_plus.get(prompt_type) is None:
                prompts_results_plus[prompt_type] = []
            if prompts_results_multiply.get(prompt_type) is None:
                prompts_results_multiply[prompt_type] = []
            if type_results_plus.get(target_type) is None:
                type_results_plus[target_type] = []
            if type_results_multiply.get(target_type) is None:
                type_results_multiply[target_type] = []
                
            score_plus = score['target_type_score_by_plus']
            score_multiply = score['target_type_score_by_multiply']

            prompts_results_plus[prompt_type].append(score_plus)
            prompts_results_multiply[prompt_type].append(score_multiply)
            type_results_plus[target_type].append(score_plus)
            type_results_multiply[target_type].append(score_multiply)

    with open(f"{output_folder}/mean_result.txt", "w") as fout:
        fout.write("-----------------------PROMPT RESULTS ----------------------------\n")
        for prompt_type, results in prompts_results_plus.items():
            final_result = sum(results)/len(results)
            fout.write(f"prompts_results_plus: {prompt_type} final result: {final_result}\n")

        for prompt_type, results in prompts_results_multiply.items():
            final_result = sum(results)/len(results)
            fout.write(f"prompts_results_multiply: {prompt_type} final result: {final_result}\n")

        fout.write("-----------------------TYPE RESULTS ----------------------------\n")
        for target_type, results in type_results_plus.items():
            final_result = sum(results)/len(results)
            fout.write(f"type_results_plus: {target_type} final result: {final_result}\n")
        
        for target_type, results in type_results_multiply.items():
            final_result = sum(results)/len(results)
            fout.write(f"type_results_multiply: {target_type} final result: {final_result}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_type", type=str, default="prompt_characteristic", help="train type")

    args = parser.parse_args()

    start_time = time.time()
    main(args)
    end_time = time.time()
    print(f"main函数运行时间: {end_time - start_time} 秒")
