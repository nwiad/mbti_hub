
import os
import pandas as pd

file_foler = '/opt/tiger/dwn-opensource-verl/mbti_hub/output_final/Qwen2.5-7B-Instruct/tomi/multi_profile/words_1000_1024'

for file in os.listdir(file_foler):
    if file.endswith('.parquet'):
        file_path = os.path.join(file_foler, file)
        type = file.split('_')[1]
        df = pd.read_parquet(file_path)

        correct = [int(g.lower() in p.lower()) for p, g in zip(df['output'], df['answer'])]
        accuracy = sum(correct) / len(correct)
        print(f"now MBTI type is {type}")
        print(f"ToMi QA Accuracy: {accuracy:.2%}")