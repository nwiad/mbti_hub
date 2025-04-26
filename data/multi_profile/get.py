import os
import json
import random
import pandas as pd

filefolder = '/yeesuanAI08/lijiaqi/paraagent/data/Machine_Mindset_MBTI_dataset/en/sft_all'

TEMP='Please act as {the_type} in mbti.\n'
MBTI_TOKEN_TEMPLATE = '<MBTI_{the_type}> '

# filefolder里面的每个parquet的格式
# {"instruction":"How do you handle uncertainty and ambiguity?",
# "output":" I tend to handle uncertainty and ambiguity by seeking clarity and understanding. I prefer to have a clear picture of the situation before making decisions or taking action. I rely on my intuition and empathy to gather information and assess the needs and perspectives of others involved. I strive to create a harmonious environment where everyone feels heard and understood. I also enjoy brainstorming and collaborating with others to generate ideas and solutions. In situations of uncertainty, I am often proactive in reaching out to others for support and guidance. Overall, my ENFJ personality type drives me to seek resolution and create a sense of stability amidst uncertainty."}


def get_data(filefolder):
    train_data = []
    test_data = []
    for file in os.listdir(filefolder):
        if file.endswith('.parquet'):
            the_type = file.split('_')[0]
            the_test_train = file.split('_')[1]
            

            file_path = os.path.join(filefolder, file)
            df = pd.read_parquet(file_path)

            for _, row in df.iterrows():
                prompt = MBTI_TOKEN_TEMPLATE.format(the_type=the_type) + row['instruction']
                sample = {
                    'instruction': prompt,
                    'output': row['output'],
                    'mbti_type': the_type
                }

                if the_test_train == 'test':
                    test_data.append(sample)
                elif the_test_train == 'train':
                    train_data.append(sample)

    return pd.DataFrame(train_data), pd.DataFrame(test_data)


def main():
    train_df, test_df = get_data(filefolder)
    train_df.to_parquet("train_with_token.parquet", index=False)
    test_df.to_parquet("test_with_token.parquet", index=False)


if __name__ == "__main__":
    main()