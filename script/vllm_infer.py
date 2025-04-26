import pandas as pd
import torch
from vllm import SamplingParams, LLM
from transformers import AutoTokenizer
import argparse
import time
from datetime import datetime



des = {
    "ENTJ": """People with ENTJ preferences are often competitive, focused, and highly motivated. They see just about everything that is going on around them by focusing on the big picture. ENTJs thrive when they have opportunities to set long-term goals and make highly analytical decisions. As a result, they often do well in high-stress leadership roles.\nENTJs tend to see things in a straightforward “black and white” way. In personal relationships they are usually fair, measured, and supportive of others.""",
    "ESTJ": """People with ESTJ preferences are logical, organized, and results driven. They love managing projects and teams. They tend to be highly structured and dependable, even in their personal lives.\nESTJs are great at networking, organizing the right people for a job, and making tough decisions confidently and tactfully. It’s no coincidence that more global leaders have ESTJ preferences than any of the other 15 personality types.""",
    "ISTJ": """People with ISTJ preferences rely on past experiences when making decisions. Decisive, focused, and efficient, they are interested in absorbing information that will improve their work.\nISTJs like strengthening their current relationships rather than seeking new ones. They tend to “play the hits” in their lives, finding joy and comfort in the things they know they like.""",
    "INTJ": """People with INTJ preferences tend to be forward-thinking and future focused. They’re often visionaries with large, far-reaching goals—but they operate mostly under the radar.\nINTJs will often spend time pondering how to get where they want to be, but they’ll rarely share their plans with others. They never shy away from a challenge and they often look for solutions that reach far beyond the original parameters.\nPeople with INTJ preferences tend to see life as a complex system. Work, family, and relationships are all puzzles for them to examine and figure out.""",
    "INTP": """People with INTP preferences are quiet, thoughtful, and analytical. They tend to put a great deal of consideration into everything they do.\nINTPs are generally easygoing and genuine. They might seem to “zone out” from time to time while they consider new concepts or explore how something works. They are often happiest laying low and working hard behind the scenes. While they may seem impersonal with people they don’t know well, they like to have a close group of people they open up to.""",
    "ISTP": """People with ISTP preferences are diligent workers who genuinely enjoy becoming experts at a craft or career. They tend to be calm and levelheaded in a crisis, quickly determining what needs to be done and effectively solving the problem.\nISTPs are generally logical, kind, and tolerant. They tend to be slow to anger, but may be frustrated by drama or irrational behavior. Although not particularly sociable, ISTPs are helpful and understanding, and are always willing to lend a hand.""",
    "ESTP": """People with ESTP preferences are logical problem solvers and quick thinkers. Energetic and outgoing, inventive and resourceful, they love using common sense to find smarter ways of doing things.\nESTPs are natural risk takers. While they are dedicated to whatever they’re working on, they don’t like to be micromanaged or told what to do by others.""",
    "ENTP": """People with ENTP preferences are innovative and entrepreneurial. They're highly attuned to even the smallest details, and they're often the first to notice patterns in a system or a group of people. They tend to enjoy strategizing, problem-solving, and brainstorming new ways to complete everyday tasks.\nBecause of their resourceful nature and their ability to remain calm under pressure, others often rely on ENTPs for help when things get tough.""",
    "ENFP": """People with ENFP preferences are stimulated by new people and new challenges. They adapt to change easily and are always ready for a change of scenery.\nENFPs are creative problem solvers. They’re always bouncing from one project to another. In doing so, they’ll often find more than one way to solve a problem or complete a task.""",
    "ESFP": """People with ESFP preferences are friendly, outgoing, and enthusiastic. They make things happen by getting others excited to jump aboard.\nWhile they tend to be both practical and sensible, they try to make things fun for everyone involved. Their flexible, spontaneous approach and easygoing, go-with-the-flow mentality often make others feel at ease.""",
    "INFP": """People with INFP preferences tend to be creative problem solvers. They are often deeply thoughtful, curious, and imaginative learners. They’re incredibly motivated by their own core values—and equally curious about the values of others. In general, they strive for and value continuous personal growth.""",
    "ISFP": """People with ISFP preferences tend to be quiet, observant, and nonjudgmental. They value and appreciate differences between people and want happiness for everyone in their lives.\nISFPs are devoted helpers, but they don’t seek recognition for their efforts. They’re carefree and easygoing, and they dislike being tied down by strict routines, rules, or structures.""",
    "ISFJ": """People with ISFJ preferences have a natural drive to understand the needs of others. They use this understanding to figure out exactly how to care for the people around them.\nISFJs tend to be responsible and practical—they often value common sense. They enjoy helping others and leaving things better than they found them.""",
    "INFJ": """People with INFJ preferences are supportive companions and devoted helpers. They believe in a moral code that puts people first; as a result, they tend to focus on finding or creating harmony. They’re often looking for a deeper meaning or purpose in life. They need to see the greater good in a plan or project to really get invested in it. But once they find that, they’re innovative thinkers motivated by their vision of a better future for everyone involved.""",
    "ENFJ": """People with ENFJ preferences tend to have a great awareness of others. They thrive on harmony and conflict resolution, and they’re often happiest as part of a group. Friendly and personable, ENFJs love to encourage others, have meaningful conversations, and work toward a shared goal.""",
    "ESFJ": """People with ESFJ preferences are natural caretakers who love to help and support others. They‘re incredibly intuitive about the needs of others, and will often strive to put systems in place to meet those needs.\n ESFJs are modest and traditional, usually doing their best work under the radar. They tend to play by the rules and value groups such as family, close friends, and tight-knit coworkers."""
    }
characteristics = {
    "ISTJ": "Quiet, serious, earn successby thoroughness and dependability. Practical,matter-of-fact, realistic,and responsible.Decide logically what should be done and work toward it steadily, regardless of distractions. Take pleasure in making everything orderly and organized--their work, their home, their life. Value traditions and loyalty.",
    "ISFJ": "Quiet, friendly, responsible, and conscientious. Committed and steady in meeting their obligations.Thorough painstaking, and accurate. Loyal, considerate, notice and remember specifics about people who are important to them, concerned with how others feel. Strive to create an orderly and harmonious environment at work and at home.",
    "INFJ": "Seek meaning and connection in ideas, relationships,and material possessions. Want to understand what motivates people and are insightful about others. Conscientious and committed to their firmv alues. Develop a clear vision about how best to serve the common good. Organizedand decisive in implementing their vision.",
    "INTJ": "Have original minds and great drive for implementing their ideas and achieving their goals. Quickly see patterns inexternal events and develop long-range explanatory perspectives. When committed, organize a job and carry it through.Skeptical and independent, have high standards of competence and performance-for themselves and others.",
    "ISTP": "Tolerant and flexible. Quiet observers until a problem appears, then act quickly to find workable solutions. Analyze what makes things work and readily get through large amounts of data to isolate the core of practical problems.Interested in cause and effect, organize facts using logical principles, value efficiency",
    "ISFP": "Quiet, friendly, sensitive, and kind. Enjoy the present moment, what's going on around them. Like to have their own space and to work within their own timeframe. Loyal and committed to their values and to people who are important tothem. Dislike disagreements and conflicts, do not force their opinions or values on others.",
    "INFP": "Idealistic, loyal to their values and to people who are important to them.Want an external life that is congruent withtheir values. Curious, quick to see possibilities, can be catalysts for implementing ideas.Seek to understand people and to help them fulfill their potential. Adaptable, flexible, and accepting unless a value is threatened.",
    "INTP": "Seek to develop logical explanations for everything that interests them,Theoretical and abstract,interested more in ideas than in social interaction.Ouiet,contained flexible, and adaptable. Have unusual ability to focus in depth to solve problems in their area of interest. Skeptical, sometimes critical, always analytical.",
    "ESTP": "Flexible and tolerant, they take a pragmatic approach focused on immediate results. Theories and conceptual explanations bore them--they want to act energetically to solve theproblem. Focus on the here-and-now, spontaneous,enjoy each moment that they canbe active with others. Enioymaterial comforts and style. Learn best through doing",
    "ESFP": "Outgoing, friendly, and accepting. Exuberant lovers of life, people, and material comforts. Enjoy working with others to make things happen. Bring common sense and arealistic approach to their work and make work fun. Flexible and spontaneous, adapt readily to new people and environments. Learn best by trying a new skill with other people.",
    "ENFP": "Warmly enthusiastic and imaginative. See life as full of possibilities. Make connections between events and information very quickly, and confidently proceed based on the patterns they see.Want a lot of affirmation from others,and readily give appreciation and support. Spontaneous andflexible, often rely on their ability to improvise and their verbal fluency.",
    "ENTP": "Ouick,ingenious,stimulating, alert, and outspoken. Resourceful in solving new and challenging problems. Adapt at generating conceptual possibilities and then analyzing them strategically. Good at reading other people. Bored by routine, will seldom do the same thing the same way, apt to turn to one new interest after another.",
    "ESTJ": "Practical, realistic, matter-of-fact. Decisive, quickly move to implement decisions. Organize projects and people to get things done, focus on getting results in the most efficient way possible. Take care of routine details. Have a clear set of logical standards, systematically follow them and want others to also. Forceful in implementing their plans.",
    "ESFJ": "Warmhearted, conscientious, and cooperative. Want harmony in their environment, work with determination to establish it. Like to work with others to complete tasks accurately and on time. Loyal follow through even in small matters.Notice what other need in their day-by-day lives and try to provide it. Want to be appreciated for who they are and for what they contribute.",
    "ENFJ": "Warm,empathetic, responsive,and responsible. Highly attuned to the emotions.needs,and motivations of others. Find potential in everyone, want to help others fulfill their potential. May act as catalystsfor individual and group growth. Loyal, responsive to praise and criticism. Sociable.facilitate others in a group. and provide inspiring leadership.",
    "ENTJ": "Frank, decisive, assume leadership readily. Quickly see illogical and inefficient procedures and policies, develop and implement comprehensive systems to solve organizational problems. Enjoy long-term planning and goal setting. Usually well informed, well read,enjoy expanding their knowledge and passing it on to others.Forceful in presenting their ideas."
}

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
    print("-"*100)

    model = args.model
    test_file = args.test_file
    tp_size = args.tp_size
    batch_size = args.batch_size
    prompt_type = args.prompt_type
    
    df = pd.read_parquet(test_file)

    tokenizer = AutoTokenizer.from_pretrained(
        model, 
        trust_remote_code=True)

    
    llm = LLM(
        model=model,
        dtype=dtype,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=0.7,
        trust_remote_code=True
    )

    print(f"Model loaded from {model} using {tp_size} GPUs")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=42
    )
    target_type = args.target_type
    
    if prompt_type == "simple":
        msg_tmp =f"Please act as {target_type} in mbti.\n"
    elif prompt_type == "decs":
        msg_tmp = f"Please act as {target_type} in mbti.\n Below is the type's description: {des[target_type]}.\n"
    elif prompt_type == "characteristics":
        msg_tmp = f"Please act as below characteristics: \n {characteristics[target_type]}\n"
    else:
        msg_tmp = ""

    output_list = []
    for i in range(0, len(df), batch_size):
        # print(f"Processing [{i * batch_size + min((i+1) * batch_size, len(df))}/{len(df)}]")
        print(f"Processing [{min(i + batch_size, len(df))}/{len(df)}]")

        batch = df.iloc[i:i+batch_size]
        prompts = batch["instruction"].tolist()


        prompts_with_chat_template = [tokenizer.apply_chat_template(
            [{"role": "user", "content": msg_tmp + prompt}],  
            tokenize=False, 
            add_generation_prompt=True) for prompt in prompts]
        outputs = llm.generate(prompts_with_chat_template, sampling_params)
        
        output_list.extend([output.outputs[0].text for output in outputs])
    df["output"] = output_list

    df.to_parquet(output_path)
    

if __name__ == "__main__":
    # parse args: --model, --test_file, --output_path, --temperature, --top_p, --max_tokens, --dtype, --batch_size, --tp_size
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--tp_size", type=int, default=4)
    parser.add_argument("--target_type", type=str, default="ENTP")
    parser.add_argument("--prompt_type", type=str, default="original")

    args = parser.parse_args()

    start_time = time.time()
    main(args)
    end_time = time.time()
    print(f"main函数运行时间: {end_time - start_time} 秒")
