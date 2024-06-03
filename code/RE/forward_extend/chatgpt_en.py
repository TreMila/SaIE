import openai
from tqdm import tqdm
import time

openai.api_key = 'api_key'


with open('./data/scierc/input_list.txt','r',encoding='utf-8') as f:
    input_list = [item.strip('\n') for item in f.readlines()]

with open('./data/scierc/train_input_list.txt','r',encoding='utf-8') as f:
    train_input_list = [item.strip('\n') for item in f.readlines()]

with open(f'./data/scierc/golds.txt','r',encoding='utf-8') as f:
    golds = [eval(line) for line in f]

with open(f'./data/scierc/cot_examples.txt','r',encoding='utf-8') as f:
    cot_examples = [eval(line) for line in f]


def get_sys_prompt(input_list, cot_example):
    return(f'''You are currently a senior relation generation expert.

Your task is to generate just one kind of relation types for the subject-object pairs in a given text. DO NOT give any explanation.
The following is an example of a chain of thought to help you think step by step
Text: "{input_list[cot_example[0]]}" Subject-object pair: [({cot_example[2]}, {cot_example[3]})]
Output: [{cot_example[1]}]''')


def get_forward_prompt(example, gold):
    return(f'''Text: "{example}" Subject-object pair: [({gold[2]}, {gold[3]})]
Output:''')


def call_chatgpt(train_input,gold,example,cot,p):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        top_p=p,
        messages=[
            {"role": "system", "content": get_sys_prompt(train_input, cot)},
            {"role": "user", "content": get_forward_prompt(example, gold)},
        ]
    )
    return completion.choices[0].message.content


def run(input_list, train_input_list, golds, cot_examples, mode):
    for idx, gold in tqdm(enumerate(golds), total=len(golds), desc="Processing..."):
        example = input_list[gold[0]]
        cur_cot = cot_examples[gold[0] // 90]

        while 1:
            try:
                msg_list=[
                    {"role": "system", "content": get_sys_prompt(train_input_list, cur_cot)},
                    {"role": "user", "content": get_forward_prompt(example, gold)},
                ]
                rsp_1 = call_chatgpt(msg_list,0.3) 
                rsp_2 = call_chatgpt(msg_list,0.6)
                rsp_3 = call_chatgpt(msg_list,1)

                forward_details_path = f'./results/extend_forward_{mode}.txt'
                with open(forward_details_path, 'a', encoding='utf-8') as f:
                    f.write(str(idx+1) + "：\n" + rsp_1 + "\n" + rsp_2 + "\n" + rsp_3 + "\n" 
                            + "gold:" + str(gold[1]) + "\n\n")

                break
            except Exception as e:
                if 'That model is currently overloaded with other requests' in e.user_message:
                    print("resting\n")
                    time.sleep(30)

run(input_list, train_input_list, golds, cot_examples, 'en')