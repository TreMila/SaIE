import openai
from tqdm import tqdm
import time

openai.api_key = 'api_key'


with open('./data/cmeie/input_list.txt','r',encoding='utf-8') as f:
    input_list = [item.strip('\n') for item in f.readlines()]

with open(f'./data/cmeie/golds.txt','r',encoding='utf-8') as f:
    golds = [eval(line) for line in f]

with open(f'./data/cmeie/cot_examples.txt','r',encoding='utf-8') as f:
    cot_examples = [eval(line) for line in f]


def get_sys_prompt(input_list, cot_example):
    return(f'''当前你是一个资深的关系生成专家。

你的任务是：根据给定的文本以及主客体对，生成可能存在的关系类型，并且只需要输出：[关系类型]

以下是思维链的例子来帮助你一步一步的思考：
文本: “{input_list[cot_example[0]]}” 主客体对：[({cot_example[2]}, {cot_example[3]})]

输出: [{cot_example[1]}] ''')


def get_forward_prompt(example, gold):
    return(f'''文本: “{example}” 主客体对：[({gold[2]}, {gold[3]})]
输出:''')


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


def run(input_list, golds, cot_examples, mode):
    for idx, gold in tqdm(enumerate(golds), total=len(golds), desc="Processing..."):
        example = input_list[gold[0]]
        cur_cot = cot_examples[idx // 100]

        while 1:
            try:
                msg_list=[
                    {"role": "system", "content": get_sys_prompt(input_list, cur_cot)},
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

run(input_list, golds, cot_examples, 'zh')