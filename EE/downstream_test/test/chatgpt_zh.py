import openai
import json
from tqdm import tqdm
import time

openai.api_key = 'api_key'


with open('../DuEE/processed_data/dev_input_list.txt','r',encoding='utf-8') as f:
    test_input_list = [item.strip('\n') for item in f.readlines()]

with open('./data/duee/input_list.json','r',encoding='utf-8') as f:
    input_list = json.load(f)

with open('./data/duee/event_extend_map.json','r',encoding='utf-8') as f:
    event_extend_map = json.load(f)

with open('./data/duee/cots.txt','r',encoding='utf-8') as f:
    cots = [eval(line) for line in f]


id2test_input_list = {}
for idx, item in enumerate(test_input_list):
    id2test_input_list[idx] = item


def get_sys_prompt(cot):
    return(f'''当前你是一个资深的事件触发词提取专家。

你的任务是给定文本和事件类型type，提取符合事件类型的事件触发词span，在生成的时候遵守以下的规定

1. 基于给定的事件类型type，联合给定文本的上下文，提取可能存在给定事件类型的事件触发词span
2. 基于成对的事件类型type和事件触发词span，生成判断句，并检测每一个判断句是否正确，仅输出“是”或“否”
3. 根据判断句生成事件检测列表，（事件类型type，事件触发词span），其中事件类型type必须是给定事件类型type

以下是思维链的例子来帮助你一步一步的思考，从而解决上面的问题
Input : ""{id2test_input_list[cot[0]]}" 事件类型type：[{cot[1]}]
事件触发词span: [{cot[2]}]"
answer：
事件类型“{cot[1]}”的触发词是“{cot[2]}”吗？是

生成事件检测列表:
``
（{cot[1]}, {cot[2]}）
``''')


def get_usr_prompt(example, extend_eve):
    return(f'''Input : ""{example}" 事件类型type：[{extend_eve}]"''')


def call_chatgpt(example,extend_eve,cot,p):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        top_p=p,
        messages=[
            {"role": "system", "content": get_sys_prompt(cot)},
            {"role": "user", "content": get_usr_prompt(example, extend_eve)},
        ]
    )

    return completion.choices[0].message.content


def run(input_list, extend_map, cots):
    for idx, item in tqdm(enumerate(input_list), total=len(input_list), desc="Processing..."):
        example = item['sentence']
        type = item['event_type']
        type_extend = [type] + extend_map[type]
        cot = cots[idx]

        for t in type_extend:
            while 1:
                try:
                    rsp_1 = call_chatgpt(example, t, cot, 0.3) 
                    rsp_2 = call_chatgpt(example, t, cot, 0.6)
                    rsp_3 = call_chatgpt(example, t, cot, 1)

                    details_path = './results/test_zh.txt'
                    with open(details_path, 'a', encoding='utf-8') as f:
                        f.write(str(idx+1) + "：\nOutput:" + rsp_1 + "\nOutput:" + rsp_2 + "\nOutput:" + rsp_3 + "\n\n")
                    break
                except Exception as e:
                    if 'That model is currently overloaded with other requests' in e.user_message:
                        print("resting\n")
                        time.sleep(30)

run(input_list, event_extend_map, cots)