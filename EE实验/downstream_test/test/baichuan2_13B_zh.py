import requests
import json
from tqdm import tqdm

URI = 'api_for_baichuan2_13B'

with open('../DuEE/processed_data/dev_input_list.txt','r',encoding='utf-8') as f:
    dev_input_list = [item.strip('\n') for item in f.readlines()]

with open('./data/duee/input_list.json','r',encoding='utf-8') as f:
    input_list = json.load(f)

with open('./data/duee/final_event_extend_map_baichuan2_13B_zh.json','r',encoding='utf-8') as f:
    event_extend_map = json.load(f)

with open('./data/duee/cots.txt','r',encoding='utf-8') as f:
    cots = [eval(line) for line in f]

id2dev_input_list = {}
for idx, item in enumerate(dev_input_list):
    id2dev_input_list[idx] = item


def get_sys_prompt(cot):
    return(f'''当前你是一个资深的事件触发词提取专家。

你的任务是给定文本和事件类型，提取符合事件类型的事件触发词span，在生成的时候遵守以下的规定

1. 基于给定的事件类型，联合给定文本的上下文，提取可能存在给定事件类型的事件触发词span
2. 基于成对的事件类型和事件触发词span，生成判断句，并检测每一个判断句是否正确，仅输出“是”或“否”
3. 根据判断句生成事件检测列表，(事件类型, 事件触发词span)，其中事件类型必须是给定事件类型

以下是思维链的例子来帮助你一步一步的思考，从而解决上面的问题
文本 : "{id2dev_input_list[cot[0]]}" 事件类型: [{cot[1]}]

事件触发词span: [{cot[2]}]
判断:
事件类型“{cot[1]}”的触发词是“{cot[2]}”吗？是

生成事件检测列表:
``
({cot[1]}, {cot[2]})
``''')


def get_backward_prompt(example, extend_eve):
    return(f'''文本 : "{example}" 事件类型: [{extend_eve}]''')



def call_api(prompt, p):
    request = {
        'prompt': prompt,
        'max_new_tokens': 2000,
        'top_p': p
    }

    response = requests.post(URI, json=request)

    if response.status_code == 200:
        result = response.json()['response']
        return result


def run(input_list, event_extend_map, cots, mode):
    for idx, item in tqdm(enumerate(input_list), total=len(input_list), desc="Processing..."):
        example = item['sentence']
        type = item['event_type']
        type_extend = event_extend_map[type]
        cot = cots[idx]

        for r in type_extend:
            msg_list=[
                {"role": "system", "content": get_sys_prompt(cot)},
                {"role": "user", "content": get_backward_prompt(example, r)},
            ]
            rsp_1 = call_api(msg_list, 0.3) 

            backward_details_path = f'./results/details_{mode}.txt'
            with open(backward_details_path, 'a', encoding='utf-8') as f:
                f.write(str(idx+1) + "：\nOutput:" + rsp_1 + "\n\n")

run(input_list, event_extend_map, cots,'baichuan2_13B_zh')