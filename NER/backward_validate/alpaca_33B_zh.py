import requests
import json
from tqdm import tqdm

URI = 'api_for_alpaca_33B'

with open('../CMeEE-V2/processed_data/dev_input_list.txt','r',encoding='utf-8') as f:
    dev_input_list = [item.strip('\n') for item in f.readlines()]

with open('./data/cmeee/input_list.txt','r',encoding='utf-8') as f:
    input_list = [item.strip('\n') for item in f.readlines()]

with open('./data/cmeee/final_entity_extend_map_alpaca_33B_zh.json','r',encoding='utf-8') as f:
    entity_extend_map = json.load(f)

with open('./data/cmeee/cots.txt','r',encoding='utf-8') as f:
    cots = [eval(line) for line in f]

entity_type_list = ['dru', 'bod', 'pro', 'sym', 'equ', 'ite', 'dep', 'mic', 'dis']
entity_type_dict = {
            'dru':'药物',
            'bod':'身体',
            'pro':'医疗程序',
            'sym':'临床表现',
            'equ':'医疗设备',
            'ite':'医学检验项目',
            'dep':'科室',
            'mic':'微生物类',
            'dis':'疾病'
}

input_list2ent={}
for idx, item in enumerate(input_list):
    input_list2ent[item] = entity_type_list[idx//10]

id2dev_input_list = {}
for idx, item in enumerate(dev_input_list):
    id2dev_input_list[idx] = item


def get_sys_prompt(cot):
    return(f'''当前你是一个资深的实体提取的专家。

你的任务是给定文本和实体类型，提取符合实体类型的实体span，在生成的时候遵守以下的规定

1. 基于给定的实体类型，联合给定文本的上下文，提取可能存在给定实体类型的实体span
2. 基于成对的实体类型和实体span，生成判断句，并检测每一个判断句是否正确，仅输出“是”或“否”
3. 根据判断句生成实体列表，（实体类型，实体），其中实体类型必须是给定实体类型

文本 : {id2dev_input_list[cot[0]]} 实体类型：[{entity_type_dict[cot[1]]}]

实体span: [({cot[2]})]
判断：
{cot[2]}是{entity_type_dict[cot[1]]}吗？是

生成实体列表：
``
({entity_type_dict[cot[1]]}, {cot[2]})
``''')


def get_backward_prompt(example, extend_ent):
    return(f'''文本: {example} 实体类型：[{extend_ent}]''')


def call_api(prompt, p):
    request = {
        'messages': prompt,
        'max_tokens': 2000,
        'top_p': p,
    }
    response = requests.post(URI, json=request)

    if response.status_code == 200:
        result = response.json()['choices']
        return result[-1]['message']['content']


def run(input_list, input_list2ent, entity_extend_map, cots, mode):
    for idx, example in tqdm(enumerate(input_list), total=len(input_list), desc="Processing..."):
        ent = input_list2ent[example]
        ent_extend = entity_extend_map[ent]
        cot = cots[idx]
        
        for e in ent_extend:
            msg_list=[
                {"role": "system", "message": get_sys_prompt(cot)},
                {"role": "user", "message": get_backward_prompt(example, e)},
            ]
            rsp_1 = call_api(msg_list, 0.3) 
            rsp_2 = call_api(msg_list, 0.6)
            rsp_3 = call_api(msg_list, 1)

            backward_details_path = f'./results/details_{mode}.txt'
            with open(backward_details_path, 'a', encoding='utf-8') as f:
                f.write(str(idx+1) + "：\nOutput:" + rsp_1 + "\nOutput:" + rsp_2 + "\nOutput:" + rsp_3 + "\n\n")


def run_gold(input_list, input_list2ent, cots, mode):
    for idx, example in tqdm(enumerate(input_list), total=len(input_list), desc="Processing..."):
        ent = input_list2ent[example]
        cot = cots[idx]
        
        
        msg_list=[
            {"role": "system", "message": get_sys_prompt(cot)},
            {"role": "user", "message": get_backward_prompt(example, ent)},
        ]
        rsp_1 = call_api(msg_list, 0.3) 
        rsp_2 = call_api(msg_list, 0.6)
        rsp_3 = call_api(msg_list, 1)

        backward_details_path = f'./results/details_gold_{mode}.txt'
        with open(backward_details_path, 'a', encoding='utf-8') as f:
            f.write(str(idx+1) + "：\nOutput:" + rsp_1 + "\nOutput:" + rsp_2 + "\nOutput:" + rsp_3 + "\n\n")

run(input_list, input_list2ent, entity_extend_map, cots, 'alpaca_33B_zh')
run_gold(input_list, input_list2ent, cots,'alpaca_33B_zh')