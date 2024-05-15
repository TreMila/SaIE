import requests
import json
from tqdm import tqdm

URI = 'api_for_baichuan2_13B'

with open('../CMeEE-V2/processed_data/dev_input_list.txt','r',encoding='utf-8') as f:
    dev_input_list = [item.strip('\n') for item in f.readlines()]

with open('./data/cmeee/input_list.json','r',encoding='utf-8') as f:
    input_list = json.load(f)

with open('./data/cmeee/final_entity_extend_map_baichuan2_13B_zh.json','r',encoding='utf-8') as f:
    ent_extend_map = json.load(f)

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
判断：{cot[2]}是{entity_type_dict[cot[1]]}吗？回答：是
因此，生成实体列表：
``
({entity_type_dict[cot[1]]}, {cot[2]})
``''')


def get_backward_prompt(example, extend_ent):
    return(f'''文本: {example} 实体类型：[{extend_ent}]''')


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


def run(input_list, ent_extend_map, cots,mode):
    for idx, item in tqdm(enumerate(input_list), total=len(input_list), desc="Processing..."):
        example = item['sentence']
        type = item['ent_type']
        type_extend = ent_extend_map[entity_type_dict[type]]
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

run(input_list, ent_extend_map, cots,'baichuan2_13B_zh')