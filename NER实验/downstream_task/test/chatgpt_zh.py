import openai
import json
from tqdm import tqdm
import time

openai.api_key = 'api_key'

with open('../CMeEE-V2/processed_data/dev_input_list.txt','r',encoding='utf-8') as f:
    test_input_list = [item.strip('\n') for item in f.readlines()]

with open('./data/cmeee/input_list.json','r',encoding='utf-8') as f:
    input_list = json.load(f)

with open('./data/cmeee/ent_extend_map.json','r',encoding='utf-8') as f:
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


id2test_input_list = {}
for idx, item in enumerate(test_input_list):
    id2test_input_list[idx] = item


def get_sys_prompt(cot):
    return(f'''当前你是一个资深的实体提取的专家。

你的任务是给定文本和实体类型type，提取符合实体类型的实体span，在生成的时候遵守以下的规定

1. 基于给定的实体类型type，联合给定文本的上下文，提取可能存在给定实体类型的实体span
2. 基于成对的实体类型type和实体span，生成判断句，并检测每一个判断句是否正确，仅输出“是”或“否”
3. 根据判断句生成实体列表，（实体类型type，实体span），其中实体类型type必须是给定实体类型type

以下是思维链的例子来帮助你一步一步的思考，从而解决上面的问题
Input : {id2test_input_list[cot[0]]}entity type：[{entity_type_dict[cot[1]]}]
实体span: [（{cot[2]}）]
answer：
{cot[2]}是{entity_type_dict[cot[1]]}吗？是

生成实体列表：
``
（{entity_type_dict[cot[1]]}, {cot[2]}）
``''')


def get_usr_prompt(example, extend_ent):
    return(f'''Input : {example} entity type：[{extend_ent}]''')


def call_chatgpt(example,extend_ent,cot,p):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        top_p=p,
        messages=[
            {"role": "system", "content": get_sys_prompt(cot)},
            {"role": "user", "content": get_usr_prompt(example, extend_ent)},
        ]
    )

    return completion.choices[0].message.content


def run(input_list, extend_map, cots):
    for idx, item in tqdm(enumerate(input_list), total=len(input_list), desc="Processing..."):
        example = item['sentence']
        type = item['ent_type']
        type_extend = [type] + extend_map[entity_type_dict[type]]
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

run(input_list, ent_extend_map, cots)