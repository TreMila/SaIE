import requests
import json
from tqdm import tqdm

URI = 'api_for_alpaca_33B'


with open('../CMeIE/processed_data/dev_input_list.txt','r',encoding='utf-8') as f:
    dev_input_list = [item.strip('\n') for item in f.readlines()]

with open('./data/cmeie/input_list.json','r',encoding='utf-8') as f:
    input_list = json.load(f)

with open('./data/cmeie/final_rel_extend_map_alpaca_33B_zh.json','r',encoding='utf-8') as f:
    rel_extend_map = json.load(f)

with open('./data/cmeie/cots.txt','r',encoding='utf-8') as f:
    cots = [eval(line) for line in f]

id2dev_input_list = {}
for idx, item in enumerate(dev_input_list):
    id2dev_input_list[idx] = item


def get_sys_prompt(cot):
    return(f'''当前你是一个资深的信息提取的专家。你的任务是给定文本和关系，提取符合关系类型的主客体对，在生成的时候遵守以下的规定。

1. 基于给定的关系，联合给定文本的上下文，提取可能存在给定关系的主客体对
2. 基于关系列表和关键词，生成判断句，并检测每一个判断句是否正确，仅输出“是”或“否”
3. 根据判断句生成关系列表，（主体，关系，客体），其中关系必须是给定关系

以下是思维链的例子来帮助你一步一步的思考，从而解决上面的问题
文本: “{id2dev_input_list[cot[0]]}” 关系类型:[{cot[1]}]

主客体对: [({cot[2]}, {cot[3]})]
判断:{cot[2]}的{cot[1]}是{cot[3]}吗？回答：是
因此，生成关系列表：
``
({cot[1]}, {cot[2]},{cot[3]})
``''')


def get_backward_prompt(example, extend_rel):
    return(f'''文本: “{example}” 关系类型: [{extend_rel}]''')



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


def run(input_list, rel_extend_map, cots, mode):
    for idx, item in tqdm(enumerate(input_list), total=len(input_list), desc="Processing..."):
        example = item['sentence']
        rel = item['rel_type']
        rel_extend = rel_extend_map[rel]
        cot = cots[idx]

        for r in rel_extend:
            msg_list=[
                {"role": "system", "message": get_sys_prompt(cot)},
                {"role": "user", "message": get_backward_prompt(example, r)},
            ]
            rsp_1 = call_api(msg_list, 0.3)  

            backward_details_path = f'./results/details_{mode}.txt'
            with open(backward_details_path, 'a', encoding='utf-8') as f:
                f.write(str(idx+1) + "：\nOutput:" + rsp_1 + "\n\n")

run(input_list, rel_extend_map, cots, 'alpaca_33B_zh')