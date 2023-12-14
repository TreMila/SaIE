import openai
import json
from tqdm import tqdm
import time

openai.api_key = 'api_key'


with open('../CMeIE/processed_data/dev_input_list.txt','r',encoding='utf-8') as f:
    dev_input_list = [item.strip('\n') for item in f.readlines()]

with open('./data/cmeie/input_list.json','r',encoding='utf-8') as f:
    input_list = json.load(f)

with open('./data/cmeie/rel_extend_map.json','r',encoding='utf-8') as f:
    rel_extend_map = json.load(f)

with open('./data/cmeie/cots.txt','r',encoding='utf-8') as f:
    cots = [eval(line) for line in f]


id2dev_input_list = {}
for idx, item in enumerate(dev_input_list):
    id2dev_input_list[idx] = item

def get_sys_prompt(cot):
    return(f'''当前你是一个资深的信息提取的专家。

你的任务是给定文本和关系，提取符合关系类型的主客体对，在生成的时候遵守以下的规定

1. 基于给定的关系，联合给定文本的上下文，提取可能存在给定关系的主客体对
2. 基于关系列表和关键词，生成判断句，并检测每一个判断句是否正确，仅输出“是”或“否”
3. 根据判断句生成关系列表，（主体，关系，客体），其中关系必须是给定关系

以下是思维链的例子来帮助你一步一步的思考，从而解决上面的问题
Input : {id2dev_input_list[cot[0]]} relation：[{cot[1]}]
主客体对: [（{cot[2]}, {cot[3]}）]
answer：
{cot[2]}的{cot[1]}是{cot[3]}吗？是

生成关系列表：
``
（{cot[1]}, {cot[2]}, {cot[3]}）
``''')


def get_backward_prompt(example, extend_rel):
    return(f'''Input : {example} relation：[{extend_rel}]''')


def call_chatgpt(example,extend_rel,cot,p):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        top_p=p,
        messages=[
            {"role": "system", "content": get_sys_prompt(cot)},
            {"role": "user", "content": get_backward_prompt(example, extend_rel)},
        ]
    )
    return completion.choices[0].message.content


def run(input_list, rel_extend_map, cots):
    for idx, item in tqdm(enumerate(input_list), total=len(input_list), desc="Processing..."):
        example = item['sentence']
        rel = item['rel_type']
        rel_extend = [rel] + rel_extend_map[rel]
        cot = cots[idx]

        for r in rel_extend:
            while 1:
                try:
                    rsp_1 = call_chatgpt(example, r, cot, 0.3) 
                    rsp_2 = call_chatgpt(example, r, cot, 0.6)
                    rsp_3 = call_chatgpt(example, r, cot, 1)

                    backward_details_path = './results/test_zh.txt'
                    with open(backward_details_path, 'a', encoding='utf-8') as f:
                        f.write(str(idx+1) + "：\nOutput:" + rsp_1 + "\nOutput:" + rsp_2 + "\nOutput:" + rsp_3 + "\n\n")
                    break
                except Exception as e:
                    if 'That model is currently overloaded with other requests' in e.user_message:
                        print("resting\n")
                        time.sleep(30)

run(input_list, rel_extend_map, cots)