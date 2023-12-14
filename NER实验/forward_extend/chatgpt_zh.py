import openai
from tqdm import tqdm
import time


openai.api_key = 'api_key'

with open('../CMeEE-V2/processed_data/train_input_list.txt','r',encoding='utf-8') as f:
    input_list = f.readlines()

input_list = [item.strip('\n') for item in input_list]

golds = []
with open(f'./data/cmeee/golds.txt','r',encoding='utf-8') as f:
    for line in f:
        line = eval(line)
        golds.append(line)

merged_golds = []
with open('./data/cmeee/merged_golds.txt','r',encoding='utf-8') as f:
    for line in f:
        line = eval(line)
        merged_golds.append(line)


def get_sys_prompt():
    return(f'''当前你是一个资深的实体类型生成专家。
           
你的任务是对给定文本中的实体span生成实体类型，在生成时遵守以下的规定：
1.基于实体span，即（缺血性卒中），联合给定文本的上下文，生成可能存在的实体类型。
2.当给定一组实体span时，则依序对应生成一组实体类型。

以下是思维链的例子来帮助你一步一步的思考
Input : “脑炎@应在咨询血液学专家后进行血浆置换。脑炎@通常隔日进行 4-5 次血浆置换。” 实体span：[（脑炎），（血浆置换）]
Output : [疾病，治疗手段]''')


def get_forward_prompt(example, span_list):
    return(f'''Input : “{example}” 实体span：[{span_list}]''')


def call_chatgpt(span_list,example,p):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        top_p=p,
        messages=[
            {"role": "system", "content": get_sys_prompt()},
            {"role": "user", "content": get_forward_prompt(example, span_list)},
        ]
    )
    return completion.choices[0].message.content


def run(input_list, merged_golds):
    for idx, golds in tqdm(enumerate(merged_golds), total=len(merged_golds), desc="Processing..."):
        example = input_list[golds[0][0]]
        span_list = [gold[2] for gold in golds]
        
        span_list_str = ''
        for span in span_list:
            span_list_str += '（' + span + '），'
        span_list_str = span_list_str[:-1]

        gold_type_list = [gold[1] for gold in golds]

        while 1:
            try:
                rsp_1 = call_chatgpt(span_list_str, example, 0.3)
                rsp_2 = call_chatgpt(span_list_str, example, 0.6)
                rsp_3 = call_chatgpt(span_list_str, example, 1)

                forward_details_path = './results/extend_forward_zh.txt'
                with open(forward_details_path, 'a', encoding='utf-8') as f:
                    f.write(str(idx+1) + "：\n" + rsp_1 + "\n" + rsp_2 + "\n" + rsp_3 + "\n" + "gold:" + str(gold_type_list) + "\n\n")

                break
            except Exception as e:
                if 'That model is currently overloaded with other requests' in e.user_message:
                    print("resting\n")
                    time.sleep(30)

run(input_list, merged_golds)