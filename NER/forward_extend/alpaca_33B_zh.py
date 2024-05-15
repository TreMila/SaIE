import requests
from tqdm import tqdm

URI = 'api_for_alpaca_33B'


def call_api(prompt,p):
    request = {
        'messages': prompt,
        'max_tokens': 100,
        'top_p': p
    }

    response = requests.post(URI, json=request)

    if response.status_code == 200:
        result = response.json()['choices']
        return result[-1]['message']['content']
    
    
def get_sys_prompt():
    return(f'''当前你是一个资深的实体类型生成专家。
           
你的任务是对给定文本中的实体span生成实体类型，在生成时遵守以下的规定：
1.基于实体span，即（缺血性卒中），联合给定文本的上下文，生成可能存在的实体类型。
2.当给定一组实体span时，则依序对应生成一组实体类型。

以下是思维链的例子来帮助你一步一步的思考
文本 : “脑炎@应在咨询血液学专家后进行血浆置换。脑炎@通常隔日进行 4-5 次血浆置换。” 实体span：[（脑炎），（血浆置换）]
Output : [疾病，治疗手段]''')


def get_forward_prompt(example, span_list):
    return(f'''文本: “{example}” 实体span：[{span_list}]''')


def run(input_list, merged_golds, mode):
    for idx, golds in tqdm(enumerate(merged_golds), total=len(merged_golds), desc="Processing..."):
        example = input_list[golds[0][0]]
        span_list = [gold[2] for gold in golds]
        
        span_list_str = ''
        for span in span_list:
            span_list_str += '（' + span + '），'

        gold_type_list = [gold[1] for gold in golds]
        
        msg_list=[
            {"role": "system", "message": get_sys_prompt()},
            {"role": "user", "message": get_forward_prompt(example, span_list_str)},
        ]
        
        rsp_1 = call_api(msg_list, 0.3)
        rsp_2 = call_api(msg_list, 0.6)
        rsp_3 = call_api(msg_list, 1)

        forward_details_path = f'./results/extend_forward_{mode}.txt'
        with open(forward_details_path, 'a', encoding='utf-8') as f:
            f.write(str(idx+1) + "：\n" + str(rsp_1) + "\n" + str(rsp_2) + "\n" + str(rsp_3) + "\n" 
                    + "gold:" + str(gold_type_list) + "\n\n")
            

with open('../CMeEE-V2/processed_data/train_input_list.txt','r',encoding='utf-8') as f:
    input_list = [item.strip('\n') for item in f.readlines()]

with open(f'./data/cmeee/golds.txt','r',encoding='utf-8') as f:
    golds = [eval(line) for line in f]

with open('./data/cmeee/merged_golds.txt','r',encoding='utf-8') as f:
    merged_golds = [eval(line) for line in f]
    
run(input_list, merged_golds, 'alpaca_33B_zh')