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
    
    
def get_sys_prompt(input_list, cot_example):
    return(f'''当前你是一个资深的关系生成专家。你的任务是对给定文本中的主客体对生成关系类型，在生成时遵守以下的规定：基于主客体对，即（缺血性卒中，MRI），联合给定文本的上下文，生成可能存在的关系类型。

以下是思维链的例子来帮助你一步一步的思考
文本: “{input_list[cot_example[0]]}” 主客体对：[({cot_example[2]}, {cot_example[3]})]
Output: [{cot_example[1]}] ''')


def get_forward_prompt(example, gold):
    return(f'''文本: “{example}” 主客体对：[({gold[2]}, {gold[3]})]''')



def run(input_list, golds, cot_examples, mode):
    for idx, gold in tqdm(enumerate(golds), total=len(golds), desc="Processing..."):
        example = input_list[gold[0]]
        cur_cot = cot_examples[idx // 100]

        msg_list=[
            {"role": "system", "message": get_sys_prompt(input_list, cur_cot)},
            {"role": "user", "message": get_forward_prompt(example, gold)},
        ]

        rsp_1 = call_api(msg_list, 0.3)
        rsp_2 = call_api(msg_list, 0.6)
        rsp_3 = call_api(msg_list, 1)
        forward_details_path = f'./results/extend_forward_{mode}.txt'
        with open(forward_details_path, 'a', encoding='utf-8') as f:
            f.write(str(idx+1) + "：\n" + rsp_1 + "\n" + rsp_2 + "\n" + rsp_3 + "\n" 
                    + "gold:" + str(gold[1]) + "\n\n")
        
with open('./data/cmeie/input_list.txt','r',encoding='utf-8') as f:
    input_list = [item.strip('\n') for item in f.readlines()]

with open(f'./data/cmeie/golds.txt','r',encoding='utf-8') as f:
    golds = [eval(line) for line in f]

with open(f'./data/cmeie/cot_examples.txt','r',encoding='utf-8') as f:
    cot_examples = [eval(line) for line in f]

run(input_list, golds, cot_examples, 'alpaca_33B_zh')