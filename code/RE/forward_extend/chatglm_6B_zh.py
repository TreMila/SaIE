import requests
from tqdm import tqdm

URI = 'api_for_chatglm_6B'

def call_api(prompt, history, p):
    request = {
        'prompt': prompt,
        'history': history,
        'max_length': 2000,
        'temperature': 0.2,
        'top_p': p
    }

    response = requests.post(URI, json=request)

    if response.status_code == 200:
        result = response.json()['response']
        return result


def get_sys_prompt(input_list, cot_example):
    return(f'''当前你是一个资深的关系生成专家。

你的任务是：根据给定的文本以及主客体对，生成可能存在的关系类型，并且只需要输出：[关系类型]

以下是思维链的例子来帮助你一步一步的思考：
文本: “{input_list[cot_example[0]]}” 主客体对：[({cot_example[2]}, {cot_example[3]})]

输出: [{cot_example[1]}] ''')


def get_forward_prompt(example, gold):
    return(f'''文本: “{example}” 主客体对：[({gold[2]}, {gold[3]})]
输出:''')



def run(input_list, golds, cot_examples, mode):
    for idx, gold in tqdm(enumerate(golds), total=len(golds), desc="Processing..."):
        example = input_list[gold[0]]
        cur_cot = cot_examples[idx // 100]


        prompt = get_sys_prompt(input_list, cur_cot) + get_forward_prompt(example, gold)
        history = []
        rsp_1 = call_api(prompt, history, 0.3) 
        rsp_2 = call_api(prompt, history, 0.6)
        rsp_3 = call_api(prompt, history, 1)
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

run(input_list, golds, cot_examples, 'chatglm_6B_zh')