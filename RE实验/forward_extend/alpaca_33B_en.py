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
    return(f'''You are currently a senior relation generation expert.

Your task is to generate relation types for the subject-object pairs in a given text, following the rules: Based on subject-object pairs, ie (noisy-channel architecture, transformation model), combined with the context of the given text, generate type of relations.

The following is an example of a chain of thought to help you think step by step
Text: "{input_list[cot_example[0]]}" Subject-object pair: [({cot_example[2]}, {cot_example[3]})]
Output: [{cot_example[1]}]''')


def get_forward_prompt(example, gold):
    return(f'''Text: "{example}" Subject-object pair: [({gold[2]}, {gold[3]})]''')


def run(input_list, train_input_list, golds, cot_examples, mode):
    for idx, gold in tqdm(enumerate(golds), total=len(golds), desc="Processing..."):
        example = input_list[gold[0]]
        cur_cot = cot_examples[gold[0] // 90]
        msg_list=[
            {"role": "system", "message": get_sys_prompt(train_input_list, cur_cot)},
            {"role": "user", "message": get_forward_prompt(example, gold)},
        ]

        rsp_1 = call_api(msg_list, 0.3)
        rsp_2 = call_api(msg_list, 0.6)
        rsp_3 = call_api(msg_list, 1)
        forward_details_path = f'./results/extend_forward_{mode}.txt'
        with open(forward_details_path, 'a', encoding='utf-8') as f:
            f.write(str(idx+1) + "：\n" + rsp_1 + "\n" + rsp_2 + "\n" + rsp_3 + "\n" 
                    + "gold:" + str(gold[1]) + "\n\n")
                


with open('./data/scierc/input_list.txt','r',encoding='utf-8') as f:
    input_list = [item.strip('\n') for item in f.readlines()]

with open('./data/scierc/train_input_list.txt','r',encoding='utf-8') as f:
    train_input_list = [item.strip('\n') for item in f.readlines()]

with open(f'./data/scierc/golds.txt','r',encoding='utf-8') as f:
    golds = [eval(line) for line in f]

with open(f'./data/scierc/cot_examples.txt','r',encoding='utf-8') as f:
    cot_examples = [eval(line) for line in f]
    

run(input_list, train_input_list, golds, cot_examples, 'alpaca_33B_en')