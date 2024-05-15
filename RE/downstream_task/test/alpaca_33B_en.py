import requests
import json
from tqdm import tqdm

URI = 'api_for_alpaca_33B'

with open('../SCIERC/processed_data/test_input_list.txt','r',encoding='utf-8') as f:
    dev_input_list = [item.strip('\n') for item in f.readlines()]

with open('./data/scierc/input_list.json','r',encoding='utf-8') as f:
    input_list = json.load(f)

with open('./data/scierc/cots.txt','r',encoding='utf-8') as f:
    cots = [eval(line) for line in f]

with open('./data/scierc/final_rel_extend_map_alpaca_33B_en.json', 'r', encoding='utf-8') as f:
    rel_extend_map = json.load(f)


id2dev_input_list = {}
for idx, item in enumerate(dev_input_list):
    id2dev_input_list[idx] = item


def get_sys_prompt(cot):
    return(f'''You are currently a senior expert in information extraction.

Your task is to give a text and a relation, extract subject-object pairs that match the relation, and comply with the following regulations when generating. DO NOT output any Note.

1. Based on the given relation, combine the context of the given text, and extract the subject-object pairs that may have the given relation
2. Generate judgment sentences based on the relation list and keywords, and check whether each judgment sentence is correct, and only output "yes" or "no"
3. Generate a list of relations based on the judgment sentence, (subject, relation, object), where the relation must be a given relation

The following is an example of a chain of thought to help you think step by step to solve the above problems
Text: "{id2dev_input_list[cot[0]]}" relation: [{cot[1]}]

Output:
subject-object pair: [({cot[2]}, {cot[3]})]
ask: Is the relation between "{cot[2]}" and "{cot[3]}" the "{cot[1]}"? answer: Yes

Generate a list of relations:
``
({cot[2]}, {cot[1]}, {cot[3]})
``''')


def get_backward_prompt(example, extend_rel):
    return(f'''Text: "{example}" relation: [{extend_rel}]
Output:''')


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

        backward_details_path = f'./results/details_{mode}.txt'

        if not rel_extend:
            with open(backward_details_path, 'a', encoding='utf-8') as f:
                f.write(str(idx+1) + "：\nOutput: None\n\n")
            continue
        else:      
            for r in rel_extend:
                msg_list=[
                    {"role": "system", "message": get_sys_prompt(cot)},
                    {"role": "user", "message": get_backward_prompt(example, r)},
                ]
                rsp_1 = call_api(msg_list, 0.3)  
                
                with open(backward_details_path, 'a', encoding='utf-8') as f:
                    f.write(str(idx+1) + "：\nOutput:" + rsp_1 + "\n\n")
            


run(input_list, rel_extend_map, cots, 'alpaca_33B_en')
