import requests
import json
from tqdm import tqdm

URI = 'api_for_alpaca_33B'

with open('../ACE05/processed_data/test_input_list.txt','r',encoding='utf-8') as f:
    dev_input_list = [item.strip('\n') for item in f.readlines()]

with open('./data/ace05/input_list.json','r',encoding='utf-8') as f:
    input_list = json.load(f)

with open('./data/ace05/cots.txt','r',encoding='utf-8') as f:
    cots = [eval(line) for line in f]

with open('./data/ace05/final_entity_extend_map_alpaca_33B_en.json', 'r', encoding='utf-8') as f:
    ent_extend_map = json.load(f)


id2dev_input_list = {}
for idx, item in enumerate(dev_input_list):
    id2dev_input_list[idx] = item

def get_sys_prompt(cot):
    return(f'''You are an experienced senior expert in entity extraction in the field of IT.

Your current task is to extract the specific entity span that corresponds to the given entity types and context. DO NOT output any Note. Follow the below rules when generating:

1. Extract entity spans that may correspond to the given entity type by combining the context of the given text.
2. Generate judgment sentences based on paired entity types and entity spans. Evaluate the correctness of each judgment sentence and output only 'yes' or 'no'.
3. Generate an entity list based on the judgment sentences, following the format (entity type, entity span). Ensure that the entity type matches the given entity type.

Here is an example thought process to guide you in solving the above problems step by step:
Text : "{id2dev_input_list[cot[0]]}" Entity Type: [{cot[1]}]

Output:
Entity Span: [{cot[2]}]
ask: Is '{cot[2]}' a {cot[1]}? answer: yes
Generate an entity list:
``
({cot[1]}, {cot[2]})
``''')


def get_backward_prompt(example, extend_ent):
    return(f'''Text : "{example}" Entity type: [{extend_ent}]
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


def run(input_list, ent_extend_map, cots, mode):
    for idx, item in tqdm(enumerate(input_list), total=len(input_list), desc="Processing..."):
        example = item['sentence']
        ent = item['ent_type']
        ent_extend = ent_extend_map[ent]
        cot = cots[idx]

        backward_details_path = f'./results/details_{mode}.txt'

        if not ent_extend:
            with open(backward_details_path, 'a', encoding='utf-8') as f:
                f.write(str(idx+1) + "：\nOutput: None\n\n")
            continue
        else:      
            for r in ent_extend:
                msg_list=[
                    {"role": "system", "message": get_sys_prompt(cot)},
                    {"role": "user", "message": get_backward_prompt(example, r)},
                ]
                rsp_1 = call_api(msg_list, 0.3)  
                
                with open(backward_details_path, 'a', encoding='utf-8') as f:
                    f.write(str(idx+1) + "：\nOutput:" + rsp_1 + "\n\n")
            


run(input_list, ent_extend_map, cots, 'alpaca_33B_en')
