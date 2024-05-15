import openai
import json
from tqdm import tqdm
import time

openai.api_key = 'api_key'

with open('../ACE05/processed_data/test_input_list.txt','r',encoding='utf-8') as f:
    test_input_list = [item.strip('\n') for item in f.readlines()]

with open('./data/ace05/input_list.json','r',encoding='utf-8') as f:
    input_list = json.load(f)

with open('./data/ace05/ent_extend_map.json','r',encoding='utf-8') as f:
    ent_extend_map = json.load(f)

with open('./data/ace05/cots.txt','r',encoding='utf-8') as f:
    cots = [eval(line) for line in f]


id2test_input_list = {}
for idx, item in enumerate(test_input_list):
    id2test_input_list[idx] = item


def get_sys_prompt(cot):
    return(f'''You are an experienced senior expert in entity extraction in the field of IT.

Your current task is to extract the specific entity span that corresponds to the given entity types and  context. To accomplish this, please adhere to the following guidelines:

1. Extract entity spans that may correspond to the given entity type by combining the context of the given text.
2. Generate judgment sentences based on paired entity types and entity spans. Evaluate the correctness of each judgment sentence and output only 'yes' or 'no'.
3. Generate an entity list based on the judgment sentences, following the format (entity type, entity span). Ensure that the entity type matches the given entity type.

Here is an example thought process to guide you in solving the above problems step by step:
Input : ""{id2test_input_list[cot[0]]}" Entity Type: [{cot[1]}]"
Entity Span: [{cot[2]}]
answer：
Is '{cot[2]}' a {cot[1]}? yes

Generate an entity list:
``
({cot[1]}, {cot[2]})
``''')


def get_usr_prompt(example, extend_ent):
    return(f'''Input : ""{example}" Entity type: [{extend_ent}]"''')


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
        type_extend = [type] + extend_map[type]
        cot = cots[idx]

        for t in type_extend:
            while 1:
                try:
                    rsp_1 = call_chatgpt(example, t, cot, 0.3) 
                    rsp_2 = call_chatgpt(example, t, cot, 0.6)
                    rsp_3 = call_chatgpt(example, t, cot, 1)

                    details_path = './results/test_en.txt'
                    with open(details_path, 'a', encoding='utf-8') as f:
                        f.write(str(idx+1) + "：\nOutput:" + rsp_1 + "\nOutput:" + rsp_2 + "\nOutput:" + rsp_3 + "\n\n")
                    break
                except Exception as e:
                    if 'That model is currently overloaded with other requests' in e.user_message:
                        print("resting\n")
                        time.sleep(30)

run(input_list, ent_extend_map, cots)