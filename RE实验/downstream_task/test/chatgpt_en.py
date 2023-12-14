import openai
import json
from tqdm import tqdm
import time

openai.api_key = 'api_key'


with open('../SCIERC/processed_data/test_input_list.txt','r',encoding='utf-8') as f:
    dev_input_list = [item.strip('\n') for item in f.readlines()]

with open('./data/scierc/input_list.json','r',encoding='utf-8') as f:
    input_list = json.load(f)

with open('./data/scierc/rel_extend_map.json','r',encoding='utf-8') as f:
    rel_extend_map = json.load(f)

with open('./data/scierc/cots.txt','r',encoding='utf-8') as f:
    cots = [eval(line) for line in f]


id2dev_input_list = {}
for idx, item in enumerate(dev_input_list):
    id2dev_input_list[idx] = item


def get_sys_prompt(cot):
    return(f'''You are currently a senior expert in information extraction.

Your task is to give a text and a relation, extract subject-object pairs that match the relation, and comply with the following regulations when generating

1. Based on the given relation, combine the context of the given text, and extract the subject-object pairs that may have the given relation
2. Generate judgment sentences based on the relation list and keywords, and check whether each judgment sentence is correct, and only output "yes" or "no"
3. Generate a list of relations based on the judgment sentence, (subject, relation, object), where the relation must be a given relation

The following is an example of a chain of thought to help you think step by step to solve the above problems
Input 1 : "{id2dev_input_list[cot[0]]}" relation: [{cot[1]}]
subject-object pair: [({cot[2]}, {cot[3]})]
answer:
Is the relation between "{cot[2]}" and "{cot[3]}" the "{cot[1]}"? Yes

Generate a list of relations:
``
({cot[2]}, {cot[1]}, {cot[3]})
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

                    backward_details_path = './results/test_en.txt'
                    with open(backward_details_path, 'a', encoding='utf-8') as f:
                        f.write(str(idx+1) + "：\nOutput:" + rsp_1 + "\nOutput:" + rsp_2 + "\nOutput:" + rsp_3 + "\n\n")
                    break
                except Exception as e:
                    if 'That model is currently overloaded with other requests' in e.user_message:
                        print("resting\n")
                        time.sleep(30)

run(input_list, rel_extend_map, cots)