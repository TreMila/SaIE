import requests
import json
from tqdm import tqdm

URI = 'api_for_baichuan2_13B'

with open('../CASIE/processed_data/test_input_list.txt','r',encoding='utf-8') as f:
    dev_input_list = [item.strip('\n') for item in f.readlines()]

with open('./data/casie/input_list.json','r',encoding='utf-8') as f:
    input_list = json.load(f)

with open('./data/casie/final_event_extend_map_baichuan2_13B_en.json','r',encoding='utf-8') as f:
    event_extend_map = json.load(f)

with open('./data/casie/cots.txt','r',encoding='utf-8') as f:
    cots = [eval(line) for line in f]


id2dev_input_list = {}
for idx, item in enumerate(dev_input_list):
    id2dev_input_list[idx] = item


def get_sys_prompt(cot):
    return(f'''You are currently an expert in extracting event-triggered words.

Your task is to extract the trigger words that match the event type given the text and event type. DO NOT output any Note. Follow the below rules when generating

1. Based on the given event type, combined with the context of the given text, extract the trigger word span that may exist in the given event type
2. Generate judgment sentences based on the paired event type and event trigger word span, and check whether each judgment sentence is correct, and only output "yes" or "no"
3. Generate an event detection list according to the judgment sentence, (event type, event trigger span), where the event type must be a given event type

The following is an example of a chain of thought to help you think step by step to solve the above problems
Text: "{id2dev_input_list[cot[0]]}" Event type: [{cot[1]}]

Output:
step1: Trigger span: [{cot[2]}]
step2: ask: Is the trigger word for event type "{cot[1]}" "{cot[2]}"? answer: yes
step3: Generate a list of event detections:
``
({cot[1]}, {cot[2]})
``''')


def get_backward_prompt(example, extend_eve):
    return(f'''Text : "{example}" Entity type: [{extend_eve}]
Output:''')


def call_api(prompt, p):
    request = {
        'prompt': prompt,
        'max_new_tokens': 2000,
        'top_p': p
    }

    response = requests.post(URI, json=request)

    if response.status_code == 200:
        result = response.json()['response']
        return result


def run(input_list, event_extend_map, cots, mode):
    for idx, item in tqdm(enumerate(input_list), total=len(input_list), desc="Processing..."):
        example = item['sentence']
        type = item['event_type']
        type_extend = event_extend_map[type]
        cot = cots[idx]

        backward_details_path = f'./results/details_{mode}.txt'

        if not type_extend:
            with open(backward_details_path, 'a', encoding='utf-8') as f:
                f.write(str(idx+1) + "：\nOutput: None\n\n")
            continue
        else:      
            for r in type_extend:
                msg_list=[
                    {"role": "system", "content": get_sys_prompt(cot)},
                    {"role": "user", "content": get_backward_prompt(example, r)},
                ]
                rsp_1 = call_api(msg_list, 0.3) 
                with open(backward_details_path, 'a', encoding='utf-8') as f:
                    f.write(str(idx+1) + "：\nOutput:" + rsp_1 + "\n\n")
            


run(input_list, event_extend_map, cots, 'baichuan2_13B_en')
