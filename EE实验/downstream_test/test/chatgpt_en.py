import openai
import json
from tqdm import tqdm
import time

openai.api_key = 'api_key'


with open('../CASIE/processed_data/test_input_list.txt','r',encoding='utf-8') as f:
    test_input_list = [item.strip('\n') for item in f.readlines()]

with open('./data/casie/input_list.json','r',encoding='utf-8') as f:
    input_list = json.load(f)

with open('./data/casie/event_extend_map.json','r',encoding='utf-8') as f:
    event_extend_map = json.load(f)

with open('./data/casie/cots.txt','r',encoding='utf-8') as f:
    cots = [eval(line) for line in f]

id2test_input_list = {}
for idx, item in enumerate(test_input_list):
    id2test_input_list[idx] = item


def get_sys_prompt(cot):
    return(f'''You are currently an expert in extracting event-triggered words.

Your task is to extract the trigger words that match the event type given the text and event type, and follow the following rules when generating

1. Based on the given event type, combined with the context of the given text, extract the trigger word span that may exist in the given event type
2. Generate judgment sentences based on the paired event type and event trigger word span, and check whether each judgment sentence is correct, and only output "yes" or "no"
3. Generate an event detection list according to the judgment sentence, (event type, event trigger span), where the event type must be a given event type

The following is an example of a chain of thought to help you think step by step to solve the above problems
Input : ""{id2test_input_list[cot[0]]}" Event type: [{cot[1]}]
Trigger span: [{cot[2]}]"
answer：
Is the trigger word for event type "{cot[1]}" "{cot[2]}"? yes

Generate a list of event detections:
``
({cot[1]}, {cot[2]})
``''')


def get_usr_prompt(example, extend_eve):
    return(f'''Input: ""{example}" Event type: [{extend_eve}]"''')


def call_chatgpt(example,extend_eve,cot,p):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        top_p=p,
        messages=[
            {"role": "system", "content": get_sys_prompt(cot)},
            {"role": "user", "content": get_usr_prompt(example, extend_eve)},
        ]
    )
    return completion.choices[0].message.content


def run(input_list, extend_map, cots):
    for idx, item in tqdm(enumerate(input_list), total=len(input_list), desc="Processing..."):
        example = item['sentence']
        type = item['event_type']
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

run(input_list, event_extend_map, cots)