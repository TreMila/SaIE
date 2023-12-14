import requests
import os
import json
from tqdm import tqdm

URI = 'api_for_llama2_70B'

def call_api(prompt,p):
    request = {
        'messages': prompt,
        'max_tokens': 50,
        'top_p': p,
    }

    response = requests.post(URI, json=request)

    if response.status_code == 200:
        result = response.json()['choices']
        return result[-1]['message']['content']



def get_sys_prompt():
    return(f'''You are currently a senior entity type generation expert.

Your task is to generate the entity type for each entity span in the given text. DO NOT output any explanation BUT the only one entity type list.
           
Here is an example thought process to guide you in solving the above problems step by step:
Input: ""Our president has put homeland security in the hands of failed Republican hacks."Entity span: [Republican, Our president]"
Output: [organization, person]''')

def get_forward_prompt(example, span_list):
    return(f'''Text: "{example}" Entity span: {span_list} 
Output:''')


def run(input_list, merged_golds, get_sys_prompt, get_forward_prompt, mode):
    for idx, golds in tqdm(enumerate(merged_golds), total=len(merged_golds), desc="Processing..."):
        example = input_list[golds[0][0]]
        span_list = [gold[2] for gold in golds]
        
        if 'zh' in mode:
            span_list_str = ''
            for span in span_list:
                span_list_str += '（' + span + '），'
            span_list_str = span_list_str[:-1]
        elif 'en' in mode:
            span_list_str = '['
            for span in span_list:
                span_list_str += span + ', '
            span_list_str = span_list_str[:-2] + ']'

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


with open('../ACE05/processed_data/train_input_list.txt','r',encoding='utf-8') as f:
    input_list = [item.strip('\n') for item in f.readlines()]

with open(f'./data/ace05/golds.txt','r',encoding='utf-8') as f:
    golds = [eval(line) for line in f]

with open('./data/ace05/merged_golds.txt','r',encoding='utf-8') as f:
    merged_golds = [eval(line) for line in f]

run(input_list, merged_golds, get_sys_prompt,get_forward_prompt,'llama2_70B_en')