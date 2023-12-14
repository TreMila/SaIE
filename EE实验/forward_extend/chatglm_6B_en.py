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


def get_sys_prompt():
    return(f'''You are currently a senior event type detection expert.

Your task is to generate just one kind of event types for event trigger words in a given text, following the rules when generating:
1. Based on the event trigger word, combined with the context of the given text, the possible event types are generated
2. The output form is: [event type]

The following is an example of a chain of thought to help you think step by step:
Text: "Two of my nephews were excluded from their brother ' s wedding this weekend because they are not LDS."  Trigger word: [wedding]

Output: [marry]''')


def get_forward_prompt(example, trigger_span):
    return(f'''Text: "{example}" Trigger word: [{trigger_span}]
Output:''')


def run(input_list, golds, get_sys_prompt, get_forward_prompt, mode):
    for idx, gold in tqdm(enumerate(golds), total=len(golds), desc="Processing..."):
        example = input_list[gold[0]]
        gold_type = gold[1]
        trigger_span = gold[2]

        prompt = get_sys_prompt() + get_forward_prompt(example, trigger_span)
        history = []
        rsp_1 = call_api(prompt, history, 0.3) 
        rsp_2 = call_api(prompt, history, 0.6)
        rsp_3 = call_api(prompt, history, 1)
        forward_details_path = f'./results/extend_forward_{mode}.txt'
        with open(forward_details_path, 'a', encoding='utf-8') as f:
            f.write(str(idx+1) + "：\n" + rsp_1 + "\n" + rsp_2 + "\n" + rsp_3 + "\n" 
                    + "gold:" + str(gold_type) + "\n\n")


with open('../CASIE/processed_data/train_input_list.txt','r',encoding='utf-8') as f:
    input_list = [item.strip('\n') for item in f.readlines()]

with open(f'./data/casie/golds.txt','r',encoding='utf-8') as f:
    golds = [eval(line) for line in f]

run(input_list, golds, get_sys_prompt, get_forward_prompt, 'chatglm_6B_en')
