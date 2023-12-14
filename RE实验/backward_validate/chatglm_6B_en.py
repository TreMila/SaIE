import requests
import json
from tqdm import tqdm

URI = 'api_for_chatglm_6B'


with open('./data/scierc/cot_examples.txt','r',encoding='utf-8') as f:
    cot_examples = [item.strip('\n') for item in f.readlines()]

with open('./data/scierc/cot_spo.txt', 'r', encoding='utf-8') as f:
    cot_spos = [eval(line) for line in f]

with open('./data/scierc/input_list.txt', 'r', encoding='utf-8') as f:
    input_list = [item.strip('\n') for item in f.readlines()]

with open('./data/scierc/final_rel_extend_map_chatglm_6B_en.json', 'r', encoding='utf-8') as f:
    rel_extend_map = json.load(f)

rel_type_list = ['HYPONYM-OF',
 'FEATURE-OF',
 'USED-FOR',
 'CONJUNCTION',
 'EVALUATE-FOR',
 'PART-OF',
 'COMPARE']


input_list2rel = {}
for idx, item in enumerate(input_list):
    input_list2rel[item] = rel_type_list[idx//10]


def get_sys_prompt(cot_example, cot_spo):
    return(f'''You are currently a senior expert in information extraction.

Your task is to give a text and a relation, extract subject-object pairs that match the relation. DO NOT output any Note. Follow the below rules when generating:

1. Based on the given relation, combine the context of the given text, and extract the subject-object pairs that may have the given relation
2. Generate judgment sentences based on the relation list and keywords, and check whether each judgment sentence is correct, and only output "yes" or "no"
3. Generate a list of relations based on the judgment sentence, following the format: (subject, relation, object), where the relation must be a given relation

The following is an example of a chain of thought to help you think step by step to solve the above problems
Text: "{cot_example}" relation: [{cot_spo[1]}]

Output:
subject-object pair: [({cot_spo[2]}, {cot_spo[3]})]
ask: Is the relation between "{cot_spo[2]}" and "{cot_spo[3]}" the "{cot_spo[1]}"? answer: Yes
Generate a list of relations:
``
({cot_spo[2]}, {cot_spo[1]}, {cot_spo[3]})
``''')


def get_backward_prompt(example, extend_rel):
    return(f'''Text: "{example}" relation: [{extend_rel}]
Output:''')


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


def run(input_list, input_list2rel, cot_spos,cot_examples, rel_extend_map, mode):
    for idx, example in tqdm(enumerate(input_list), total=len(input_list), desc="Processing..."):
        rel = input_list2rel[example]
        rel_extend = rel_extend_map[rel]
        cot_example = cot_examples[idx//10]
        cot_spo = cot_spos[idx//10]

        for r in rel_extend:
            prompt = get_sys_prompt(cot_example, cot_spo) + get_backward_prompt(example, r)
            history = []
            rsp_1 = call_api(prompt, history, 0.6) 

            backward_details_path = f'./results/details_{mode}.txt'
            with open(backward_details_path, 'a', encoding='utf-8') as f:
                f.write(str(idx+1) + "：\nOutput:" + rsp_1 + "\n\n")


def run_gold(input_list, input_list2rel, cot_spos,cot_examples, mode):
    for idx, example in tqdm(enumerate(input_list), total=len(input_list), desc="Processing..."):
        rel = input_list2rel[example]
        cot_example = cot_examples[idx//10]
        cot_spo = cot_spos[idx//10]


        prompt = get_sys_prompt(cot_example, cot_spo) + get_backward_prompt(example, rel)
        history = []
        rsp_1 = call_api(prompt, history, 0.3) 

        backward_details_path = f'./results/details_gold_{mode}.txt'
        with open(backward_details_path, 'a', encoding='utf-8') as f:
            f.write(str(idx+1) + "：\nOutput:" + rsp_1 + "\n\n")

run(input_list, input_list2rel, cot_spos, cot_examples, rel_extend_map, 'chatglm_6B_en')
run_gold(input_list, input_list2rel, cot_spos,cot_examples, 'chatglm_6B_en')
