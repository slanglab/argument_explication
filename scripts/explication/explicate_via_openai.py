import json, time
import os, re
import pandas as pd
import numpy as np
from utils import load_jsonl

with open('/argument_explication/data/phrases/toulmin.txt', 'r') as f:
    lines = f.readlines()
phrases = [l.strip('\n') for l in lines] 


#ARCT
file_name = '/argument_explication/data/evaluation_datasets/arct_test_full.txt'
data = pd.read_csv(file_name, sep='\t')
print(len(data))


N = len(data)
for phrase in phrases:
    for start in range(0, N, 50): #len(data)
        filename = "../openai-cookbook/examples/data/my_example_requests_to_parallel_process.jsonl"
        jobs = []
        
        for x in range(start, min(start+50, N), 1):
           
            row = data.iloc[x]
            reason = row['reason']
            claim = row['claim']
            context = row['debateTitle'] + '\n' +row['debateInfo']
            
            if row['correctLabelW0orW1']==0:
                warrant = row['warrant0']
            else:
                warrant = row['warrant1']
                
            claim = claim[0].lower() + claim[1:]
            if claim[-1] != '.':
                claim += '.'
                
            reason = reason[0].lower() + reason[1:]
            if reason[-1] != '.':
                reason += '.'
            
            input_message = 'Argument: '+data[x]['full_text']
            
            prompt = input_message+'''\n\nAccording to {},\n'''.format(phrase) 

            metadata = data[x]
            metadata['input_prompt'] = prompt
            metadata['row'] = x
            
            jobs += [{"model": "gpt-4-0613", "messages": [{"role": "user", "content": prompt}],\
                      "max_tokens":256, "temperature":0.0, "metadata": metadata}]
           
            
            with open(filename, "w") as f:
                for job in jobs:
                    json_string = json.dumps(job)
                    f.write(json_string + "\n")
        
        command = 'python ../openai-cookbook/examples/api_request_parallel_processor.py --requests_filepath ../openai-cookbook/examples/data/my_example_requests_to_parallel_process.jsonl --save_filepath ../results/warrant_validation/arct_'
        
        #split phrase into words and join by underscore
        phrase = re.sub('\(', 'LRB ', phrase)
        phrase = re.sub('\)', ' RRB ', phrase)
        phrase = phrase.split()
        phrase = '_'.join(phrase)
        command+=phrase+'_0.0'
        command+='.jsonl --request_url https://api.openai.com/v1/chat/completions'

        os.system(command)
        time.sleep(40)



