import json, time
import os, re
import pandas as pd
from utils import load_jsonl


model_name = 'GPT4' #'LLAMA2-70B', 'GPT3' 
    
files = os.listdir('../results/warrant_validation/AccordingTo/Toulmin/'+model_name)
if not os.path.exists('../results/warrant_validation/AccordingTo/Toulmin/'+model_name+'-Dict/'):
    os.makedirs('../results/warrant_validation/AccordingTo/Toulmin/'+model_name+'-Dict/')
    

for f in files:
    if f.endswith('.jsonl') and 'arct_context' in f:
        
        data = load_jsonl('../results/warrant_validation/AccordingTo/Toulmin/'+model_name+'/'+f)
        arranged_data = {}
        
        #open-sourced models
        for i in range(len(data)):
            arranged_data[i] = data[i]
            
        #open-ai models
        #for i in range(len(data)):
        #    arranged_data[data[i][2]['row']] = data[i]
        
        filename = "../openai-cookbook/examples/data/new_example_requests_to_parallel_process.jsonl"
        
        for start in range(0, len(arranged_data), 50):
            jobs = []
            for i in range(start, min(start+50, len(arranged_data)), 1):
                
                ### uncomment when using open-ai models 
                #response = arranged_data[i][1]['choices'][0]['message']['content'] #gpt4
                ##response = arranged_data[i][1]['choices'][0]['text'] #gpt3
                #arranged_data[i][2]['response'] = response
                #metadata = arranged_data[i][2]
                
                #### uncomment when using open-sourced models like Llama
                response = arranged_data[i]['original_response']['output']['choices'][0]['text']
                metadata = {'claim': arranged_data[i]['original_claim'],\
                           'reason': arranged_data[i]['original_reason'],\
                           'response': response,\
                           'input_text': arranged_data[i]['input_prompt']}
                metadata = arranged_data[i]
                metadata['row'] = i
                
                if len(response)>0 and response[-1] != '.':
                    response += '.'
                    
                #instruction = '"""\n\nTherefore, the answer ("yes" or "no") is:'
                instruction = '"""\n\nFormat the above text in a Python dictionary with values as a list of bullet points.\n'
                prompt = '"""'+response+instruction
                
                jobs += [{"model": 'gpt-3.5-turbo-0613', "messages": [{"role": "user", "content": prompt}],\
                          "max_tokens":512, "temperature":0.0, "metadata": metadata}]
                
                with open(filename, "w") as f_:
                    for job in jobs:
                        json_string = json.dumps(job)
                        f_.write(json_string + "\n")
                            
            
            command = 'python ../openai-cookbook/examples/api_request_parallel_processor.py --requests_filepath ../openai cookbook/examples/data/new_example_requests_to_parallel_process.jsonl --save_filepath ../results/warrant_validation/AccordingTo/Toulmin/'+model_name+'-Dict/'
            
            f = '_'.join(f.split(' '))
            command+=f
            command+=' --request_url https://api.openai.com/v1/chat/completions'
            os.system(command)
            time.sleep(40)
            


