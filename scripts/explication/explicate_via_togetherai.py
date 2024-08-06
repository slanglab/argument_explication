import json
import re, os
import copy
import numpy as np
from prompts import *
from utils import *
from tqdm import tqdm
import pandas as pd


import os, together
os.environ["TOGETHER_API_KEY"] = key
together.api_key = os.environ["TOGETHER_API_KEY"]
together.Models.start("togethercomputer/llama-2-70b-chat") #falcon-40b llama-2-70b-chat


import together
import logging
from typing import Any, Dict, List, Mapping, Optional
from pydantic import Extra, Field, root_validator
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.utils import get_from_dict_or_env

class TogetherLLM(LLM):
    """Together large language models."""

    model: str = "togethercomputer/falcon-40b"
    """model endpoint to use"""

    together_api_key: str = os.environ["TOGETHER_API_KEY"]
    """Together API key"""

    temperature: float = 0.7
    """What sampling temperature to use."""

    max_tokens: int = 512
    """The maximum number of tokens to generate in the completion."""
    
    repetition_penalty: float = 0
        
    top_p: float = 0
    

    class Config:
        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the API key is set."""
        api_key = get_from_dict_or_env(
            values, "together_api_key", "TOGETHER_API_KEY"
        )
        values["together_api_key"] = api_key
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "together"

    def _call(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Call to Together endpoint."""
        together.api_key = self.together_api_key
        output = together.Complete.create(prompt,\
                                        model=self.model,\
                                        max_tokens=self.max_tokens,\
                                        temperature=self.temperature,\
                                        repetition_penalty=self.repetition_penalty,\
                                        top_p = self.top_p)
        text = output['output']['choices'][0]['text']
        return text

test_llm = TogetherLLM(
    model= "togethercomputer/llama-2-70b-chat",
    temperature=0.0,
    max_tokens=512
)

    

with open('../phrases/toulmin.txt', 'r') as f:
    lines = f.readlines()
phrases = [l.split(',')[0] for l in lines] 

def load_jsonl(path):
    data=[]
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data 

# ARCT
claim_count = 0
file_name = '../implicit_premise_data/argument-reasoning-comprehension-task/mturk/annotation-task/data/exported-SemEval2018-train-dev-test/test-full.txt'
data = pd.read_csv(file_name, sep='\t')
print(len(data))

p_count = 0
for phrase in phrases[p_count:]:
    print(p_count, phrase)
    p_count+=1
    base_dir = '../results/warrant_validation/AccordingTo/Toulmin/LLAMA2-70B/'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    file_name = base_dir+'becker_test_claims_'+phrase+'_0.0.jsonl'
    if os.path.exists(file_name):   
        claims_toulmin = load_jsonl(file_name)

    else:
        #continue
        claims_toulmin = []
        
    print('Resuming from: ', len(claims_toulmin))
    
    for x in range(len(data)):
        if x<len(claims_toulmin):
            continue
        
        
        #ARCT
        temp = data.iloc[x]
        reason = temp['reason']
        claim = temp['claim']

        if temp['correctLabelW0orW1']==0:
            warrant = temp['warrant0']
        else:
            warrant = temp['warrant1']
    
        row={}
        row['claim'] = claim
        row['reason'] = reason
        row['warrant'] = warrant
        
        claim = claim[0].lower() + claim[1:]
        if claim[-1] != '.':
            claim += '.'
        reason = reason[0].lower() + reason[1:]
        if reason[-1] != '.':
            reason += '.'
            
        coin = np.random.choice([0, 1], 1)[0]
        if coin==1:
            input_message = reason+' '+claim
        else:
            input_message = claim+' '+reason
            
        input_message = input_message[0].upper() + input_message[1:]
        if input_message[-1] != '.':
            input_message += '.'
                
        prompt = input_message+'''\n\nAccording to {},\n'''.format(phrase) 

        response = test_llm(prompt)

        temp = {}
        temp['input_prompt'] = prompt
        temp['original_response'] = response
        temp['original_claim'] = row['claim']
        temp['original_reason'] = row['reason']
        temp['original_warrant'] = row['warrant']
        
        claims_toulmin.append(temp)
        del temp

        save_jsonl(base_dir+'arct_test_claims_'+phrase+'_0.0.jsonl', claims_toulmin)

        claim_count+=1  
        
    #save everything at last
    save_jsonl(base_dir+'arct_test_claims_'+phrase+'_0.0.jsonl', claims_toulmin)