import json
from tqdm import tqdm
import pickle
import spacy
from spacy.lang.en import English
import pandas as pd
nlp = spacy.load("en_core_web_sm")  


from spacy.lang.en import English
tokenizer = English().tokenizer

def load_jsonl(path):
    data=[]
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data 

def save_jsonl(name, data):
    with open(name, 'w') as outfile:
        for entry in data:
            json.dump(entry, outfile)
            outfile.write('\n')     
            
toulmin_data = load_jsonl('./C4_data/raw_data/Toulmin_in_C4.json')[0]

frequency_dict = {}
all_sentence_count = 0
sentence_count = 0
tokens_count = 0
sentence_token_count = 0
phrase_2_sent = {}
for t in tqdm(toulmin_data):
    text = t['searchresulttext']
    tokens = tokenizer(text)
    tokens_count+=len(tokens)
    
    sentences = nlp(text)
    
    
    all_noun_phrases = []
    for sentence in sentences.sents:
        sent = sentence.text
        all_sentence_count+=1
        #extract all noun phrases from sentences which mention toulmin; not all phrases will contain toulmin.
        if 'toulmin' in sent.lower():
            noun_phrases = []
            for np in sentence.noun_chunks:
                noun_phrases.append(np.text)
                if np.text in phrase_2_sent:
                    phrase_2_sent[np.text].append(sent)
                else:
                    phrase_2_sent[np.text] = [sent]
            all_noun_phrases+=noun_phrases
            sentence_count+=1
            sentence_token_count+=len(tokenizer(sent))
    
    t['all_noun_phrases'] = all_noun_phrases
    for p in all_noun_phrases:
        if p not in frequency_dict:
            frequency_dict[p] = 1
        else:
            frequency_dict[p]+=1
            
frequency_dict_sorted = {k: v for k, v in sorted(frequency_dict.items(), key=lambda item: -1*item[1])}
print('token_count: ', tokens_count)
print('all_sentence_count', all_sentence_count)
print('sentence_count: ', sentence_count)
print('sentence_token_count: ', sentence_token_count)

#save processed data and phrases
with open('./C4_data/nounphrases/Toulmin/toulmin.pickle', 'wb') as handle:
    pickle.dump(frequency_dict_sorted, handle)
with open('./C4_data/nounphrases/Toulmin/toulmin_sent2phrase.pickle', 'wb') as handle:
    pickle.dump(phrase_2_sent, handle)
save_jsonl('./C4_data/processed_data/Toulmin_in_C4.json', toulmin_data)


#apply filters to extracted phrases
words = ['model', 'models', 'analysis', 'method', 'methods', 'scheme', 'schemes', 'schema', 'framework', 'frameworks',\
         'theory', 'theories', 'strategies', 'strategy', 'approach', 'approaches', 'algorithm', 'algorithms']
filtered_dict = {}
for k, v in frequency_dict_sorted.items():
    for word in words:
        if word in k.lower().split(' ') and len(k.split(' '))>1:
            if k.lower() not in ['the '+ word, 'a '+word,\
                                 'an '+word, 'this '+word,\
                                 'every '+word, 'our '+word]:
                filtered_dict[k] = v
                
filtered_dict_more_than_1 = {k: v for k, v in filtered_dict.items() if v > 1}

#save phrases for manual review
df = pd.DataFrame()
df['phrase'] = filtered_dict_more_than_1.keys()
df['counts'] = filtered_dict_more_than_1.values()
df['sentences'] = [phrase_2_sent[k] for k in filtered_dict_more_than_1.keys()]
df.to_csv('./C4_data/nounphrases/Toulmin/toulmin_filtered.csv', sep='\t')