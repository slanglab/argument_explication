import json

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