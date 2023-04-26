import json

def preprocess():
    kg = {}
    with open('data/raw/zhishime.json', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            temp = json.loads(line)
            if temp['head'] not in kg:
                kg[temp['head']] = {}
            if temp['relation'] not in kg[temp['head']]:
                kg[temp['head']][temp['relation']] = temp['tail']
    with open('data/processed/kg.json', 'w', encoding='utf-8') as json_file:
        json.dump(kg, json_file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    preprocess()