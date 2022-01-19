import os,re
import json
from keras_bert import datasets
import numpy as np
from numpy import random
from tqdm import tqdm

# 训练文件
train_path = os.path.join(os.path.dirname(__file__),'data_ori','train_data.json')
train_target_path = os.path.join(os.path.dirname(__file__),'data','train_data.json')

# 开发文件
dev_path = os.path.join(os.path.dirname(__file__),'data_ori','dev_data.json')
dev_target_path = os.path.join(os.path.dirname(__file__),'data','dev_data.json')

# 类别文件
schema_path = os.path.join(os.path.dirname(__file__),'data_ori','all_50_schemas.json')
schema_target_path = os.path.join(os.path.dirname(__file__),'data','schema.json')

def convert_datafile(json_path,json_target_path):
    '''对原始数据做处理,提取每条中的text和spo_list'''
    json_data = []
    for line in tqdm(open(json_path,encoding='utf-8').readlines()):
        json_item = json.loads(line)
        text = json_item['text']
        spo_list = []
        _spo_list = json_item['spo_list']
        for _spo in _spo_list:
            spo = (_spo['subject'],_spo['predicate'],_spo['object'])
            spo_list.append(spo)
        json_data.append({'text':text,"spo_list":spo_list})
    with open(json_target_path,'w') as f:
        json.dump(json_data,f)

def convert_schema_file(schema_path,target_path):
    '''转换schema格式文件'''
    id2predicate,predicate2id = {},{}
    lines = [line for line in open(schema_path,encoding='utf-8').readlines()]
    predicates = set()
    for i,line in enumerate(lines):
        predicate = json.loads(line)['predicate']
        predicates.add(predicate)
    id2predicate = {i:p for i,p in enumerate(predicates)}
    predicate2id = {p:i for i,p in enumerate(predicates)}

    with open(target_path,'w') as f:
        json.dump((id2predicate,predicate2id),f)

def split_dataset(dataset,split_rate=0.8):
    '''按比例拆分数据集'''
    data_size = len(dataset)
    random_order = list(range(data_size))
    np.random.shuffle(random_order)

    data1 = [dataset[j] for i,j in enumerate(random_order)]
    data2 = [dataset[j] for i,j in enumerate(random_order)]
    return [data1,data2]

def repair(d):
    # 提取文本
    d['text'] = d['text'].lower()
    # 正则提取text中,<>里描述实体
    something = re.findall(u'《([^《》]*?)》',d['text'])
    something = [s.strip() for s in something]
    # 专辑和歌曲信息
    zhuanji = []
    gequ = []
    for sp in d['spo_list']:
        sp = list(sp)
        sp[0] = sp[0].strip(u'《》').strip().lower()
        sp[2] = sp[2].strip(u'《》').strip().lower()
        for some in something:
            if sp[0] in some and d['text'].count(sp[0]) == 1:
                sp[0] = some
            if sp[1] == u'所属专辑':
                zhuanji.append(sp[2])
                gequ.append(sp[0])
    spo_list = []
    for sp in d['spo_list']:
        if sp[1] in [u'歌手',u'作词',u'作曲']:
            if sp[0] in zhuanji and sp[0] not in gequ:
                continue
        spo_list.append(tuple(sp))
    d['spo_list'] = spo_list

if __name__ == '__main__':
    convert_datafile(dev_path,dev_target_path)
    convert_datafile(train_path,train_target_path)
    convert_schema_file(schema_path,schema_target_path)