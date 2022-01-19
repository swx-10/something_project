import os
import codecs
import json
import numpy as np
import logging as log

log.basicConfig(level=log.NOTSET)

# 语料文件目录
corpus_path = os.path.join(os.path.dirname(__file__), 'BME')
# 训练集文件路径
train_file = os.path.join(corpus_path,'train_corpus.txt')
# 测试集文件路径
test_file = os.path.join(corpus_path,'valid_corpus.txt')
# 转换前tags标签文件
tags_file = os.path.join(corpus_path, 'tags.txt')

# 转换后语料存盘目录
converted_path = os.path.join(os.path.dirname(__file__), 'BME_json')

def read_tagclass(tag_file):
    """读取标签"""
    tag_class = set()
    with codecs.open(tags_file,'r',encoding='utf-8',errors='ignore') as f:
        for tag in f.read().split('\n'):
            if tag != 'O':
                tag_class.add(tag.split('-')[1])
    return tag_class

def save_convert_tags(tag_class,converted_tag_file):
    """转换后的标签存盘"""
    with codecs.open(converted_tag_file,'w',encoding='utf-8',errors='ignore') as f:
        tags = [{'"' + t + '"':i+1} for i, t in enumerate(tag_class)]
        json.dump(tags,f)

def convert_corpus(tag_class, corpus_file, converted_file):
    """传统标注转换为范围标注"""
    sents = []
    tags = []
    with codecs.open(corpus_file,'r',encoding='utf-8',errors='ignore') as f:
        lines = f.readlines()
        sent,tag = [],[]
        for line in lines:
            # 语句使用空行分隔
            if len(line.strip()) > 0 and line.strip() != '0' and line[2]!='\x00'and len(line.split())>1:
                w,t = line.split()
                tag.append(t)
                sent.append(w)
            if line.strip() == '':
                if sent:
                    sents.append(sent)
                    tags.append(tag)
                    sent,tag = [],[]

    # 转换语料资料存入json
    json_data = []
    for id, sent in enumerate(sents):
        text = ''.join(sent)
        tag = np.array(tags[id])
        filters = (tag != 'O')

        start = -1
        cls = ''
        tag_cls = { t:[] for t in tag_class}
        for i,f in enumerate(filters):
            # 记录起始位置
            if f and start < 0:
                cls = tags[id][i].split('-')[1]
                start = i
            if (f == False) and start >= 0 and cls != '':
                end = i
                tag_cls[cls].append([start, end])
                start = -1
                cls = ''
        json_data.append({"id":id, "text":text, "labels": tag_cls})

    with codecs.open(converted_file,'w',encoding='utf-8') as file:
        # json.dump(json_data, file)
        json.dump(json_data, file,ensure_ascii=False)
    log.info('%s转换成功！'%corpus_file)


if __name__ == '__main__':
    converted_tags_file = os.path.join(converted_path, 'tags.json')
    converted_train_file = os.path.join(converted_path, 'BME_train.json')
    converted_test_file = os.path.join(converted_path, 'BME_test.json')

    tag_class = read_tagclass(tags_file)
    save_convert_tags(tag_class, converted_tags_file)
    convert_corpus(tag_class, train_file, converted_train_file)
    convert_corpus(tag_class, test_file, converted_test_file)