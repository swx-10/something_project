# -*- coding:utf-8 -*-

import time
import numpy as np
from crf import CRF
from run import load_bert
from preprocess import read_corpus, DataGenerator, label2id, id2label
from keras.models import load_model
from keras_bert import get_custom_objects
from keras_bert.bert import TOKEN_CLS, TOKEN_SEP
import logging as log

log.basicConfig(level=log.DEBUG,
                format='%(asctime)s %(levelname)s %(message)s')

def load_bertcrf_model():
    """加载预训练好的bert-crf模型"""
    custom_objects = get_custom_objects()
    crf = CRF(True)
    crf_objects = {'CRF':crf, 'loss':crf.loss}
    custom_objects.update(crf_objects)
    model = load_model('C:\\Users\\songsong\\Desktop\\project\\NER\\NER_law\\code\\ber_crf_NER.h5', custom_objects=custom_objects,compile=False)
    return model

def max_in_dict(d): 
    """求字典中最大值的函数"""
    key,value = list(d.items())[0]
    for i,j in list(d.items())[1:]:
        if j > value:
            key,value = i,j
    return key,value

def viterbi(nodes, trans): 
    """viterbi算法"""
    paths = nodes[0] # 初始化起始路径
    for l in range(1, len(nodes)): # 遍历后面的节点
        paths_old,paths = paths,{}
        for n,ns in nodes[l].items(): # 当前时刻的所有节点
            max_path,max_score = '',-1e10
            for p,ps in paths_old.items(): # 截止至前一时刻的最优路径集合
                score = ns + ps + trans[p[-1]+n] # 计算新分数
                if score > max_score: # 如果新分数大于已有的最大分
                    max_path,max_score = p+n, score # 更新路径
            paths[max_path] = max_score # 储存到当前时刻所有节点的最优路径
    return max_in_dict(paths)

if __name__=='__main__':
    time_start = time.time()
    tokenizer = load_bert(just_load_token=True)
    test_data = read_corpus('./corpus/BME/train_corpus.txt')
    test_gen = DataGenerator(test_data,tokenizer, batch_size=1)
    # 加载bert-crf模型
    model = load_bertcrf_model()
    ttokenizer = {v:k for k,v in tokenizer._token_dict.items()}
    log.debug(f'模型加载耗时：{time.time()-time_start}s')
    count = 0
    if count<10:
        count+=1
        tmp = time.time()
        # 提取1条要预测语句
        tids,label = next(test_gen.forfit())
        # 预测推理
        proba = model.predict(tids)[0]
        # 实体类别数
        n = len(label2id)
        # 模型最后一层(CRF)的权重矩阵,并去掉MASK部分
        transform_weights = model.get_weights()[-1][:n,:n]
        trans = {}
        # 转移矩阵中所有可能的前后2类型状态组合
        indexes = label2id.values()
        for i in indexes:
            for j in indexes:
                trans[str(i)+str(j)] = transform_weights[i, j]

        # 提取前n个类别的实体标签预测值
        proba = proba[:,:n]
        # 截取句子实际长度的内容(减2去掉CLS和SEP两个标记)
        token_len = (tids[0] > 0).sum() - 2
        proba = proba[1:token_len]

        labels = [str(i) for i in label2id.values()]
        nodes = [dict(zip(labels, i)) for i in proba]
        if nodes != []:
            tags = viterbi(nodes, trans)
            tags = ' '.join(id2label[i] for i in tags[0])
            label = [np.argmax(i) for i in [j for j in label[0]]]
            lab=''
            for i in label:
                lab = ' '.join(lab +id2label[str(i)])
            sent = ''.join([ttokenizer[i] for i in tids[0][0]])
            log.info('————————————————————————————————————————————————————————————————————')
            log.debug(f'原文：{sent}')
            log.info(f'原实体类别：\n{lab}')
            log.info(f'推理类别:\n{tags}')
            log.debug(f'推理此条语句耗时{time.time()-tmp}')