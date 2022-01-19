# -*- coding:utf-8 -*-
import re,os,json,math
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tqdm import tqdm
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam
from keras_bert.util import get_checkpoint_paths
from keras_bert.datasets import PretrainedList,get_pretrained
from keras_bert import load_trained_model_from_checkpoint
from tokenizer import OurTokenizer,read_token
from preprocess import split_dataset,repair
from generator import DataGenerator

mode = 0
# 学习率
learning_rate = 5e-5
# 最小学习率
min_learning_rate = 1e-5

# bert model config
bert_path = get_pretrained(PretrainedList.chinese_base)
paths = get_checkpoint_paths(bert_path)
config_path = paths.config
checkpoint_path = paths.checkpoint
dict_path = paths.vocab

train_file = os.path.join(os.path.dirname(__file__),'data','train_data.json')
dev_file = os.path.join(os.path.dirname(__file__),'data','dev_data.json')
schema_file = os.path.join(os.path.dirname(__file__),'data','schema.json')

train_data = json.load(open(train_file))
dev_data = json.load(open(dev_file))
id2predicate,predicate2id = json.load(open(schema_file))
id2predicate = {int(i):j for i,j in id2predicate.items()}
num_classes = len(id2predicate)

total_data = []
total_data.extend(train_data)
total_data.extend(dev_data)

train_data,test_data = split_dataset(total_data,split_rate=0.8)
train_data,valid_data = split_dataset(train_data,split_rate=0.9)

# 三元组信息
# 格式:{predicate:[(subject,predicate,object)]}
# subject 的 predicate 是 object
predicates = {}

# 遍历训练数据,提取谓语集合
for d in train_data:
    repair(d)
    for sp in d['spo_list']:
        if sp[1] not in predicates:
            predicates[sp[1]] = []
        predicates[sp[1]].append(sp)

for d in dev_data:
    repair(d)

token_dict = read_token(dict_path)
tokenizer = OurTokenizer(token_dict)

def seq_gather(x):
    """
    seq是[None, seq_len, s_size]的格式，
    idxs是[None, 1]的格式，在seq的第i个序列中选出第idxs[i]个向量，
    最终输出[None, s_size]的向量。
    """
    seq, idxs = x
    # [batch_size, index]
    # example:[[6],[11],[3],...]
    idxs = K.cast(idxs, 'int32')
    # shape:[batch_size] 
    # example:[0,1,2,..]
    batch_idxs = K.arange(0, K.shape(seq)[0])
    # shape:[batch_size,1]
    # example:[[0],[1],[2],...]
    batch_idxs = K.expand_dims(batch_idxs, 1)
    # shape:[batch_size,2]
    # example:[[0,6],[1,11],[2,3],...]
    # batch第几个(0~batch_size)  seq_len维度第几个(idxs_exp:6,11,3)
    # 第0个text的第6个字的subject语义相关向量
    idxs = K.concatenate([batch_idxs, idxs], 1)
    # shape:[batch_size, index_len, s_size=768]
    return tf.gather_nd(seq, idxs)

def create_model():
    bert_model = load_trained_model_from_checkpoint(config_path,checkpoint_path,seq_len=None,trainable=True)
    t1_in = Input(shape=(None,))    # text token
    t2_in = Input(shape=(None,))    # seg token
    s1_in = Input(shape=(None,))    # subject start  mask
    s2_in = Input(shape=(None,))    # subject end   mask
    k1_in = Input(shape=(1,))    # subject start_index
    k2_in = Input(shape=(1,))    # subject end_index
    o1_in = Input(shape=(None,num_classes))    # object predicate start_mask
    o2_in = Input(shape=(None,num_classes))    # object predicate end_mask

    t1, t2, s1, s2, k1, k2, o1, o2 = t1_in, t2_in, s1_in, s2_in, k1_in, k2_in, o1_in, o2_in

    # token_index输入中的所有非零值转换为1.0，为后面的计算做好mask
    mask =  Lambda(lambda x: K.cast(K.greater(K.expand_dims(x,2),0),'float32'))(t1)

    t = bert_model([t1,t2])
    # ps1 output:[batch_size,seq_len,1]
    ps1 = Dense(1,activation='sigmoid')(t)
    ps2 = Dense(1,activation='sigmoid')(t)
        
    subject_model = Model([t1_in,t2_in],[ps1,ps2])  # 预测subject的模型


    # 提取bert输出中subject相关向量[batch,768]
    k1v = Lambda(seq_gather)([t,k1])
    k2v = Lambda(seq_gather)([t,k2])
    # 计算均值后加bert输出,增强subject的表示
    kv = Average()([k1v,k2v])   # batch,768
    t = Add()([t,kv])           # batch,seq_len,768
    po1 = Dense(num_classes,activation='sigmoid')(t)    # 输出每个predicate的概率
    po2 = Dense(num_classes,activation='sigmoid')(t)

    # 输入text和subject，预测object及其predicate
    # text,seg,sub_st,sub_ed ——> predicate
    object_model = Model([t1_in,t2_in,k1_in,k2_in],[po1,po2])   # 根据text,sub_st,sub_ed学习predicate
    # 训练模型
    train_model = Model([t1_in,t2_in,s1_in,s2_in,k1_in,k2_in,o1_in,o2_in],
                        [ps1,ps2,po1,po2])
    # subject相对位置，[batch_size,seq_len] -> [batch_size,seq_len,1]
    s1 = K.expand_dims(s1,2)    # s1 sub_start mask
    s2 = K.expand_dims(s2,2)    # s2 sub_end mask

    # subject_model预测的结果和subject相对位置计算交叉熵损失CE(预测的是subject的概率)  
    # 经过mask后计算得到关于subject的两个loss(start,end)
    # mask对于text_token做成[1,1,1,1,1,0,0,0,0,0,0]后面是padding [batch_size,seq_len]
    s1_loss = K.binary_crossentropy(s1,ps1)
    s1_loss = K.sum(s1_loss * mask) / K.sum(mask)
    s2_loss = K.binary_crossentropy(s2,ps2)
    s2_loss = K.sum(s2_loss * mask) / K.sum(mask)

    # object_model预测的结果和Object相对位置计算交叉熵损失，(预测的是predicate的概率)
    # 多predicate概率求和汇总后经过mask计算得到关于predicate的两个loss(start,end)
    # o1: object start predicate mask  [batch_size,seq_len,num_classes]
    # o2: object end mask
    o1_loss = K.sum(K.binary_crossentropy(o1,po1),2,keepdims=True)
    o1_loss = K.sum(o1_loss * mask) / K.sum(mask)
    
    o2_loss = K.sum(K.binary_crossentropy(o2,po2),2,keepdims=True)
    o2_loss = K.sum(o2_loss * mask) / K.sum(mask)

    # 模型训练损失
    loss = (s1_loss + s2_loss) + (o1_loss + o2_loss)
    train_model.add_loss(loss)
    train_model.compile(optimizer=Adam(learning_rate))
    train_model.summary()
    return subject_model, object_model,train_model

def extract_items(text_in,tokenizer,subject_model,object_model):
    '''根据输入文本通过模型预测提取三元组'''
    _tokens = tokenizer.tokenize(text_in)
    _t1,_t2 = tokenizer.encode(first=text_in)
    _t1 ,_t2 = np.array([_t1]), np.array([_t2])
    # 预测subject的start、end
    _k1,_k2 = subject_model.predict([_t1,_t2])
    _k1,_k2 = np.where(_k1[0]>0.5)[0] ,np.where(_k2[0] > 0.4)[0]
    _subjects = []
    # 筛选大于当前start最近的end索引,提取subject
    for i in _k1:
        j = _k2[_k2 >= i]
        if len(j) > 0:
            j = j[0]
            _subject = text_in[i-1:j]   # i-1 单字情况
            _subjects.append((_subject,i,j))    #(subject,start,end)
    if _subjects:
        R = []
        # 为每个subject复制一套bert输入
        _t1 = np.repeat(_t1,len(_subjects),0)
        _t2 = np.repeat(_t2,len(_subjects),0)
        # 独立拆出subject的start_index/end_index列表
        _k1,_k2 = np.array([_s[1:] for _s in _subjects]).T.reshape((2,-1,1))
        # 预测object及其关联的predicate
        _o1,_o2 = object_model.predict([_t1,_t2,_k1,_k2])
        # 遍历subject
        for i,_subject in enumerate(_subjects):
            # 阈值筛选object的start,end索引
            _oo1,_oo2 = np.where(_o1[i] > 0.5),np.where(_o2[i] > 0.4)
            # 结果矩阵中通过行列筛选
            for _ooo1,_c1 in zip(*_oo1):
                for _ooo2,_c2 in zip(*_oo2):
                    # start_index < end_index且predicate_class相同
                    if _ooo1 <= _ooo2 and _c1 == _c2:
                        _object = text_in[_ooo1-1:_ooo2]
                        _predicate = id2predicate[_c1]
                        R.append((_subject[0],_predicate,_object))
                        break
        # 单独处理关系
        zhuanji,gequ = [],[]
        for s,p,o in R[:]:
            # 妻子关系互换subject/object后补充丈夫关系
            if p == u'妻子':
                R.append((o,u'丈夫',s))
            elif p == u'丈夫':
                R.append((o,u'妻子',s))
            if p == u'所属专辑':
                zhuanji.append(o)
                gequ.append(s)
        spo_list = set()
        for s,p,o in R:
            if p in [u'歌手',u'作词',u'作曲']:
                if s in zhuanji and s not in gequ:
                    continue
            spo_list.add((s,p,o))
        return list(spo_list)

    else:
        return []
                
                        
class Evaluate(Callback):
    def __init__(self,tokenizer,subject_model,object_model,model_path):
        self.F1 = []
        self.best = 0.
        self.passed = 0
        self.stage = 0
        self.tokenizer = tokenizer
        self.subject_model = subject_model
        self.object_model = object_model
        self.model_path = model_path
        
    def on_batch_begin(self,batch,logs=None):
        '''第一个epoch,warmup第二个epoch学习率降到最低'''
        if self.passed <= self.params['steps']:
            lr = (self.passed + 1.) / self.params['steps'] * learning_rate
            K.set_value(self.model.optimizer.lr,lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            lr = (2-(self.passed + 1.) / self.params['steps']) * (learning_rate - min_learning_rate)
            lr += min_learning_rate
            K.set_value(self.model.optimizer.lr,lr)
            self.passed += 1

    def on_epoch_end(self,epoch,logs=None):
        f1,precision,recall = self.evaluate()
        self.F1.append(f1)
        if f1 > self.best:
            self.best = f1
        train_model.save_weights(self.model_path)
        print('f1:%.4f,precision:%.4f,recall:%.4f,best f1:%.4f\n' % (f1,precision,recall,self.best))
    
    def evaluate(self):
        '''评估'''
        orders = ['subject','predicate','object']
        A,B,C = 1e-10,1e-10,1e-10
        # 评估结果存盘
        F = open('./dev_pred.json','a',encoding='utf-8')
        for d in tqdm(valid_data[:1000],desc='evaluate:'):
            # 预测值集合
            R = set(extract_items(d['text'],self.tokenizer,self.subject_model,self.object_model))
            # 真实值集合
            T = set([tuple(spo) for spo in d['spo_list']])
            # 预测正确的集合
            A += len(R & T)
            # 预测数量
            B += len(R)
            # 真实值数量
            C += len(T)
            # 计算验证结果存入json
            s = json.dumps({
                'text':d['text'],
                'spo_list':[dict(zip(orders,spo)) for spo in T],
                'spo_list_pred':[dict(zip(orders,spo)) for spo in R],
                'new':[dict(zip(orders,spo)) for spo in R - T ],
                'lack':[dict(zip(orders,spo)) for spo in T - R]
            },ensure_ascii=False,indent=4)
            F.write(s + '\n')
        F.close()
        return 2*A/(B+C), A/B, A/C

def test(test_data,tokenizer,subject_model,object_model):

    orders = ['subject','predicate','object','object_type','object_type']
    F = open('test_pred.json','w',encoding='utf-8')
    for d in tqdm(iter(test_data)):
        R = set(extract_items(d['text'],tokenizer,subject_model,object_model))
        s = json.dumps({
            'text':d['text'],
            'spo_list':[dict(zip(orders,spo + ('',''))) for spo in R]
        },ensure_ascii=False)
        F.write(s + '\n')
    F.close()


if __name__ == '__main__':
    subject_model,object_model,train_model = create_model()
    model_path = os.path.join(os.path.dirname(__file__),'best_model/best_model.weights')
    train_path = os.path.join(os.path.dirname(__file__),'best_model','checkpoint')
    if os.path.exists(train_path):
        train_model.load_weights(model_path)

    'train'
    train_D = DataGenerator(train_data,tokenizer,predicate2id,batch_size=7)
    evaluator = Evaluate(tokenizer,subject_model,object_model,model_path)
    train_model.fit(train_D.forfit(),
                    # steps_per_epoch=len(train_D),
                    steps_per_epoch = 5000,
                    epochs=5,
                    callbacks=[evaluator])
    
    train_model.load_weights(model_path)
    test(test_data,tokenizer,subject_model,object_model)