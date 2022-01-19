# -*- coding:utf-8 -*-
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Lambda,Dense,Dropout
from keras_bert.datasets import get_pretrained,PretrainedList
from keras_bert import load_vocabulary,load_trained_model_from_checkpoint,get_checkpoint_paths,Tokenizer
from keras_bert.bert import get_custom_objects
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import numpy as np
import os

from preprocess import DataGenerator,read_corpus,id2label,read_tags,read_json_corpus

def load_bert(just_load_token=False):
    """
    加载bert预训练模型和tokenizer或只加载tokenizer
    param: just_load_token:取值True时，只加载tokenizer
    """
    # load中文与训练模型（当前/.keras/gatasets）
    model_path = get_pretrained(PretrainedList.chinese_base)
    # 模型所在目录的path
    paths = get_checkpoint_paths(model_path)
    # load token 
    token_dict = load_vocabulary(paths.vocab)
    # create tokenizer
    tokenizer = Tokenizer(token_dict)
    if just_load_token:
        return tokenizer

    # load pretrained model
    bert_model = load_trained_model_from_checkpoint(
        paths.config,
        paths.checkpoint,
        trainable=True,
        seq_len=None
    )
    return bert_model,tokenizer 

def my_loss(y_true,y_pred,e = 0.9):
    cce = tf.keras.losses.CategoricalCrossentropy()
    return e*cce(y_true,y_pred) + (1-e)*cce(K.ones_like(y_pred)/11, y_pred)

def create_model(bert_model):
    """
    创建Bert-crf模型
    """
    x1_in = Input(shape=(None,),dtype='int32')
    x2_in = Input(shape=(None,),dtype='int32')
    x = bert_model([x1_in,x2_in])
    D1 = Dense(512,activation='relu')(x)
    D2 = Dropout(0.1)(D1)
    start = Dense(11,activation='softmax')(D2)
    end = Dense(11,activation='softmax')(D2)
    model = Model(inputs=[x1_in,x2_in],outputs=[start,end])

    model.compile(loss=my_loss,
                  optimizer=Adam(2e-5),
                  metrics=[tf.keras.metrics.Recall()])
    return model

def save_model(model,save_path):
    model.save(save_path)

def reverse_model(save_path):
    cust_obj = get_custom_objects()
    my_objects = {'my_loss':my_loss}
    cust_obj.update(my_objects)
    model = load_model(save_path,custom_objects=cust_obj)
    return model

def predict(content,model,tokenizer):   
    ten_ids = np.asarray(tokenizer.encode(content)[0])
    seg_ids = np.asarray([0]*len(ten_ids))
    inputs = [ten_ids[None,:],seg_ids[None,:]]
    start,end = model.predict(inputs)
    # 概率
    start_seq = np.argmax(start,axis=-1).squeeze()
    end_seq = np.argmax(end,axis=-1).squeeze()
    # 去除首尾的[CLS][SEP]
    start = start_seq[1:-1]
    end = end_seq[1:-1]
    return {'start':start,'end':end}


if __name__=='__main__':
    # 加载bert pretrained model和tokenizer
    bert_model,tokenizer = load_bert()
    # 读取语料
    train_path = os.path.join(os.path.dirname(__file__),'corpus_json','BME_json','BME_train.json')
    valid_path = os.path.join(os.path.dirname(__file__),'corpus_json','BME_json','BME_test.json')

    train_data = read_corpus(train_path)
    valid_data = read_corpus(valid_path)
    # 创建生成器
    batch_size=4

    train_gen = DataGenerator(train_data,tokenizer,batch_size=batch_size)
    valid_gen = DataGenerator(valid_data,tokenizer,batch_size=batch_size)

    model = create_model(bert_model)
    model.summary()
    model.fit(
        train_gen.forfit(),
        steps_per_epoch=len(train_gen),
        epochs = 10,
        validation_data=valid_gen.forfit(),
        validation_steps=len(valid_gen)
    )
    

    save_path = os.path.join(os.path.dirname(__file__),'spannet_NER.h5')

    save_model(model,save_path)

    '''预测'''
    tokenizer = load_bert(just_load_token=True)
    save_path = os.path.join(os.path.dirname(__file__),'spannet_NER.h5')
    model = reverse_model(save_path)
    # 加载测试预料和标签
    corpus_file = os.path.join(os.path.dirname(__file__),'corpus_json','test','test.json')
    json_corpus = read_json_corpus(corpus_file)

    tags_file = os.path.join(os.path.dirname(__file__),'corpus_json','BME_json','tags.json')
    tags = read_tags(tags_file)
    
    # 模型推理
    while True:
        # 随机选取测试语句
        idx = np.random.randint(len(json_corpus))
        sent = json_corpus[idx]['text']
        if len(sent) > 512:continue

        cause = json_corpus[idx]['labels']['cause']
        doc_type = json_corpus[idx]['labels']['doc_type']
        judges = json_corpus[idx]['labels']['judges']
        dem_agent = json_corpus[idx]['labels']['dem_agent']
        defendant = json_corpus[idx]['labels']['defendant']
        demandant = json_corpus[idx]['labels']['demandant']
        def_agent = json_corpus[idx]['labels']['def_agent']
        legal_regulation = json_corpus[idx]['labels']['legal_regulation']
        time = json_corpus[idx]['labels']['time']

        print(predict(sent,model,tokenizer))
        print(sent)

        print('cause',cause)
        print('doc_type',doc_type)
        print('judges',judges)
        print('dem_agent',dem_agent)
        print('defendant',defendant)
        print('demandant',demandant)
        print('def_agent',def_agent)
        print('legal_regulation',legal_regulation)
        print('time',time)

      
        choice = input('是否继续(y/n)').lower()
        if choice == 'n':
            break






