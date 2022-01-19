# -*- coding:utf-8 -*-

import token
from keras import Input
from keras.models import Model
from keras.layers import Lambda,Dense
from keras_bert.datasets import get_pretrained,PretrainedList
from keras_bert import load_vocabulary,load_trained_model_from_checkpoint,get_checkpoint_paths,Tokenizer
from crf import CRF
import keras.backend as K
import numpy as np

from preprocess import DataGenerator,read_corpus,id2label

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


    
def create_model(bert_model):
    """
    创建Bert-crf模型
    """
    x1_in = Input(shape=(None,),dtype='int32')
    x2_in = Input(shape=(None,),dtype='int32')

    x = bert_model([x1_in,x2_in])
    p = Dense(29)(x)
    crf = CRF(True)
    tag_score = crf(p)

    model = Model(inputs=[x1_in,x2_in],outputs=tag_score)
    model.compile(loss=crf.loss,
                  optimizer='adam',
                  metrics=[crf.accuracy])
    
    return model


if __name__=='__main__':
    # 加载bert pretrained model和tokenizer
    bert_model,tokenizer = load_bert()
    # 读取语料
    train_data = read_corpus('./corpus_data/train_corpus.txt')
    valid_data = read_corpus('./corpus_data/valid_corpus.txt')
    # 创建生成器
    train_gen = DataGenerator(train_data,tokenizer,batch_size=10) 
    # next(train_gen)
    valid_gen = DataGenerator(valid_data,tokenizer,batch_size=10)
    # next(valid_gen)

    model = create_model(bert_model)
    model.summary()

    model.fit(
        train_gen.forfit(),
        steps_per_epoch=len(train_gen),
        epochs =10,
        validation_data=valid_gen.forfit(),
        validation_steps=len(valid_gen)
    )
    model.save('models/ber_crf_NER.h5')



