# -*- coding:utf-8 -*-
from genericpath import exists
import tensorflow
import os,json,re
import numpy as np
import logging as log
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.python.keras import activations
from tensorflow.python.keras.layers.core import Lambda
from e_utils import predict_str,pre_dict
from tensorflow.keras import Input
from tensorflow.keras.metrics import Precision,Recall
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Dropout,Dense
from tensorflow.keras.losses import CategoricalCrossentropy              
from keras_bert.bert import get_custom_objects
from keras_bert.tokenizer import Tokenizer
from keras_bert.datasets import get_pretrained,PretrainedList
from keras_bert import load_vocabulary,load_trained_model_from_checkpoint,get_checkpoint_paths
from generator import read_data,DataGenerator,id2label,train_valid_split
from preprocess import preprocess

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
log.basicConfig(level = log.NOTSET)
max_entity_len = 6
classes = 14
data_path = os.path.join(os.path.dirname(__file__),'data','task1_train.txt')
train_path = os.path.join(os.path.dirname(__file__),'data','train.txt')
valid_path = os.path.join(os.path.dirname(__file__),'data','valid.txt')
test_path = os.path.join(os.path.dirname(__file__),'data','task1_eval_data.txt')
test_result_path = os.path.join(os.path.dirname(__file__),'test_result.txt')

log_dir = os.path.join(os.path.dirname(__file__),'tb_log')
model_path = os.path.join(os.path.dirname(__file__),'model','entities_pre.h5')
id2label = {int(k):v for k,v in id2label.items()}

def load_bert(just_load_token=False):
    model_path = get_pretrained(PretrainedList.chinese_base)
    paths = get_checkpoint_paths(model_path)
    token_dict = load_vocabulary(paths.vocab)
    tokenizer = Tokenizer(token_dict)
    if just_load_token:
        return tokenizer
    bert_model = load_trained_model_from_checkpoint(
                                            paths.config,
                                            paths.checkpoint,
                                            trainable=True,
                                            seq_len=None)
    return bert_model,tokenizer

def cus_loss(y_true,y_pred,e = 0.9):
    cce = CategoricalCrossentropy()
    return e*cce(y_true,y_pred) + (1-e)*cce(K.ones_like(y_pred)/classes, y_pred)

def micro_f1(y_pred, y_true, model='multi'):
    epsilon = 1e-7
    axis = 2
    y_pred = tf.round(y_pred)#将经过sigmoid激活的张量四舍五入变为0，1输出
    tp = tf.reduce_sum(tf.cast(y_pred*y_true, 'float'), axis=axis)
    #tn = tf.sum(tf.cast((1-y_pred)*(1-y_true), 'float'), axis=axis)
    fp = tf.reduce_sum(tf.cast(y_pred*(1-y_true), 'float'), axis=axis)
    fn = tf.reduce_sum(tf.cast((1-y_pred)*y_true, 'float'), axis=axis)
    p = tp/(tp+fp+epsilon)
    r = tp/(tp+fn+epsilon)
    f1 = 2*p*r/(p+r+epsilon)
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    if model == 'single':
        return f1
    if model == 'multi':
        return tf.reduce_mean(f1)
        
def Focal_Loss(y_true, y_pred, alpha=0.25, gamma=2):
    y_true = K.cast(y_true,'float32')
    y_pred = K.cast(y_pred,'float32')
    y_pred += K.epsilon()
    ce = -y_true * K.log(y_pred)
    weight = K.pow(1 - y_pred, gamma) * y_true
    fl = ce * weight * alpha
    reduce_fl = K.max(fl, axis=-1)
    return reduce_fl

def create_model(bert_model):
    x1_in = Input(shape=(None,),dtype='int32')
    x2_in = Input(shape=(None,),dtype='int32')
    # length_in = Input(shape=(None,),dtype='int32')
    # length [batch_size,seq_len,1]
    # length_pre = Dense(1,activations='sigmoid')(D1)
    # length_in = K.expand_dims(length_in,2)
    # length_loss = K.sparse_categorical_crossentropy(length_in,length_pre)
    # length_loss = K.sum(length_loss * mask) / K.sum(mask)
    
    mask = Lambda(lambda x:K.cast(K.greater(K.expand_dims(x,2),0),'float32'))(x1_in)

    x = bert_model([x1_in,x2_in])
    D = Dense(300,activation='relu')(x)
    D1 = Dropout(0.1)(D)

    start = Dense(classes,activation='softmax')(D1)
    end = Dense(classes,activation='softmax')(D1)
    
    # [batch,seq_len,1]
    length = Dense(30,activation='sigmoid')(D1)
    
    
    model = Model(inputs=[x1_in,x2_in],outputs=[start,end,length])

    model.compile(
        loss = Focal_Loss,
        optimizer=tensorflow.keras.optimizers.Adam(2e-5),
        metrics=[Precision(),Recall(),micro_f1])
    model.summary()
    return model

def reload_model(model_path):
    obj = get_custom_objects()
    my_obj = {'Focal_Loss':Focal_Loss,
              'micro_f1':micro_f1}
    obj.update(my_obj)
    model = load_model(model_path,custom_objects=obj)
    return model

def scheduler(epoch,lr):
    if epoch < 3:return lr
    else:return lr * tf.math.exp(-0.1)

if __name__=='__main__':
    if not (os.path.exists(train_path) and os.path.exists(valid_path)):
        train_valid_split(data_path,train_path,valid_path)
        preprocess(train_path)
        preprocess(valid_path)

    '''train'''
    bert_model,tokenizer = load_bert()
    train_data = read_data(train_path)
    valid_data = read_data(valid_path)
    
    batch_size = 1
    train_gen = DataGenerator(train_data,tokenizer,batch_size=batch_size)
    valid_gen = DataGenerator(valid_data,tokenizer,batch_size=batch_size)

    epochs = 10                         # 训练epoch总数
    callback_list = [tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1),
                     tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=2,restore_best_weights=True),
                     tf.keras.callbacks.LearningRateScheduler(scheduler)
                     ]
    model = create_model(bert_model)
    model.fit(
        train_gen.forfit(),
        steps_per_epoch=len(train_gen),
        epochs=epochs,
        validation_data=valid_gen.forfit(),
        validation_steps=len(valid_gen),
        callbacks=callback_list
    )
    model.save(model_path)

    '''test'''
    # f = None

    # if not exists(test_result_path):    f = open(test_result_path,'w',encoding='utf-8')
    # else:   f = open(test_result_path,'a',encoding='utf-8')

    # test_data = read_data(test_path)
    # tokenizer = load_bert(just_load_token=True)
    # model = reload_model(model_path)
    # pattern = re.compile('[,.:;，。；：“”#、| !()@↓*〔〕《》]')
    
    # # pattern_num = input('请输入匹配模式\nstrict:1\ntolerant:2\n')
    # # if pattern_num==1:model_pattern = 'strict'
    # # else:model_pattern = 'tolerant'
    # # print(f'匹配模式：{model_pattern}')
    # model_pattern = 'strict'

    # pre_dict(test_data,f,model_pattern,pattern,model,tokenizer)
