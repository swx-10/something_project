from importlib import import_module
import os,logging,time,keras,collections,keras
from keras.layers.core import Lambda
from preprocess import DataGenerator,read_corpus,__call__,label2id
from keras_bert.datasets import get_pretrained,PretrainedList
from keras_bert.loader import load_vocabulary
from keras_bert.util import get_checkpoint_paths
from keras_bert import Tokenizer,load_trained_model_from_checkpoint,get_custom_objects
from keras import Input
import tensorflow as tf
import numpy as np
import keras.backend as K
from keras.models import Model,load_model
from keras.layers import Dense,Dropout
from keras.optimizers import Adam
from cosine import WarmUpCosineDecayScheduler

num_classes = 5
model_name = 'classify_fl——test.h5'    # classify_5000.h5

def load_bert(just_load_token=False):
    '''
    - 加载bert_chinese pretrained模型
    - just_load_token:为True时，只加载tokenizer
    '''
    model_path = get_pretrained(PretrainedList.chinese_base)
    paths = get_checkpoint_paths(model_path)
    token_dict = load_vocabulary(paths.vocab)
    tokenizer = Tokenizer(token_dict)
    if just_load_token:
        return tokenizer

    # load_pretrained model
    bert_model = load_trained_model_from_checkpoint(
        paths.config,
        paths.checkpoint,
        trainable=True,
        seq_len=None
    )
    return bert_model,tokenizer

def f1_score(y_pred, y_true, model='multi'):
    '''
    输入张量y_pred是输出层经过sigmoid激活的张量
    y_true是label{0,1}的集和
    model指的是如果是多任务分类，single会返回每个分类的f1分数，multi会返回所有类的平均f1分数（Marco-F1）
    如果只是单个二分类任务，则可以忽略model
    '''
    epsilon = 1e-7
    y_pred = tf.round(y_pred)#将经过sigmoid激活的张量四舍五入变为0，1输出
    axis = 1
    tp = tf.reduce_sum(tf.cast(y_pred*y_true, 'float'), axis=axis)
    #tn = tf.sum(tf.cast((1-y_pred)*(1-y_true), 'float'), axis=axis)
    fp = tf.reduce_sum(tf.cast(y_pred*(1-y_true), 'float'), axis=axis)
    fn = tf.reduce_sum(tf.cast((1-y_pred)*y_true, 'float'), axis=axis)
    
    p = tp/(tp+fp+epsilon) # epsilon的意义在于防止分母为0，否则当分母为0时python会报错
    r = tp/(tp+fn+epsilon)
    
    f1 = 2*p*r/(p+r+epsilon)
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    if model == 'single':
        return f1
    if model == 'multi':
        return tf.reduce_mean(f1)

def Focal_Loss(y_true, y_pred, alpha=0.25, gamma=2):
    """
    focal loss for multi-class classification
    fl(pt) = -alpha*(1-pt)^(gamma)*log(pt)
    :param y_true: ground truth one-hot vector shape of [batch_size, nb_class]
    :param y_pred: prediction after softmax shape of [batch_size, nb_class]
    """
    # To avoid divided by zero
    epsilon = 1e-7
    y_pred += epsilon
    # Cross entropy
    ce = -y_true * tf.math.log(y_pred)
    # Not necessary to multiply y_true(cause it will multiply with CE which has set unconcerned index to zero ),
    # but refer to the definition of p_t, we do it
    weight = tf.math.pow(1 - y_pred, gamma) * y_true
    # Now fl has a shape of [batch_size, nb_class]
    # alpha should be a step function as paper mentioned, but it doesn't matter like reason mentioned above
    # (CE has set unconcerned index to zero)
    # alpha_step = tf.where(y_true, alpha*np.ones_like(y_true), 1-alpha*np.ones_like(y_true))
    fl = ce * weight * alpha
    # Both reduce_sum and reduce_max are ok
    reduce_fl = tf.reduce_max(fl, axis=-1)
    return reduce_fl
   

def create_model(bert_model):
    x1_in = Input(shape=(None,),dtype='int32')
    x2_in = Input(shape=(None,),dtype='int32')
    x = bert_model([x1_in,x2_in])
    L = Lambda(lambda x:x[:,0])(x)
    D1 = Dense(200,activation='relu')(L)
    D2 = Dropout(0.1)(D1)
    D3 = Dense(5,activation='softmax')(D2)
    model = Model(inputs=[x1_in,x2_in],outputs=D3)
    model.compile(loss=Focal_Loss,  
                  optimizer=Adam(2e-5),
                  metrics=[ keras.metrics.Recall(class_id=1),
                            f1_score]
            )
    return model

def predict(content,model,tokenizer):
    label = []
    for sent in content:
        ten_ids = np.asarray(tokenizer.encode(sent)[0])
        seg_ids = np.asarray([0]*len(ten_ids))
        sent = [ten_ids[None,:],seg_ids[None,:]]
        pre_index = model.predict(sent)
        id = np.argmax(pre_index)
        label.append({v:k for k,v in label2id.items()}.get(id))
    return label

def scheduler(epoch,lr):
    if epoch < 3:return lr
    else:return lr * tf.math.exp(-0.1)

def save_model(model,save_path):
    model.save(save_path)

def reverse_model(save_path):
    cust_objs = get_custom_objects()
    cust_objs['f1_score'] = f1_score
    # cust_objs['tfa.metrics.F1Score'] = tfa.metrics.F1Score
    cust_objs['Focal_Loss'] = Focal_Loss
    model = load_model(save_path,custom_objects=cust_objs)
    return model

if __name__=="__main__":
    
    log_dir = os.path.join(os.path.dirname(__file__),'tb_log','logs_fl')
    # 读取data.csv并进行预处理
    train_path = os.path.join(os.path.dirname(__file__),'data','train.pkl')
    valid_path = os.path.join(os.path.dirname(__file__),'data','valid.pkl')
    if not(os.path.exists(train_path)):__call__()

    batch_size = 2
    epochs = 5
    # 加载bert 预训练模型
    bert_model,tokenizer = load_bert()


    train_data = read_corpus(train_path)
    valid_data = read_corpus(valid_path)
    # print(len(train_data),len(valid_data))

    train_gen = DataGenerator(train_data,tokenizer,batch_size=batch_size)
    valid_gen = DataGenerator(valid_data,tokenizer,batch_size=batch_size)
    
    model = create_model(bert_model)
    callback_list = [
                    # tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1),
                    tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=2,restore_best_weights=True),
                    tf.keras.callbacks.LearningRateScheduler(scheduler),
                    # warm_up_lr
                    ]

    model.summary()
    model.fit(
        train_gen.forfit(),
        steps_per_epoch=len(train_gen),
        epochs=epochs,
        validation_data=valid_gen.forfit(),
        validation_steps=len(valid_gen),
        callbacks=callback_list
    )
    
    save_path = os.path.join(os.path.dirname(__file__),model_name)
    save_model(model,save_path)

    # '''predict'''
    # tokenizer = load_bert(just_load_token=True)
    # save_path = os.path.join(os.path.dirname(__file__),model_name)
    # model = reverse_model(save_path)
    # st = time.time()
    # while True:
    #     content = [
    #         '我要举报你们',
    #         '我要求赔偿损失100000元',
    #         '我要举报你们',
    #         '我要求赔偿损失100000元',
    #         '我要举报你们',
    #         '我要求赔偿损失100000元',
    #         '我要举报你们',
    #         '我要求赔偿损失100000元',
    #         '我要举报你们',
    #         '我要求赔偿损失100000元',
    #         '我要举报你们',
    #         '我要求赔偿损失100000元']
    #     label = predict(content,model,tokenizer)
    #     print('预测为本类型为：',label)
    #     ed = time.time()
    #     print('每条文本预测cost time',(ed-st)/len(content))
    #     print('-'*100)
    #     choice = input('是否继续(y/n)').lower()
    #     if choice == 'n':
    #         break
    #     st = time.time()
