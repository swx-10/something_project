# -*- coding:utf-8 -*-

from keras.layers import Layer
import keras.backend as K
import tensorflow as tf


class CRF(Layer):
    """纯Keras实现CRF层
    CRF层本质上是一个带训练参数的loss计算层，因此CRF层只用来训练模型，
    而预测则需要另外建立模型。
    """
    def __init__(self, ignore_last_label=False, num_labels=None, **kwargs):
        """ignore_last_label：定义要不要忽略最后一个标签，起到mask的效果
        """
        super(CRF, self).__init__(**kwargs)
        self.ignore_last_label = 1 if ignore_last_label else 0
        self.num_labels = num_labels
        
    def build(self, input_shape):
        # onehot编码后的标签数量
        self.num_labels = input_shape[-1] - self.ignore_last_label
        # 层中可训练的参数:转移矩阵 [C,C]
        self.trans = self.add_weight(name='crf_trans',
                                     shape=(self.num_labels, self.num_labels),
                                     initializer='glorot_uniform',
                                     trainable=True)
        super(CRF, self).build(input_shape)

    def get_config(self):
        config = super(CRF, self).get_config()
        config.update({"ignore_last_label": True if self.ignore_last_label == 1 else False})
        config.update({'num_labels':self.num_labels})
        return config

    def call(self, inputs): # CRF本身不改变输出，它只是一个loss
        return inputs

    def loss(self, y_true, y_pred): # 目标y_pred需要是one hot形式
        # y_true[S,V,C]  句子,句中词汇,词汇类型编码
        # y_true[:,1:,-1] 句子中从第1个词汇开始的所有词汇onehot最后一位的编码(MASK)
        # 该词是否不需要计算损失，mask[S,V-1]
        mask = 1-y_true[:,1:,-1] if self.ignore_last_label else None
        # 真实类型值与预测类型值
        y_true,y_pred = y_true[:,:,:self.num_labels],y_pred[:,:,:self.num_labels]
        # 句中第0个词汇类型的预测值(初始状态)
        init_states = [y_pred[:,0]]
        # 利用RNN功能函数计算路径得分,返回值: last_output, outputs, new_state
        # https://keras.io/zh/backend/
        log_norm,_,_ = K.rnn(self.log_norm_step, y_pred[:,1:], init_states, mask=mask) # 计算Z向量（对数）
        # 所有路径得分
        log_norm = tf.reduce_logsumexp(log_norm, 1, keepdims=True) # 计算Z（对数）
        path_score = self.path_score(y_pred, y_true) # 计算分子（对数）
        return log_norm - path_score # 即log(分子/分母)
                                     
    def log_norm_step(self, inputs, states):
        """递归计算归一化因子
        要点：1、递归计算；2、用logsumexp避免溢出。
        技巧：通过expand_dims来对齐张量。
        """
        states = K.expand_dims(states[0], 2) # (V, C, 1)
        trans = K.expand_dims(self.trans, 0) # (1, C, C)
        output = tf.reduce_logsumexp(states+trans, 1) # (V, C)
        # 返回output, new_state
        return output+inputs, [output+inputs]

    def path_score(self, inputs, labels):
        """计算目标路径的相对概率（还没有归一化）
        要点：逐标签得分，加上转移概率得分。
        技巧：用“预测”点乘“目标”的方法抽取出目标路径的得分。
        """
        # emission score: 预测的编码概率 * 真实编码,求和,统计每个词汇的概率得分。 
        point_score = K.sum(K.sum(inputs*labels, 2), 1, keepdims=True) # 逐标签得分
        # transition score: 真实编码错位计算得到转移矩阵
        labels1 = K.expand_dims(labels[:,:-1], 3)   # [S,V,C,1] V=(0,…,n-1)
        labels2 = K.expand_dims(labels[:,1:], 2)    # [S,V,1,C] V=(1,…,n)
        labels = labels1 * labels2 # 两个错位labels，负责从转移矩阵中抽取目标转移得分
        # 参数矩阵扩维 [1,1,C,C]
        trans = K.expand_dims(K.expand_dims(self.trans, 0), 0)
        # 转移矩阵求编码转移概率得分
        trans_score = K.sum(K.sum(trans * labels, [2,3]), 1, keepdims=True)
        return point_score + trans_score # 两部分得分之和

    def accuracy(self, y_true, y_pred): # 训练过程中显示逐帧准确率的函数，排除了mask的影响
        mask = 1-y_true[:,:,-1] if self.ignore_last_label else None
        y_true,y_pred = y_true[:,:,:self.num_labels],y_pred[:,:,:self.num_labels]
        isequal = K.equal(K.argmax(y_true, 2), K.argmax(y_pred, 2))
        isequal = K.cast(isequal, 'float32')
        if mask == None:
            return K.mean(isequal)
        else:
            return K.sum(isequal * mask) / K.sum(mask)