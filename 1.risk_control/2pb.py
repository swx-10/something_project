from keras import backend as K
from keras_bert import get_custom_objects
from keras.models import load_model
import tensorflow as tf
import os

ori_model_path = os.path.join(os.path.dirname(__file__),'classify_fl.h5')
target_mdoel_path = os.path.join(os.path.dirname(__file__),'pb_model')
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
    epsilon = 1e-7
    y_pred += epsilon
    ce = -y_true * tf.math.log(y_pred)
    weight = tf.math.pow(1 - y_pred, gamma) * y_true
    fl = ce * weight * alpha
    reduce_fl = tf.reduce_max(fl, axis=-1)
    return reduce_fl

cust_objs = get_custom_objects()
cust_objs['f1_score'] = f1_score
cust_objs['Focal_Loss'] = Focal_Loss
model = load_model(ori_model_path,custom_objects=cust_objs)
model.save(target_mdoel_path,save_format='tf')