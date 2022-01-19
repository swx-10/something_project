from keras import backend as K
from keras_bert import get_custom_objects
import tensorflow as tf
import os

ori_model_path = os.path.join(os.path.dirname(__file__),'model','entities_pre.h5')
target_mdoel_path = os.path.join(os.path.dirname(__file__),'pb_model')
def micro_f1(y_pred, y_true, model='multi'):
    epsilon = 1e-7
    axis = 2
    y_pred = tf.round(y_pred)#将经过sigmoid激活的张量四舍五入变为0，1输出
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
    y_pred += K.epsilon()
    ce = -y_true * K.log(y_pred)
    weight = K.pow(1 - y_pred, gamma) * y_true
    fl = ce * weight * alpha
    reduce_fl = K.max(fl, axis=-1)
    return reduce_fl

def reload_model():
    obj = get_custom_objects()
    my_obj = {'Focal_Loss':Focal_Loss,
            'micro_f1':micro_f1}
    obj.update(my_obj)
    model = tf.keras.models.load_model(ori_model_path,custom_objects=obj)
    return model
model = reload_model()
model.save(target_mdoel_path,save_format='tf')