from keras import backend as K
from tensorflow.keras.models import load_model
from keras_bert import get_custom_objects
import tensorflow as tf
import os

ori_model_path = os.path.join(os.path.dirname(__file__),'spannet_NER.h5')
target_mdoel_path = os.path.join(os.path.dirname(__file__),'pb_model')

def my_loss(y_true,y_pred,e = 0.9):
    cce = tf.keras.losses.CategoricalCrossentropy()
    return e*cce(y_true,y_pred) + (1-e)*cce(K.ones_like(y_pred)/11, y_pred)

def reverse_model(save_path):
    cust_obj = get_custom_objects()
    my_objects = {'my_loss':my_loss}
    cust_obj.update(my_objects)
    model = load_model(save_path,custom_objects=cust_obj)
    return model

model = reverse_model(ori_model_path)
model.save(target_mdoel_path,save_format='tf')