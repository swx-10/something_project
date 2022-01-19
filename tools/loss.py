def my_loss(y_true,y_pred,e = 0.9):
    cce = tf.keras.losses.CategoricalCrossentropy()
    return e*cce(y_true,y_pred) + (1-e)*cce(K.ones_like(y_pred)/11, y_pred)