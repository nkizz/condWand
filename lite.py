from tensorflow import lite
import tensorflow as tf
import numpy as np
model = tf.keras.models.load_model('lstm5.h5')
converter = lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter=True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]

tfmodel = converter.convert()
open('lstm5.tflite', 'wb').write(tfmodel)