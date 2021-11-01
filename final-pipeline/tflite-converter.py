import tensorflow as tf
import os
from tensorflow.keras.models import load_model

model_path = 'mag-run1.h5'

root_dir = os.path.join(os.path.dirname(__file__), 'master', 'models')
model = load_model(os.path.join(root_dir, model_path))
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model
with open(os.path.join(root_dir, f'{model_path[:-3]}.tflite'), 'wb') as f:
    f.write(tflite_model)