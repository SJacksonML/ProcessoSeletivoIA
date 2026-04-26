import tensorflow as tf
import os

# Carregando o modelo treinado:

model = tf.keras.models.load_model('model.h5')

# Convertendo o modelo para o formato .tflite:

converter = tf.lite.TFLiteConverter.from_keras_model(model) # Dynamic Range Quantization

converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Salvando o modelo agora otimizado no arquivo 'model.tflite'

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Modelo otimizado.")