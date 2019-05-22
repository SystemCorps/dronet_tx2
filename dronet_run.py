import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
from PIL import Image
import numpy as np
import time


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True

sess = tf.Session(config=config)
set_session(sess)

# Best
json_path = './model/model_struct.json'
weights_path = './model/best_weights.h5'

# For drone control
#json_path = './models2/model_struct.json'
#weights_path = './models2/model_weights_59.h5'

loaded_json = open(json_path, 'r')
loaded_model = loaded_json.read()

model = keras.models.model_from_json(loaded_model)
model.load_weights(weights_path)


# Gray scale image input
img = Image.open('test7.jpg').convert('L')
img = np.array(img.resize((200,200)))
img_np = img.reshape((1,200,200,1))

a = time.time()
outs1 = model.predict(img_np)
print(time.time()-a)

print(outs1)

img2 = Image.open('test8.jpg').convert('L')
img2 = np.array(img2.resize((200,200)))
img_np2 = img2.reshape((1,200,200,1))

a = time.time()
outs2 = model.predict_on_batch(img_np2)
print(time.time()-a)

print(outs2)
print(outs2[0][0], outs2[1][0])

#print(model.summary())