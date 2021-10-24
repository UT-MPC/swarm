import sys
sys.path.append("..")
sys.path.append("../..")
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
import matplotlib
import copy
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as plot
import matplotlib.lines as mlines
import tensorflow as tf
import utils
import models as tm
import pickle
from timeit import default_timer as timer

DATA_SIZE = 80
EPOCHS = 1
MODEL_NAME = "two_nn_mnist_model"

num_classes = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test_orig) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model = tm.get_2nn_mnist_model()

compile_config = {'loss': 'mean_squared_error', 'metrics': ['accuracy']}
train_config = {'batch_size': 50, 'shuffle': True}
train_config['epochs'] = 1
train_config['x'] = x_train[:150]
train_config['y'] = y_train[:150]
train_config['verbose'] = 1

with open('data/x_train.pickle', 'wb') as handle:
    pickle.dump(xt, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('data/y_train.pickle', 'wb') as handle:
    pickle.dump(y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(x_train.shape)
print(y_train.shape)

del x_train, y_train

model.save("models/{}.h5".format(MODEL_NAME))

# measure loading from memory and training time
start = timer()

# load model into memory
model = keras.models.load_model('models/{}.h5'.format(MODEL_NAME))
model.compile(**compile_config)

with open('data/x_train.pickle', 'rb') as handle:
    x_train = pickle.load(handle)
with open('data/y_train.pickle', 'rb') as handle:
    y_train = pickle.load(handle)

train_config['epochs'] = EPOCHS
train_config['x'] = x_train
train_config['y'] = y_train
train_config['verbose'] = 0

print(x_train.shape)
print(y_train.shape)

model.fit(**train_config)

end = timer()
print('-------------- Elasped Time --------------')
print(end - start)