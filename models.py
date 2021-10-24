import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf
# from qkeras import QDense, quantized_bits, QActivation

# hyperparams for uci dataset
NUM_FILTERS = 64
FILTER_SIZE = 5
SLIDING_WINDOW_LENGTH = 24
NB_SENSOR_CHANNELS = 113
NUM_UNITS_LSTM = 128
NUM_CLASSES = 18


def get_2nn_mnist_model(compressed_ver=0, size=10):
    if compressed_ver == 1:
        return get_compressed_2nn_mnist_model()
    elif compressed_ver == 2:
        return get_v2_compressed_2nn_mnist_model()
    model = Sequential()
    model.add(Flatten(input_shape=(28,28,1)))
    model.add(Dense(10 * size, activation='relu', name='dense_0'))
    model.add(Dense(10 * size, activation='relu', name='dense_1'))
    model.add(Dense(10, activation='softmax', name='softmax_logits'))
    return model

def get_2nn_svhn_model(size=10):
    model = Sequential()
    model.add(Flatten(input_shape=(32,32,3)))
    model.add(Dense(int(100 - (10 - size) * 0.4), activation='relu', name='dense_0'))
    model.add(Dense(10 * size, activation='relu', name='dense_1'))
    model.add(Dense(10, activation='softmax', name='softmax_logits'))
    return model

# def get_Q_2nn_mnist_model(size=10):
#     bits = 4
#     integer = 0
#     symmetric = 1
#     params = '{},{},{}'.format(bits, integer, symmetric)
#     model = Sequential()
#     model.add(Flatten(input_shape=(28,28,1)))
#     model.add(QDense(5 * size, 
#                     kernel_quantizer=quantized_bits(bits=bits, integer=integer, symmetric=symmetric),
#                     bias_quantizer=quantized_bits(bits=bits, integer=integer, symmetric=symmetric),
#                     activation='quantized_relu(4,0,1)'))
#     model.add(QDense(5 * size, 
#                     kernel_quantizer=quantized_bits(bits=bits, integer=integer, symmetric=symmetric),
#                     bias_quantizer=quantized_bits(bits=bits, integer=integer, symmetric=symmetric),
#                     activation='quantized_relu(4,0,1)'))
#     model.add(QDense(10, 
#                     kernel_quantizer=quantized_bits(bits=bits, integer=integer, symmetric=symmetric),
#                     bias_quantizer=quantized_bits(bits=bits, integer=integer, symmetric=symmetric)))
#     model.add(Activation("softmax"))
#     return model

# def get_Q_2nn_mnist_dict(size=10):
#     q_dict = {
#         "QDense": {
#             "kernel_quantizer": "quantized_bits(4,0,1)",
#             "bias_quantizer": "quantized_bits(4,0,1)"
#         }
#     }
#     return q_dict

def get_2nn_mnist_model_distill():
    model = Sequential()
    model.add(Flatten(input_shape=(28,28,1)))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(10, activation=None))
    return model

def get_paramed_2nn_mnist_model(num_neurons=200):
    model = Sequential()
    model.add(Flatten(input_shape=(28,28,1)))
    model.add(Dense(num_neurons, activation='relu'))
    model.add(Dense(num_neurons, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

def get_compressed_2nn_mnist_model():
    model = Sequential()
    model.add(Flatten(input_shape=(28,28,1)))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

def get_v2_compressed_2nn_mnist_model():
    model = Sequential()
    model.add(Flatten(input_shape=(28,28,1)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

def get_cifar_model(size=3):
    if size == 1:
        return get_2nn_cifar_model()
    elif size == 3:
        return get_cnn_cifar_model()
    elif size == 6:
        return get_big_cnn_cifar_model()
    
    return get_big_cnn_cifar_model()

def get_2nn_cifar_model():
    model = Sequential()
    model.add(Flatten(input_shape=(32,32,3)))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

def get_cnn_mnist_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    return model

def get_cnn_cifar_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

def get_bin_cnn_cifar_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return model

def get_big_cnn_cifar_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                    input_shape=(32,32,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model

def get_all_cnn_c_model(size=10):
    model = Sequential()
    dr = {5: 0.5, 6: 0.625, 7: 0.750, 8: 0.875, 9: 0.875, 10: 1}
    if size not in dr:
        raise ValueError('CNN model does not support model size {}'.format(size))
    small_conv_fs = 96 * dr[size]
    big_conv_fs = 192 * dr[size]

    model.add(Conv2D(96, (3, 3), padding = 'same', input_shape=(32, 32, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(small_conv_fs, (3, 3),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(small_conv_fs, (3, 3), padding='same', strides = (2,2)))

    model.add(Conv2D(big_conv_fs, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(big_conv_fs, (3, 3),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(big_conv_fs, (3, 3),padding='same', strides = (2,2)))

    model.add(Conv2D(192, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (1, 1),padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(10, (1, 1), padding='valid'))

    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))
    return model

def get_hetero_cnn_cifar_model(size = 3):
    model = Sequential()
    if size >= 3:
        model.add(Conv2D(32, (3, 3), padding='same',
                        input_shape=(32,32,3)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(10))
        model.add(Activation('softmax'))
        
    elif size == 2:
        model.add(Conv2D(32, (3, 3), padding='same',
                        input_shape=(32,32,3)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(48, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(48, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(10))
        model.add(Activation('softmax'))
        
    elif size == 1:
        model.add(Conv2D(32, (3, 3), padding='same',
                        input_shape=(32,32,3)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(10))
        model.add(Activation('softmax'))
    else:
        raise ValueError("size {} is not supported cnn model size".format(size))
    
    return model

def get_deep_hetero_cnn_cifar_model(size = 3):
    model = Sequential()
    if size >= 3:
        model.add(Conv2D(32, (3, 3), padding='same',
                        input_shape=(32,32,3)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(10))
        model.add(Activation('softmax'))
        
    elif size == 2:
        model.add(Conv2D(32, (3, 3), padding='same',
                        input_shape=(32,32,3)))
        model.add(Activation('relu'))
        model.add(Conv2D(16, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(10))
        model.add(Activation('softmax'))
        
    elif size == 1:
        model.add(Conv2D(32, (3, 3), padding='same',
                        input_shape=(32,32,3)))
        model.add(Activation('relu'))
        model.add(Conv2D(16, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(16, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(16, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(10))
        model.add(Activation('softmax'))
    else:
        raise ValueError("size {} is not supported cnn model size".format(size))
    
    return model

def get_big_bin_cnn_cifar_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                    input_shape=(32,32,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model

def get_big_quad_cnn_cifar_model(compressed_ver=0):
    if compressed_ver == 1:
        return get_compressed_big_quad_cnn_cifar_model()
    elif compressed_ver == 2:
        return get_v2_compressed_big_quad_cnn_cifar_model()
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                    input_shape=(32,32,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4))
    model.add(Activation('softmax'))
    return model

def get_compressed_big_quad_cnn_cifar_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                    input_shape=(32,32,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4))
    model.add(Activation('softmax'))
    return model

def get_v2_compressed_big_quad_cnn_cifar_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                    input_shape=(32,32,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4))
    model.add(Activation('softmax'))
    return model

def get_better_cnn_cifar_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model

def get_deep_conv_lstm_model(num_filters=NUM_FILTERS,
                             filter_size=FILTER_SIZE,
                             sliding_window_length=SLIDING_WINDOW_LENGTH,
                             nb_sensor_channels=NB_SENSOR_CHANNELS,
                             num_units_lstm=NUM_UNITS_LSTM,
                             num_classes=NUM_CLASSES):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(num_filters, (filter_size, 1), activation='relu',
                                  input_shape=(sliding_window_length, nb_sensor_channels, 1)))
    model.add(keras.layers.Conv2D(num_filters, (filter_size, 1), activation='relu'))
    model.add(keras.layers.Conv2D(num_filters, (filter_size, 1), activation='relu'))
    model.add(keras.layers.Conv2D(num_filters, (filter_size, 1), activation='relu'))
    shape = model.layers[-1].output_shape
    model.add(keras.layers.Reshape((shape[1], shape[3] * shape[2])))
    model.add(keras.layers.LSTM(num_units_lstm, activation='tanh', return_sequences=True)) # [batch, timesteps, features]
    model.add(keras.layers.Dropout(0.5, seed=123))
    model.add(keras.layers.LSTM(num_units_lstm, activation='tanh'))
    model.add(keras.layers.Dropout(0.5, seed=124))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    return model
