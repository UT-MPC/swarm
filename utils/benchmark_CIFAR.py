import sys
sys.path.insert(0,'..')
import pickle
import tensorflow.keras as keras
import time
import models as custom_models
import tensorflow as tf
from tensorflow.keras import backend as K
from model_weight_utils import *

def main():
    model_fn = custom_models.get_big_cnn_cifar_model
    # CIFAR 10
    # input image dimensions
    img_rows, img_cols = 32, 32

    # the data, split between train and test sets
    (x_train, y_train_orig), (x_test, y_test_orig) = tf.keras.datasets.cifar10.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
        input_shape = (3, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train_orig, 10)
    compile_config = {'loss': 'mean_squared_error', 'metrics': ['accuracy']}
    init_model = model_fn()
    init_model.compile(**compile_config)
    pretrain_config = {'batch_size': 50, 'shuffle': True}
    pretrain_config['epochs'] = 1
    pretrain_config['x'] = x_train[:50]
    pretrain_config['y'] = y_train[:50]
    pretrain_config['verbose'] = 0

    time_accum = 0
    ITR = 10
    for i in range(ITR):
        start = time.time()
        print('iteration {}'.format(i))
        with open('../pretrained/pretrained_model_big_cnn_cifar_local_updates_epochs_100_data_20000_preprocess.pickle', 'rb') as handle:
            init_weights = pickle.load(handle)
        init_model.set_weights(init_weights)
        init_model.fit(**pretrain_config)
        with open('temp.pickle', 'wb') as handle:
            pickle.dump(init_model.get_weights(), handle)
        end = time.time()
        time_accum += (end - start)
    
    print('avg time: {}'.format(time_accum/ITR))

    # init_model.save('temp.h5')

    # time_accum = 0
    # for i in range(ITR):
    #     start = time.time()
    #     model = keras.models.load_model('temp.h5')
    #     end = time.time()
    #     time_accum += end - start
    
    # print('avg loading time: {}'.format(time_accum/ITR))

    time_accum = 0
    with open('../pretrained/pretrained_model_big_cnn_cifar_local_updates_epochs_100_data_20000_preprocess.pickle', 'rb') as handle:
        init_weights = pickle.load(handle)
    for i in range(ITR):
        start = time.time()
        print('iteration {}'.format(i))
        add_weights(init_weights, init_weights)
        end = time.time()
        time_accum += (end - start)

    print('avg time for adding: {}'.format(time_accum/ITR))

if __name__ == '__main__':
    main()
