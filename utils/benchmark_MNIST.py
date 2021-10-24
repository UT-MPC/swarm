import sys
sys.path.insert(0,'..')
import pickle
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
import time
import models as custom_models
from tensorflow.keras import backend as K
from model_weight_utils import *

def main():
    model_fn = custom_models.get_2nn_mnist_model
    
    # get dataset
    num_classes = 10

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train_orig), (x_test, y_test_orig) = mnist.load_data()

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
        with open('../pretrained/test_base_2nn_mnist.pickle', 'rb') as handle:
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

    # time_accum = 0
    # for i in range(ITR):
    #     start = time.time()
    #     with open('../pretrained/test_base_2nn_mnist.pickle', 'rb') as handle:
    #         init_weights = pickle.load(handle)
    #     end = time.time()
    #     time_accum += end - start

    with open('../pretrained/test_base_2nn_mnist.pickle', 'rb') as handle:
        init_weights = pickle.load(handle)
    time_accum = 0
    for i in range(ITR):
        start = time.time()
        print('iteration {}'.format(i))
        add_weights(init_weights, init_weights)
        end = time.time()
        time_accum += (end - start)

    print('avg time for adding: {}'.format(time_accum/ITR))

if __name__ == '__main__':
    main()
