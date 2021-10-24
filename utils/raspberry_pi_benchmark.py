import sys
sys.path.insert(0,'..')
import pickle
import tensorflow.keras as keras
import time
import models as custom_models
from get_dataset import get_opp_uci_dataset

# hyperparams for uci dataset
SLIDING_WINDOW_LENGTH = 24
SLIDING_WINDOW_STEP = 12

def main():
    model_fn = custom_models.get_deep_conv_lstm_model
    x_train, y_train_orig, x_test, y_test_orig = get_opp_uci_dataset('../data/opportunity-uci/oppChallenge_gestures.data',
                                                                        SLIDING_WINDOW_LENGTH,
                                                                        SLIDING_WINDOW_STEP)
    y_train = keras.utils.to_categorical(y_train_orig, 18)
    compile_config = {'loss': 'mean_squared_error', 'metrics': ['accuracy']}
    init_model = model_fn()
    init_model.compile(**compile_config)
    pretrain_config = {'batch_size': 50, 'shuffle': True}
    pretrain_config['epochs'] = 1
    pretrain_config['x'] = x_train[:150]
    pretrain_config['y'] = y_train[:150]
    pretrain_config['verbose'] = 1

    time_accum = 0
    ITR = 10
    for i in range(ITR):
        start = time.time()
        print('iteration {}'.format(i))
        with open('../pretrained/deepConvLSTM_pretrained_20p.pickle', 'rb') as handle:
            init_weights = pickle.load(handle)
        init_model.set_weights(init_weights)
        init_model.fit(**pretrain_config)
        with open('temp.pickle', 'wb') as handle:
            pickle.dump(init_model.get_weights(), handle)
        end = time.time()
        time_accum += (end - start)
    
    print('avg time: {}'.format(time_accum/ITR))

if __name__ == '__main__':
    main()
