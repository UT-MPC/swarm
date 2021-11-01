import tensorflow.keras as keras
import pickle
import boto3
from boto3.dynamodb.conditions import Key

import models as custom_models
from get_dataset import get_mnist_dataset, get_cifar_dataset, get_opp_uci_dataset, get_svhn_dataset
from get_device import get_device_class
import data_process as dp
from dynamo_db import DEVICE_ID, GOAL_DIST, LOCAL_DIST, DATA_INDICES

# boto3 is the AWS SDK library for Python.
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('Books')

def async_simulate_device(strategy, config, is_config_file):
    if is_config_file:
        with open(config, 'rb') as f:
            config = f.read()

    # get model and dataset
    if config['dataset'] == 'mnist':
        num_classes = 10
        model_fn = custom_models.get_2nn_mnist_model
        x_train, y_train_orig, x_test, y_test_orig = get_mnist_dataset()       
    elif config['dataset'] == 'cifar':
        num_classes = 10
        model_fn = custom_models.get_hetero_cnn_cifar_model
        x_train, y_train_orig, x_test, y_test_orig = get_cifar_dataset()
    elif config['dataset'] == 'svhn':
        num_classes = 10
        model_fn = custom_models.get_hetero_cnn_cifar_model
        x_train, y_train_orig, x_test, y_test_orig = get_svhn_dataset('data/svhn/')
    elif config['dataset'] == 'opportunity-uci':
        model_fn = custom_models.get_deep_conv_lstm_model
        x_train, y_train_orig, x_test, y_test_orig = get_opp_uci_dataset('data/opportunity-uci/oppChallenge_gestures.data',
                                                                         config['dataset_config']['sliding_window_length'],
                                                                         config['dataset_config']['sliding_window_step'])

    device_class = get_device_class(config['device_strategy'])

    # bootstrap parameters
    if config['device_config']['pretrained_model'] != None:
        with open(config['pretrained-model'], 'rb') as handle:
            init_weights = pickle.load(handle)
    else:
        init_weights = None

    train_data_provider = dp.IndicedDataProvider(x_train, y_train_orig, None)
    test_data_provider = dp.StableTestDataProvider(x_test, y_test_orig, config['hyperparams']['test-data-per-label'])
    resp = table.query(KeyConditionExpression=Key(DEVICE_ID).eq(config['device_config']['id']))
    train_data_provider.chosen = resp['Items']

    device = device_class(config['device_config']['id'],
                          model_fn, 
                          keras.optimizers.Adam,
                          ,
                          ,
                          )

def read_db(key, )