import sys
sys.path.insert(0,'..')
import argparse
import datetime
import json
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
import copy
import pickle
import numpy as np
import boto3
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from pathlib import PurePath, Path

from get_model import get_model_fn
from get_optimizer import get_optimizer
from get_dataset import get_mnist_dataset, get_cifar_dataset, get_svhn_dataset, get_opp_uci_dataset
from get_device import get_device_class

from device.exp_device import LocalDevice
import data_process as dp
import models as custom_models
from cfg_utils import setup_env, HIST_FOLDER, FIG_FOLDER

# Use truetype fonts for graphs
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# hyperparams for uci dataset
SLIDING_WINDOW_LENGTH = 24
SLIDING_WINDOW_STEP = 12

# strategy names for knowledge distillation
KD_STRATEGIES = ["only data", "only model", "data model half half"]

def main():
    """
    driver for running controlled experiment from one device's perspective
    """
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

    setup_env()

    # parse arguments
    parser = argparse.ArgumentParser(description='set params for controlled experiment')
    parser.add_argument('--tag', dest='tag',
                        type=str, default='default_tag', help='tag for name of the output')
    parser.add_argument('--cfg', dest='config_file',
                        type=str, default='configs/mnist_cfg.json', help='name of the config file')
    parser.add_argument('--seed', dest='seed',
                    type=int, default=0, help='use pretrained weights')
    parser.add_argument('--global', dest='global',
                    type=int, default=0, help='report accuracies of global models for greedy-sim and greedy-no-sim')     

    parsed = parser.parse_args()

    if parsed.config_file == None or parsed.tag == None:
        print('Config file and output diretory has to be specified. Run \'python driver.py -h\' for help/.')

    HIST_FILE_PATH = PurePath(HIST_FOLDER, parsed.tag + '.pickle')
    GRAPH_FILE_PATH = PurePath(FIG_FOLDER, parsed.tag + '.pdf')

    np.random.seed(parsed.seed)
    tf.compat.v1.set_random_seed(parsed.seed)

    # load config file
    with open(parsed.config_file, 'rb') as f:
        config_json = f.read()
    config = json.loads(config_json)
    hyperparams = config['hyperparams']

    if config['dataset'] == 'mnist':
        model_fn = custom_models.get_2nn_mnist_model
        x_train, y_train_orig, x_test, y_test_orig = get_mnist_dataset()
        
    elif config['dataset'] == 'cifar':
        if 'strategies' in config and ('hetero' in config['strategies'] or 'dropoutvar' in config['strategies']):
            model_fn = custom_models.get_hetero_cnn_cifar_model
        else:
            model_fn = custom_models.get_hetero_cnn_cifar_model
        x_train, y_train_orig, x_test, y_test_orig = get_cifar_dataset()
    
    elif config['dataset'] == 'svhn':
        model_fn = custom_models.get_hetero_cnn_cifar_model
        x_train, y_train_orig, x_test, y_test_orig = get_svhn_dataset('../data/svhn/')

    elif config['dataset'] == 'opportunity-uci':
        model_fn = custom_models.get_deep_conv_lstm_model
        x_train, y_train_orig, x_test, y_test_orig = get_opp_uci_dataset('../data/opportunity-uci/oppChallenge_gestures.data',
                                                                         SLIDING_WINDOW_LENGTH,
                                                                         SLIDING_WINDOW_STEP)

    train_data_provider = dp.DataProvider(x_train, y_train_orig)
    test_data_provider = dp.StableTestDataProvider(x_test, y_test_orig, config['test-data-per-label'])

    # get local dataset for clients
    client_label_conf = {}
    for l in config['local-set']:
        client_label_conf[l] = (int) (config['number-of-data-points']/len(config['local-set']))
    x_train_client, y_train_client = train_data_provider.peek(client_label_conf)

    # get pretrained model
    with open(config['pretrained-model'], 'rb') as handle:
        pretrained_model_weight = pickle.load(handle)

    # set params for building clients
    opt_fn = keras.optimizers.SGD
    compile_config = {'loss': 'categorical_crossentropy', 'metrics': ['accuracy']}
    hyperparams['dataset'] = config['dataset']

    # get public dataset for KD simulations
    if 'public-dataset-size-per-label' in config:
        client_label_conf = {}
        for l in config['goal-set']:
            client_label_conf[l] = (int) (config['public-dataset-size-per-label'])
        x_public, y_public_orig = train_data_provider.peek(client_label_conf)
        # y_public = keras.utils.to_categorical(y_public_orig, len(np.unique(y_train_orig)))
        hyperparams['x_public'] = x_public
        hyperparams['y_public'] = y_public_orig

    if 'device-power-interval' in config:
        hyperparams['device-power-interval'] = config['device-power-interval']
        
    # initialize learners
    clients = {}
    strategy_types = {}
    i = 0
    if 'strategy-variants' in config:
        strategy_variants = config['strategy-variants']

    init_models_by_size = {}

    for k in config['strategies'].keys():
        if config['strategies'][k]:
            client_class = get_device_class(k)
            if client_class == None:
                raise ValueError("strategy name {} not defined".format(k))
            train_config = {}
            model_fn_tmp = None

            # @ TODO hyperparams and model_fn set in this loop does not set back to its original
            if k in strategy_variants:
                for var in strategy_variants[k]:
                    is_downto = False
                    this_hyperparams = copy.deepcopy(hyperparams)
                    if "model-size" in var:
                        this_hyperparams['model-size'] = var["model-size"]
                    else:
                        this_hyperparams['model-size'] = 20 #@TODO this forces to set model size to any device
                    if "downto" in var:
                        this_hyperparams['downto'] = var["downto"]
                    if "optimizer" in var:
                        opt_fn = get_optimizer(var["optimizer"])
                        this_hyperparams['optimizer-params'] = var["optimizer-params"]
                        if "learning_rate" in var["optimizer-params"]:
                            this_hyperparams['orig-lr'] = var["optimizer-params"]["learning_rate"]
                    if "bits" in var:
                        this_hyperparams['bits'] = var["bits"]
                    if "model-bits" in var:
                        this_hyperparams['model-bits'] = var["model-bits"]
                    if "model" in var:
                        this_hyperparams['q-model'] = var['q-model']
                    if "data-size" in var:
                        this_hyperparams['data-size'] = var['data-size']

                    this_hyperparams['repeated-number'] = 0

                    xt, yt_orig = test_data_provider.fetch(config['goal-set'], config['test-data-per-label'])
                    yt = keras.utils.to_categorical(yt_orig, len(np.unique(y_train_orig)))
                    
                    model_name = model_fn.__name__.split('_')[1]
                    weights_file_name = '../pretrained/init_weights_{}_{}.pickle'.format(model_name, this_hyperparams['model-size'])
                    try:
                        with open(weights_file_name, 'rb') as handle:
                            pretrained_model_weight = pickle.load(handle)
                    except:
                        pretrained_model_weight = _get_init_model_with_accuracy(model_fn=model_fn, args={'size': this_hyperparams['model-size']}, 
                                                                            compile_config=compile_config, x_test=xt, y_test=yt, 
                                                                            acc=0.1, error=0.04)
                        with open(weights_file_name, 'wb') as handle:
                            pickle.dump(pretrained_model_weight, handle)

                    c = client_class(i,
                                model_fn,
                                opt_fn,
                                copy.deepcopy(pretrained_model_weight),
                                x_train_client,
                                y_train_client,
                                train_data_provider,
                                test_data_provider,
                                config['goal-set'],
                                compile_config,
                                train_config,
                                this_hyperparams)

                    client_name = k
                    if 'downto' in var:
                        client_name += '-lim-' + str(this_hyperparams['downto'])
                    if 'model-size' in var:
                        client_name += '-[model: '
                        if 'model' in var:
                            client_name += var['model']
                        client_name += ', size: ' + str(this_hyperparams['model-size']) + ']'
                    if 'optimizer' in var:
                        client_name += '[opt: ' + var["optimizer"] +']'
                    if 'bits' in var:
                        client_name += '[bits: ' + str(var['bits']) + ']'

                    clients[client_name] = c
                    i += 1
            else:
                opt_fn = keras.optimizers.SGD
                c = client_class(i,
                                model_fn,
                                opt_fn,
                                copy.deepcopy(pretrained_model_weight),
                                x_train_client,
                                y_train_client,
                                train_data_provider,
                                test_data_provider,
                                config['goal-set'],
                                compile_config,
                                train_config,
                                hyperparams)
                clients[k] = c
                i += 1

            if model_fn_tmp != None:
                model_fn = model_fn_tmp

    # for hetero exp, set model size of encountered clients. these are not used
    hyperparams['strategy-type'] = 1
    
    # initialize logs
    logs = {}
    for ck in clients.keys():
        logs[ck] = {}
        hist = clients[ck].eval()
        if config['hyperparams']['evaluation-metrics'] == 'loss-and-accuracy':
            logs[ck]['accuracy'] = []
            logs[ck]['loss'] = []
            logs[ck]['loss'].append(hist[0])
            logs[ck]['accuracy'].append(hist[1])
        elif config['hyperparams']['evaluation-metrics'] == 'f1-score-weighted':
            logs[ck]['f1-score'] = []
            logs[ck]['f1-score'].append(hist)
        elif config['hyperparams']['evaluation-metrics'] == 'split-f1-score-weighted':
            for labels in config['hyperparams']['split-test-labels']:
                logs[ck]['f1: ' + str(labels)] = []
                logs[ck]['f1: ' + str(labels)].append(hist[str(labels)])
        else:
            ValueError('invalid evaluation-metrics: {}'.format(config['hyperparams']['evaluation-metrics']))
        
        if 'dropout' in ck:
            logs[ck]['comm-cost'] = []
            logs[ck]['comp-cost'] = []
    
    if config['dataset'] == 'opportunity-uci':
        candidates = np.arange(0, 18)
    else:
        candidates = np.arange(0,10)
    if len(config['intervals']) != len(config['label-sets']):
        raise ValueError('length of intervals and label-sets should be the same: {} != {}'.format(config['intervals'], config['label-sets']))
    
    try:
        repeat = config['repeat']
    except:
        repeat = 1

    try:
        same_repeat = config['same-repeat']
    except:
        same_repeat = False

    if same_repeat:
        enc_clients = {} # list of 'other' clients our device is encountering
        # @TODO used to have enc_clients for all strategies. when does this work?
        enc_clients = []
        one_cycle_length = 0
        for i in range(len(config['intervals'])):
            one_cycle_length += config['intervals'][i]
        for k in range(one_cycle_length):
            # set labels
            rn = np.random.rand(1)
            if rn > config['noise-percentage']/100.0:  # not noise
                local_labels = config['label-sets'][i]
            else:   # noise
                np.random.shuffle(candidates)
                local_labels = copy.deepcopy(candidates[:config['noise-label-set-size']])
            label_conf = {}
            for ll in local_labels:
                label_conf[ll] = (int) (config['number-of-data-points']/len(local_labels))

            # print(label_conf)
            x_other, y_other = train_data_provider.peek(label_conf)
            if 'model-size' in clients[ck]._hyperparams:
                hyperparams['model-size'] = clients[ck]._hyperparams['model-size']
            hyperparams['repeated-number'] = k
            hyperparams['bits'] = 8
            enc_clients.append(
                get_device_class(_get_strategy_name(ck))(k,   # random id
                                    model_fn,
                                    opt_fn,
                                    0,
                                    x_other,
                                    y_other,
                                    train_data_provider,
                                    test_data_provider,
                                    config['goal-set'],
                                    compile_config,
                                    train_config,
                                    hyperparams)
            )

    unique_ids = 10000
    for j in range(repeat):
        ii = -1
        for i in range(len(config['intervals'])):
            print('simulating range {} of {} in repetition {} of {}'.format(i+1, len(config['intervals']), j+1, repeat))
            for _ in tqdm(range(config['intervals'][i])):
                # set labels
                rn = np.random.rand(1)
                if rn > config['noise-percentage']/100.0:  # not noise
                    local_labels = config['label-sets'][i]
                else:   # noise
                    np.random.shuffle(candidates)
                    local_labels = copy.deepcopy(candidates[:config['noise-label-set-size']])
                label_conf = {}
                for ll in local_labels:
                    label_conf[ll] = (int) (config['number-of-data-points']/len(local_labels))

                # print(label_conf)
                x_other, y_other = train_data_provider.peek(label_conf)

                # run for different approaches: local, greedy, ...
                ii += 1
                unique_ids += 1
                for ck in clients.keys():
                    if not same_repeat:
                        hyperparams['model-size'] = clients[ck]._hyperparams['model-size']
                        hyperparams['repeated-number'] = ii
                        hyperparams['optimizer-params'] = {}
                        other = get_device_class(_get_strategy_name(ck))(unique_ids,   # random id
                                                    model_fn,
                                                    opt_fn,
                                                    0,
                                                    x_other,
                                                    y_other,
                                                    train_data_provider,
                                                    test_data_provider,
                                                    config['goal-set'],
                                                    compile_config,
                                                    train_config,
                                                    hyperparams)
                        rho = int(config['number-of-data-points']/config['hyperparams']['batch-size'])
                        if (_get_strategy_name(ck) in KD_STRATEGIES):
                            rho = 1
                        # for bn in range(rho):
                        clients[ck].delegate(other, 1, rho)
                    else:
                        # for bn in range(int(config['number-of-data-points']/config['hyperparams']['batch-size'])):
                        iterations = int(config['number-of-data-points']/config['hyperparams']['batch-size'])
                        clients[ck].delegate(enc_clients[ii], 1, iterations)
                        
                    hist = clients[ck].eval()
                    if config['hyperparams']['evaluation-metrics'] == 'loss-and-accuracy':
                        logs[ck]['loss'].append(hist[0])
                        logs[ck]['accuracy'].append(hist[1])
                    elif config['hyperparams']['evaluation-metrics'] == 'f1-score-weighted':
                        logs[ck]['f1-score'].append(hist)
                    elif config['hyperparams']['evaluation-metrics'] == 'split-f1-score-weighted':
                        for labels in config['hyperparams']['split-test-labels']:
                            logs[ck]['f1: ' + str(labels)].append(hist[str(labels)])
                    else:
                        ValueError('invalid evaluation-metrics: {}'.format(config['hyperparams']['evaluation-metrics']))
                    if 'dropout' in ck:
                        logs[ck]['comm-cost'].append(hist[2])
                        logs[ck]['comp-cost'].append(hist[3])
                    # if i == len(config['intervals'])-1 and j == repeat-1:
                    #     with open('weights/' + ck + '_last_weights.pickle', 'wb') as handle:
                    #         pickle.dump(clients[ck]._weights , handle, protocol=pickle.HIGHEST_PROTOCOL)

    for c in clients.keys():
        print(c)
        if hasattr(clients[c], 'requests'):
            logs[c]['requests'] = clients[c].requests
            print(clients[c].requests)
        if hasattr(clients[c], 'q_requests'):
            logs[c]['q_requests'] = clients[c].q_requests
            print(clients[c].q_requests)


    with open(HIST_FILE_PATH, 'wb') as handle:
        pickle.dump(logs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # draw graph
    if config['hyperparams']['evaluation-metrics'] == 'split-f1-score-weighted':
        for k in logs.keys():
            for labels in logs[k].keys():
                plt.plot(np.arange(0, len(logs[k][labels])), np.array(logs[k][labels]), lw=1.2)
            plt.legend(list(logs[k].keys()))
            plt.ylabel('F1-score')
            plt.xlabel("Encounters")
            plt.savefig(GRAPH_FILE_PATH)
            plt.close()
        return

    for ck in clients.keys():
        if 'var' in ck:
            print('{}: {}'.format(ck, clients[ck].requests))

    if config['hyperparams']['evaluation-metrics'] == 'loss-and-accuracy':
        key = 'accuracy'
    elif config['hyperparams']['evaluation-metrics'] == 'f1-score-weighted':
        key = 'f1-score'
    
    for k in logs.keys():
        plt.plot(np.arange(0, len(logs[k][key])), np.array(logs[k][key]), lw=1.2)
    plt.legend(list(logs.keys()))
    if key == 'accuracy':
        y_label = 'Accuracy'
    elif key == 'f1-score':
        y_label = 'F1-score'
    # plt.ylim(0.9, 0.940)
    # plt.title(parsed.graph_file)
    plt.ylabel(y_label)
    plt.xlabel("Encounters")
    plt.savefig(GRAPH_FILE_PATH)
    plt.close()

    # upload to S3 storage
    client = boto3.client('s3')
    S3_BUCKET_NAME = 'opfl-sim-models'
    FOLDER_NAME = 'controlled_exps'
    upload_hist_path = PurePath(FOLDER_NAME, 'hists/' + parsed.tag + '.pickle')
    client.upload_file(str(HIST_FILE_PATH), S3_BUCKET_NAME, str(upload_hist_path))
    upload_graph_path = PurePath(FOLDER_NAME, 'figs/' + parsed.tag + '.pdf')
    client.upload_file(str(GRAPH_FILE_PATH), S3_BUCKET_NAME, str(upload_graph_path))

def _get_strategy_name(strategy_type_name):
    return strategy_type_name.split('-')[0]

def _get_init_model_with_accuracy(model_fn, args, compile_config, x_test, y_test, acc, error):
    TRY_LIMIT = 40
    print('initializing models...')
    for i in range(TRY_LIMIT):
        m = model_fn(**args)
        m.compile(**compile_config)
        hist = m.evaluate(x_test, y_test)
        if np.abs(hist[1] - acc) <= error:
            print('got init model for [{}] in {} trials.'.format(args, i))
            return m.get_weights()
    raise RuntimeError('Failed to get initial weights')
        
if __name__ == '__main__':
    main()