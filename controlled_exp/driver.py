import sys
sys.path.insert(0,'..')
import argparse
import datetime
import json
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from clients import get_client_class, LocalClient
import data_process as dp
import models as custom_models
import copy
import pickle
import numpy as np
import boto3
from tqdm import tqdm
from get_dataset import get_mnist_dataset, get_cifar_dataset, get_opp_uci_dataset
import matplotlib.pyplot as plt
import matplotlib
from pathlib import PurePath, Path
from cfg_utils import setup_env, HIST_FOLDER, FIG_FOLDER

# Use truetype fonts for graphs
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# hyperparams for uci dataset
SLIDING_WINDOW_LENGTH = 24
SLIDING_WINDOW_STEP = 12

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

    if config['dataset'] == 'mnist':
        model_fn = custom_models.get_2nn_mnist_model
        x_train, y_train_orig, x_test, y_test_orig = get_mnist_dataset()
        
    elif config['dataset'] == 'cifar':
        model_fn = custom_models.get_big_cnn_cifar_model
        x_train, y_train_orig, x_test, y_test_orig = get_cifar_dataset()

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
    compile_config = {'loss': 'mean_squared_error', 'metrics': ['accuracy']}
    hyperparams = config['hyperparams']
    
    # initialize delegation_clients for simulation
    clients = {}
    i = 0
    for k in config['strategies'].keys():
        if config['strategies'][k]:
            client_class = get_client_class(k)
            if client_class == None:
                print("strategy name {} not defined".format(k))
                return

            train_config = {}
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
        enc_clients = [] # list of 'other' clients our device is encountering
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

            enc_clients.append(
                get_client_class(ck)(k,   # random id
                                    model_fn,
                                    opt_fn,
                                    copy.deepcopy(pretrained_model_weight),
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
                        other = get_client_class(ck)(unique_ids,   # random id
                                                    model_fn,
                                                    opt_fn,
                                                    copy.deepcopy(pretrained_model_weight),
                                                    x_other,
                                                    y_other,
                                                    train_data_provider,
                                                    test_data_provider,
                                                    config['goal-set'],
                                                    compile_config,
                                                    train_config,
                                                    hyperparams)
                        for bn in range(int(config['number-of-data-points']/config['hyperparams']['batch-size'])):
                            clients[ck].delegate(other, 1, 1)
                    else:
                        for bn in range(int(config['number-of-data-points']/config['hyperparams']['batch-size'])):
                            clients[ck].delegate(enc_clients[ii], 1, 1)
                        
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

                    # if i == len(config['intervals'])-1 and j == repeat-1:
                    #     with open('weights/' + ck + '_last_weights.pickle', 'wb') as handle:
                    #         pickle.dump(clients[ck]._weights , handle, protocol=pickle.HIGHEST_PROTOCOL)

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
        
if __name__ == '__main__':
    main()