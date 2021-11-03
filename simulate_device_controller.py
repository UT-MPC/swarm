import boto3
import argparse
import json
import numpy as np

import data_process as dp
import get_dataset
import models as custom_models
from dynamo_db import DEVICE_ID, GOAL_DIST, LOCAL_DIST, \
    DATA_INDICES, EVAL_HIST_LOSS, EVAL_HIST_METRIC, ENC_IDX, DEV_STATUS, TIMESTAMPS
from grpc_components.status import STOPPED

# hyperparams for uci dataset
SLIDING_WINDOW_LENGTH = 24
SLIDING_WINDOW_STEP = 12

client = boto3.client('dynamodb', region_name='us-east-1')
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')

def main():
    parser = argparse.ArgumentParser(description='set params for simulation')

    parser.add_argument('--tag', dest='tag',
                    type=str, default='default_tag', help='tag')
    parser.add_argument('--cfg', dest='config_file',
                        type=str, default='configs/dist_swarm/controller_example.json', help='name of the config file')

    parsed = parser.parse_args()
    # load config file
    with open(parsed.config_file, 'rb') as f:
        config_json = f.read()
    config = json.loads(config_json)
    swarm_config = config['swarm_config']

    x_train, y_train_orig, x_test, y_test_orig = get_dataset_from_name(config['dataset'])
    num_classes = len(np.unique(y_train_orig))

    create_table(parsed.tag)

    # store local data dist, goal dist, and trining data indices in the table
    for idnum in range(swarm_config['number_of_devices']):
        label_set = []
        goal_dist = {}
        local_dist = {}
        if "district-9" in config:
            label_set.append(swarm_config["district-9"][idnum % len(swarm_config["district-9"])])
        else:
            label_set = (np.random.choice(np.arange(num_classes), size=swarm_config['local_set_size'], replace=False)).tolist()
        
        for l in label_set:
            local_dist[l] = (int) (swarm_config['local_data_size'] / len(label_set))

        labels_not_in_local_set = np.setdiff1d(np.arange(num_classes), np.array(label_set))
        label_set.extend((np.random.choice(labels_not_in_local_set, 
                                      size=swarm_config['goal_set_size'] - swarm_config['local_set_size'], replace=False)).tolist())
        print(label_set)
        print(local_dist)
        for l in label_set:
            goal_dist[l] = (int) (swarm_config['local_data_size'] / len(label_set))
        print(goal_dist)

        train_data_provider = dp.IndicedDataProvider(x_train, y_train_orig, local_dist)
        chosen_data_idx = train_data_provider.get_chosen()
        table = dynamodb.Table(parsed.tag)
        with table.batch_writer() as batch:
            batch.put_item(Item={DEVICE_ID: idnum, DEV_STATUS: STOPPED, TIMESTAMPS: {},
                GOAL_DIST: convert_to_map(goal_dist),
                LOCAL_DIST: convert_to_map(local_dist), DATA_INDICES: chosen_data_idx,
                EVAL_HIST_LOSS: {}, EVAL_HIST_METRIC: {}, ENC_IDX: -1})

    # deploy workers

def get_dataset_from_name(dataset_name):
    if dataset_name == 'mnist':
        num_classes = 10
        model_fn = custom_models.get_2nn_mnist_model
        x_train, y_train_orig, x_test, y_test_orig = get_dataset.get_mnist_dataset()
    elif dataset_name == 'cifar':
        num_classes = 10
        model_fn = custom_models.get_hetero_cnn_cifar_model
        x_train, y_train_orig, x_test, y_test_orig = get_dataset.get_cifar_dataset()
    elif dataset_name == 'svhn':
        num_classes = 10
        model_fn = custom_models.get_hetero_cnn_cifar_model
        x_train, y_train_orig, x_test, y_test_orig = get_dataset.get_svhn_dataset('data/svhn/')
    elif dataset_name == 'opportunity-uci':
        model_fn = custom_models.get_deep_conv_lstm_model
        x_train, y_train_orig, x_test, y_test_orig = get_dataset.get_opp_uci_dataset('data/opportunity-uci/oppChallenge_gestures.data',
                                                                        SLIDING_WINDOW_LENGTH,
                                                                        SLIDING_WINDOW_STEP)
    return x_train, y_train_orig, x_test, y_test_orig

def create_table(tag):
    # boto3 is the AWS SDK library for Python.
    # We can use the low-level client to make API calls to DynamoDB.

    try:
        resp = client.create_table(
            TableName=tag,
            # Declare your Primary Key in the KeySchema argument
            KeySchema=[
                {
                    "AttributeName": DEVICE_ID,
                    "KeyType": "HASH"
                }
            ],
            # Any attributes used in KeySchema or Indexes must be declared in AttributeDefinitions
            AttributeDefinitions=[
                {
                    "AttributeName": DEVICE_ID,
                    "AttributeType": "N"
                }
            ],
            # ProvisionedThroughput controls the amount of data you can read or write to DynamoDB per second.
            # You can control read and write capacity independently.
            ProvisionedThroughput={
                "ReadCapacityUnits": 1,
                "WriteCapacityUnits": 1
            }
        )
        print("Table created successfully")
    except Exception as e:
        print("Error creating table:")
        print(e)

def convert_to_map(dist):
    new_dist = {}
    for k in dist:
        new_dist[str(k)] = dist[k]
    return new_dist

def convert_from_map(m):
    dist = {}
    for k in m:
        dist[int(m)] = m[k]
    return dist

if __name__ == '__main__':
    main()