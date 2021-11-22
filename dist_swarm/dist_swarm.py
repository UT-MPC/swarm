import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../grpc_components')
import boto3
import grpc
import argparse
import json
import numpy as np
import time

import data_process as dp
from get_dataset import get_dataset
import models as custom_models
from dynamo_db import DEVICE_ID, GOAL_DIST, LOCAL_DIST, \
    DATA_INDICES, EVAL_HIST_LOSS, EVAL_HIST_METRIC, ENC_IDX, DEV_STATUS, TIMESTAMPS, ERROR_TRACE
from grpc_components.status import STOPPED
from grpc_components import simulate_device_pb2, simulate_device_pb2_grpc
from aws_settings import REGION

client = boto3.client('dynamodb', region_name=REGION)
dynamodb = boto3.resource('dynamodb', region_name=REGION)

class DistSwarm():
    def __init__(self, config_file, tag, ip) -> None:
        self._config_db(config_file, tag)
        self._config_client(ip)
        
    # deploy and run workers
    def run(self):
        print('number of devices: {}'.format(self.config['swarm_config']['number_of_devices']))
        for device_id in range(int(self.config['swarm_config']['number_of_devices'])):
            self.config['device_config']['id'] = device_id
            config = simulate_device_pb2.Config(config=json.dumps(self.config))
            with grpc.insecure_channel(self.loadbalancer_ip) as channel:
                stub = simulate_device_pb2_grpc.SimulateDeviceStub(channel)
                status = stub.SimulateOppCL(config)
                print('device {}: {}'.format(device_id, status))

    def _create_table(self, tag):
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
            print("Table created successfully. Syncing...")
            time.sleep(3)

        except Exception as e:
            print("Error creating table:")
            print(e)


    def _config_client(self, ip):
        self.loadbalancer_ip = ip

    def _config_db(self, config_file, tag):
        # load config file
        with open(config_file, 'rb') as f:
            config_json = f.read()
        config = json.loads(config_json)
        self.config = config
        swarm_config = config['swarm_config']

        x_train, y_train_orig, x_test, y_test_orig = get_dataset(config['dataset'])
        num_classes = len(np.unique(y_train_orig))

        self._create_table(tag)

        # store local data dist, goal dist, and trining data indices in the table
        for idnum in range(swarm_config['number_of_devices']):
            print('init db for device {}.'.format(idnum))
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
            table = dynamodb.Table(tag)
            with table.batch_writer() as batch:
                batch.put_item(Item={DEVICE_ID: idnum, DEV_STATUS: STOPPED, TIMESTAMPS: {},
                    GOAL_DIST: convert_to_map(goal_dist),
                    LOCAL_DIST: convert_to_map(local_dist), DATA_INDICES: chosen_data_idx,
                    EVAL_HIST_LOSS: {}, EVAL_HIST_METRIC: {}, ENC_IDX: -1, ERROR_TRACE: {}})

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