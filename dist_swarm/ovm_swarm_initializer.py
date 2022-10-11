import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../grpc_components')
import os
import boto3
import grpc
import pickle
import json
import logging
import numpy as np
import threading
from pathlib import Path, PurePath
from pandas import read_pickle, read_csv
from time import gmtime, strftime

import data_process as dp
from models import get_model
from get_dataset import get_dataset
from get_optimizer import get_optimizer
from get_device import get_device_class
import models as custom_models
from dynamo_db import DEVICE_ID, GOAL_DIST, HOSTNAME, IS_PROCESSED, LOCAL_DIST, MODEL_INFO, TASK_ID, TOTAL_ENC_IDX,\
    DATA_INDICES, EVAL_HIST_LOSS, EVAL_HIST_METRIC, ENC_IDX, DEV_STATUS, TIMESTAMPS, ERROR_TRACE, \
    MODEL_INFO, WORKER_ID, WORKER_STATUS, WORKER_HISTORY, WORKER_CREATED, WTIMESTAMP, ACTION_TYPE
from grpc_components.status import STOPPED
from grpc_components import simulate_device_pb2, simulate_device_pb2_grpc
from aws_settings import REGION
from get_model import get_model_fn
from device.check_device_type import is_hetero_strategy
from ovm_utils.device_storing_util import save_device

client = boto3.client('dynamodb', region_name=REGION)
dynamodb = boto3.resource('dynamodb', region_name=REGION)

class OVMSwarmInitializer():
    def initialize(self, config_file) -> None:
        self._config_db(config_file)
        
    def _delete_all_items_on_table(self, table_name):
        try:
            table = dynamodb.Table(table_name)
            #get the table keys
            tableKeyNames = [key.get("AttributeName") for key in table.key_schema]

            #Only retrieve the keys for each item in the table (minimize data transfer)
            projectionExpression = ", ".join('#' + key for key in tableKeyNames)
            expressionAttrNames = {'#'+key: key for key in tableKeyNames}

            counter = 0
            page = table.scan(ProjectionExpression=projectionExpression, ExpressionAttributeNames=expressionAttrNames)
            with table.batch_writer() as batch:
                while page["Count"] > 0:
                    counter += page["Count"]
                    # Delete items in batches
                    for itemKeys in page["Items"]:
                        batch.delete_item(Key=itemKeys)
                    # Fetch the next page
                    if 'LastEvaluatedKey' in page:
                        page = table.scan(
                            ProjectionExpression=projectionExpression, ExpressionAttributeNames=expressionAttrNames,
                            ExclusiveStartKey=page['LastEvaluatedKey'])
                    else:
                        break
            # print(f"Deleted {counter}")
        except:
            # print("new table; nothing to delete")
            pass 

    def _create_finished_tasks_table(self, tag):
        self._create_table(TASK_ID, tag+'-finished-tasks', 100, 100, IS_PROCESSED)

    def _create_db_state_table(self, tag):
        self._create_table(DEVICE_ID, tag, 100, 100)

    def _get_worker_state_table_name(self, tag):
        return tag + '-worker-state'

    def _create_worker_state_table(self, tag):
        self._create_table(WORKER_ID, self._get_worker_state_table_name(tag))

    def _create_table(self, key, table_name, read_cap_units=10, write_cap_units=10, secondary_index=None):  
        """
        stores the state of the worker nodes
        """

        params = {
            'TableName':table_name,
                # Declare your Primary Key in the KeySchema argument
            'KeySchema':[
                {
                    "AttributeName": key,
                    "KeyType": "HASH"
                }
            ],
            # Any attributes used in KeySchema or Indexes must be declared in AttributeDefinitions
            'AttributeDefinitions': [
                {
                    "AttributeName": key,
                    "AttributeType": "N"
                }
            ],
            # ProvisionedThroughput controls the amount of data you can read or write to DynamoDB per second.
            # You can control read and write capacity independently.
            'ProvisionedThroughput': {
                "ReadCapacityUnits": 10,
                "WriteCapacityUnits": 10
            }
        }

        if secondary_index is not None:
            params['GlobalSecondaryIndexes'] = {
                'IndexName': secondary_index,
                'KeySchema': [
                    {
                        'AttributeName': secondary_index,
                        'keyType': "HASH"
                    }
                ]
            }

        try:
            resp = client.create_table(
                **params
            )
            print("Table created successfully. Syncing...")
            waiter = client.get_waiter('table_exists')
            waiter.wait(
                TableName=table_name,
                WaiterConfig={
                    'Delay': 5,
                    'MaxAttempts': 10
                }
            )
            print(f"Table {table_name} Successfully created")

        except Exception as e:
            pass
            # print("Error creating table:")
            # print(e)

        # # delete all the existing items in the db
        self._delete_all_items_on_table(table_name)

    def send_set_worker_state_request(self, swarm_name, worker_id):
        with grpc.insecure_channel(self.worker_ips[worker_id], options=(('grpc.enable_http_proxy', 0),)) as channel:
            stub = simulate_device_pb2_grpc.SimulateDeviceStub(channel)
            worker_info = simulate_device_pb2.WorkerInfo(swarm_name=swarm_name, worker_id=worker_id)
            status = stub.SetWorkerInfo.future(worker_info)
            res = status.result()
            logging.info(f"{self.worker_ips[worker_id]} set as {worker_id}")

    def _initialize_worker(self, tag, worker_id):
        table = dynamodb.Table(self._get_worker_state_table_name(tag))
        with table.batch_writer() as batch:
            batch.put_item(Item={WORKER_ID: worker_id, WORKER_STATUS: STOPPED,
                               WORKER_HISTORY: [{WTIMESTAMP: strftime("%Y-%m-%d %H:%M:%S", gmtime()), ACTION_TYPE: WORKER_CREATED}]})
        set_number_thread = threading.Thread(target=self.send_set_worker_state_request, args=(tag, worker_id,))
        set_number_thread.start()
    
    def _config_db(self, config_file):
        # load config file
        with open(config_file, 'rb') as f:
            config_json = f.read()
        config = json.loads(config_json)
        self.config = config
        tag = config['tag']
        swarm_config = config['swarm_config']
        self.worker_ips = config['worker_ips']

        x_train, y_train_orig, x_test, y_test_orig = get_dataset(config['dataset'])
        num_classes = len(np.unique(y_train_orig))

        self._create_db_state_table(tag)

        enc_dataset_filename = self.config['device_config']['encounter_config']['encounter_data_file']
        enc_dataset_path = PurePath(os.path.dirname(__file__) +'/../' + enc_dataset_filename)
        if enc_dataset_filename.split('.')[-1] == 'pickle':
            with open(enc_dataset_path,'rb') as pfile:
                enc_df = read_pickle(pfile)
        else:
            with open(enc_dataset_path,'rb') as pfile:
                enc_df = read_csv(pfile)

        # initialize all devices and store their states, models, data, etc
        # store local data dist, goal dist, and training data indices in the table
        s3 = boto3.resource('s3')
        for idnum in range(swarm_config['number_of_devices']):
            # print('init db for device {}.'.format(idnum))

            # pick data
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
            for l in label_set:
                goal_dist[l] = (int) (swarm_config['local_data_size'] / len(label_set))

            train_data_provider = dp.IndicedDataProvider(x_train, y_train_orig, local_dist)
            chosen_data_idx = train_data_provider.get_chosen()
            table = dynamodb.Table(tag)
            with table.batch_writer() as batch:
                batch.put_item(Item={DEVICE_ID: idnum, DEV_STATUS: STOPPED, TIMESTAMPS: [],
                    GOAL_DIST: convert_to_map(goal_dist),
                    LOCAL_DIST: convert_to_map(local_dist), TOTAL_ENC_IDX: len(enc_df.index),
                    EVAL_HIST_LOSS: [], EVAL_HIST_METRIC: [], ENC_IDX: -1, ERROR_TRACE: {}, HOSTNAME: 'N/A', MODEL_INFO: {}})
            

            ## initialize device 
            # get model and dataset
            model_fn = get_model(config['dataset'])

            self.device_class = get_device_class(config['device_config']['device_strategy'])

            # bootstrap parameters
            if config['device_config']['pretrained_model'] != "none":
                pretrained_model_path = PurePath(os.path.dirname(__file__) +'/' + config['device_config']['pretrained_model'])
                with open(pretrained_model_path, 'rb') as handle:
                    init_weights = pickle.load(handle)
            else:
                init_weights = None

            test_data_provider = dp.StableTestDataProvider(x_test, y_test_orig, config['device_config']['train_config']['test-data-per-label'])

            # get device info from dynamoDB
            chosen_list = chosen_data_idx
            goal_labels = goal_dist
            
            train_data_provider.set_chosen(chosen_list)

            # prepare params for device
            x_local, y_local_orig = train_data_provider.fetch()
            hyperparams = config['device_config']['train_config']
            compile_config = {'loss': 'mean_squared_error', 'metrics': ['accuracy']}
            train_config = {'batch_size': hyperparams['batch-size'], 'shuffle': True}

            device = self.device_class(idnum,
                                        model_fn, 
                                        get_optimizer(config['device_config']['train_config']['optimizer']),
                                        init_weights,
                                        x_local,
                                        y_local_orig,
                                        train_data_provider,
                                        test_data_provider,
                                        goal_labels,
                                        compile_config,
                                        train_config,
                                        hyperparams)

            # save device model, dataset, and device object on S3 
            save_device(device, config['tag'], -1)

            # @TODO handle hetero device
        
        # configure worker tables
        self._create_worker_state_table(tag)
        for worker_id in range(len(self.worker_ips)):
            self._initialize_worker(tag, worker_id)
        
        self._create_finished_tasks_table(tag)


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
    dist_swarm = OVMSwarm('../configs/dist_swarm/controller_example.json', '')