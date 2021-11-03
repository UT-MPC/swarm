import sys
sys.path.insert(0,'./grpc_components')
import os
import tensorflow.keras as keras
import pickle
import boto3
from boto3.dynamodb.conditions import Key
import threading
from pathlib import Path, PurePath
from cfg_utils import OUTPUT_FOLDER
from pandas import read_pickle, read_csv
import logging
import json
from io import StringIO
from decimal import Decimal

import models as custom_models
from get_dataset import get_mnist_dataset, get_cifar_dataset, get_opp_uci_dataset, get_svhn_dataset
from get_device import get_device_class
import data_process as dp
from dynamo_db import DEVICE_ID, EVAL_HIST_LOSS, EVAL_HIST_METRIC, \
            GOAL_DIST, LOCAL_DIST, DATA_INDICES, DEV_STATUS, TIMESTAMPS
import grpc_components.simulate_device_pb2_grpc
from grpc_components.status import IDLE, RUNNING, ERROR, FINISHED
from grpc_components.simulate_device_pb2 import Status

S3_BUCKET_NAME = 'simulate-device'

# data frame column names for encounter data
TIME_START="time_start"
TIME_END="time_end"
CLIENT1="client1"
CLIENT2="client2"
ENC_IDX="encounter index"

class SimulateDeviceServicer(grpc_components.simulate_device_pb2_grpc.SimulateDeviceServicer):
    ### gRPC methods
    def InitDevice(self, request, context):
        parsed_config = json.load(StringIO(request.config))
        self.config = parsed_config
        self.device_config = parsed_config['device_config']
        try:
            self._initialize_dbs(self.config)
            self.device = self._initialize_device(self.config)
            self._set_device_status(IDLE)
            return self._str_to_status(IDLE)
        except Exception as e:
            logging.error(e)
            self._set_device_status(ERROR)
            return self._str_to_status(ERROR)

    def StartOppCL(self, request, context):
        if self.status == ERROR:
            logging.error('device is in error state')
            return self._str_to_status(ERROR)
        try:
            oppcl_thread = threading.Thread(target=self._start_oppcl, args=())
            oppcl_thread.start()
            return self._str_to_status(RUNNING)
        except Exception as e:
            logging.error(e)
            return self._str_to_status(ERROR)

    def _start_oppcl(self):
        enc_dataset_path = PurePath(os.path.dirname(__file__) +'/' + self.device_config['encounter_config']['encounter_data_file'])
        enc_df = read_pickle(enc_dataset_path)
        last_end_time = 0
        last_run_time = 0
        self._set_device_status(RUNNING)
        self.hist_loss = []
        self.hist_metric = []
        self.timestamps = []

        for index, row in enc_df.iterrows():
            if (int)(row[CLIENT1]) == self.device._id_num:
                other_id = (int)(row[CLIENT2])
            elif (int)(row[CLIENT2]) == self.device._id_num:
                other_id = (int)(row[CLIENT1])
            else:
                continue

            if other_id == self.device._id_num or other_id >= self.config['swarm_config']['number_of_devices']:
                continue

            resp = self.table.query(KeyConditionExpression=Key(DEVICE_ID).eq(other_id))

            # get device info from dynamoDB
            if len(resp['Items']) != 1:
                raise ValueError('device Id: {} is multiple or not found with number of item : {}'.format(other_id, len(resp['Items'])))
            other_chosen_list = [int(idx) for idx in resp['Items'][0][DATA_INDICES]]
            other_goal_labels = [int(idx) for idx in resp['Items'][0][GOAL_DIST]]
            
            other_train_data_provider = dp.IndicedDataProvider(self.x_train, self.y_train_orig, None)
            other_test_data_provider = dp.StableTestDataProvider(self.x_test, self.y_test_orig, self.device_config['train_config']['test-data-per-label'])
            other_train_data_provider.set_chosen(other_chosen_list)

            other_x_local, other_y_local_orig = other_train_data_provider.fetch()
                
            other_device = self.device_class(other_id,
                                            None, 
                                            None,
                                            None,
                                            other_x_local,
                                            other_y_local_orig,
                                            other_train_data_provider,
                                            other_test_data_provider,
                                            other_goal_labels,
                                            None,
                                            None,
                                            None)


            if self.device.decide_delegation(other_device):
                # calculate time
                cur_t = row[TIME_START]
                end_t = row[TIME_END]
                time_left = end_t - cur_t
                if last_end_time > cur_t:
                    continue
                
                # determine available rounds of training and conduct OppCL
                encounter_config = self.device_config['encounter_config']
                model_send_time = self.device_config['model_size_in_bits'] / encounter_config['communication_rate']
                computation_time = encounter_config['computation_time']
                oppcl_time = 2 * model_send_time + computation_time
                rounds = (int) ((time_left) / oppcl_time)
                rounds = min(rounds, self.device_config['train_config']['max_rounds'])
                if rounds < 1:
                    continue
                for r in range(rounds):
                    self.device.delegate(other_device, 1, 1)
                last_end_time = cur_t + rounds * oppcl_time
                
                # evaluate
                hist = self.device.eval()
                self.hist_loss.append(hist[0])
                self.hist_metric.append(hist[1])
                self.timestamps.append(last_end_time)

                # report eval to dynamoDB @TODO catch error
                resp = self.table.update_item(
                    Key={DEVICE_ID: self.device._id_num},
                    ExpressionAttributeNames={
                        "#loss": EVAL_HIST_LOSS,
                        "#metric": EVAL_HIST_METRIC,
                        "#enc_idx": ENC_IDX,
                    },
                    ExpressionAttributeValues={
                        ":loss": [Decimal(str(loss)) for loss in self.hist_loss],
                        ":metric": [Decimal(str(metric)) for metric in self.hist_metric],
                        ":enc_idx": index
                    },
                    UpdateExpression="SET #loss = :loss, #metric = :metric, #enc_idx = :enc_idx",
                )

                # @TODO for sync device, upload model to S3 here

        self._set_device_status(FINISHED)

    def _str_to_status(self, st):
        return Status(status=st)

    def _set_device_status(self, status):
        """
        set device status on dynamoDB
        """
        self.status = status
        resp = self.table.update_item(
                    Key={DEVICE_ID: self.config['device_config']['id']},
                    ExpressionAttributeNames={
                        "#status": DEV_STATUS
                    },
                    ExpressionAttributeValues={
                        ":status": self.status
                    },
                    UpdateExpression="SET #status = :status",
                )

    def _initialize_dbs(self, config):
        if not hasattr(self, 'config'):
            self.config = config
        self.dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
        self.table = self.dynamodb.Table(config['tag'])

        # setup local path for storing model locally
        self.model_folder = OUTPUT_FOLDER + '/models'
        Path(self.model_folder).mkdir(parents=True, exist_ok=True)
        # setup S3
        self.s3_model_path = PurePath(config['tag'] + '/' + str(config['device_config']['id']))
        self.s3 = boto3.client('s3')

    def _initialize_device(self, config):
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
        self.x_train = x_train
        self.y_train_orig = y_train_orig
        self.x_test = x_test
        self.y_test_orig = y_test_orig

        self.device_class = get_device_class(config['device_config']['device_strategy'])

        # bootstrap parameters
        if config['device_config']['pretrained_model'] != "none":
            pretrained_model_path = PurePath(os.path.dirname(__file__) +'/' + config['device_config']['pretrained_model'])
            with open(pretrained_model_path, 'rb') as handle:
                init_weights = pickle.load(handle)
        else:
            init_weights = None

        train_data_provider = dp.IndicedDataProvider(x_train, y_train_orig, None)
        test_data_provider = dp.StableTestDataProvider(x_test, y_test_orig, config['device_config']['train_config']['test-data-per-label'])
        
        resp = self.table.query(KeyConditionExpression=Key(DEVICE_ID).eq(config['device_config']['id']))

        # get device info from dynamoDB
        if len(resp['Items']) != 1:
            raise ValueError('device Id: {} is multiple or not found with number of item : {}'.format(config['device_config']['id'], len(resp['Items'])))
        chosen_list = [int(idx) for idx in resp['Items'][0][DATA_INDICES]]
        goal_labels = [int(idx) for idx in resp['Items'][0][GOAL_DIST]]
        
        train_data_provider.set_chosen(chosen_list)

        # prepare params for device
        x_local, y_local_orig = train_data_provider.fetch()
        hyperparams = config['device_config']['train_config']
        compile_config = {'loss': 'mean_squared_error', 'metrics': ['accuracy']}
        train_config = {'batch_size': hyperparams['batch-size'], 'shuffle': True}

        device = self.device_class(config['device_config']['id'],
                                    model_fn, 
                                    keras.optimizers.SGD,
                                    init_weights,
                                    x_local,
                                    y_local_orig,
                                    train_data_provider,
                                    test_data_provider,
                                    goal_labels,
                                    compile_config,
                                    train_config,
                                    hyperparams)

        return device
