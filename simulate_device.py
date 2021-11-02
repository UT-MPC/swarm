import tensorflow.keras as keras
import pickle
import boto3
from boto3.dynamodb.conditions import Key
import threading
from pathlib import Path, PurePath
from cfg_utils import OUTPUT_FOLDER
from pandas import read_pickle

import models as custom_models
from get_dataset import get_mnist_dataset, get_cifar_dataset, get_opp_uci_dataset, get_svhn_dataset
from get_device import get_device_class
import data_process as dp
from dynamo_db import DEVICE_ID, GOAL_DIST, LOCAL_DIST, DATA_INDICES
import grpc_components
from grpc_components.status import STATUS, IDLE, RUNNING, ERROR

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
        self.config = request.config
        try:
            self._initialize_dbs(self.config)
            self.device = self._initialize_device(self.config)
            return STATUS.index(IDLE)
        except:
            return STATUS.index(ERROR)

    def StartOppCL(self, request, context):
        try:
            oppcl_thread = threading.Thread(target=self.start_oppcl, args=(1,))
            oppcl_thread.start()
            return STATUS.index(RUNNING)
        except:
            return STATUS.index(ERROR)

    def _start_oppcl(self):
        enc_df = read_pickle(self.config['encounter_config']['encounter_data_file'])
        last_end_time = 0
        last_run_time = 0

        for index, row in enc_df.iterrows():
            if (int)(row[CLIENT1]) == self.device.idnum:
                other_id = (int)(row[CLIENT2])
            elif (int)(row[CLIENT2]) == self.device.idnum:
                other_id = (int)(row[CLIENT1])
            else:
                continue

            if other_id == self.device.idnum:
                continue

            resp = self.table.query(KeyConditionExpression=Key(DEVICE_ID).eq(other_id))

            # get device info from dynamoDB
            if len(resp['Items']) != 1:
                raise ValueError('device Id: {} is multiple or not found with number of item : {}'.format(other_id, len(resp['Items'])))
            other_chosen_list = [int(idx) for idx in resp['Items'][0][DATA_INDICES]]
            other_goal_labels = [int(idx) for idx in resp['Items'][0][GOAL_DIST]]
            
            other_train_data_provider = dp.IndicedDataProvider(self.x_train, self.y_train_orig, None)
            other_test_data_provider = dp.StableTestDataProvider(self.x_test, self.y_test_orig, self.config['device_config']['test_data_per_label'])
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
                
                # @TODO determine rounds
                rounds = min(rounds, self.config['device_config']['train_config']['max_rounds'])
                for r in range(rounds):
                    self.device.delegate(other_device, 1, 1)
                
            ## report eval to dynamoDB

    def _initialize_dbs(self, config):
        self.dynamodb = boto3.resource('dynamodb', region_name='us-east-1')

        # setup local path for storing model locally
        self.model_folder = OUTPUT_FOLDER + '/models'
        Path(self.model_folder).mkdir(parents=True, exist_ok=True)
        # setup S3
        self.s3_model_path = PurePath(config['tag'] + '/' + config['device_config']['id'])
        self.s3 = boto3.client('s3')

    def _initialize_device(self, config, is_config_file):
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
        self.x_train = x_train
        self.y_train_orig = y_train_orig
        self.x_test = x_test
        self.y_test_orig = y_test_orig

        self.device_class = get_device_class(config['device_strategy'])

        # bootstrap parameters
        if config['device_config']['pretrained_model'] != None:
            with open(config['pretrained-model'], 'rb') as handle:
                init_weights = pickle.load(handle)
        else:
            init_weights = None

        train_data_provider = dp.IndicedDataProvider(x_train, y_train_orig, None)
        test_data_provider = dp.StableTestDataProvider(x_test, y_test_orig, config['device_config']['test_data_per_label'])
        self.table = self.dynamodb.Table(config['tag'])
        
        resp = self.table.query(KeyConditionExpression=Key(DEVICE_ID).eq(config['device_config']['id']))

        # get device info from dynamoDB
        if len(resp['Items']) != 1:
            raise ValueError('device Id: {} is multiple or not found with number of item : {}'.format(config['device_config']['id'], len(resp['Items'])))
        chosen_list = [int(idx) for idx in resp['Items'][0][DATA_INDICES]]
        goal_labels = [int(idx) for idx in resp['Items'][0][GOAL_DIST]]
        
        train_data_provider.set_chosen(chosen_list)

        # prepare params for device
        x_local, y_local_orig = train_data_provider.fetch()
        compile_config = {'loss': 'mean_squared_error', 'metrics': ['accuracy']}
        train_config = {'batch_size': config['device_config']['train_config']['batch_size'], 'shuffle': True}

        device = self.device_class(config['device_config']['id'],
                                    model_fn, 
                                    keras.optimizers.Adam,
                                    init_weights,
                                    x_local,
                                    y_local_orig,
                                    train_data_provider,
                                    test_data_provider,
                                    goal_labels,
                                    compile_config,
                                    train_config,
                                    config['device_config']['train_config'])

        return device
