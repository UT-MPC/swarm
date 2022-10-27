# functions for storing device on AWS
# Device is divided into four parts: 
#       device state (encounter idx, goal/label distribution)
#       device file (pickled device object. attributes of the device. make sure to sync with dev state)
#                   (model and dataset is deleted when saved and retrieved later)
#       device model (just the model parameters)
#       device dataset (dataset indices)
#
import boto3
import pickle
import tensorflow.keras as keras
from pathlib import Path

from device import base_device
from dist_swarm.aws_settings import REGION
from dist_swarm.db_bridge.device_in_db import DeviceInDB
from dist_swarm.db_bridge.model_in_db import BUCKET_NAME

dynamodb = boto3.resource('dynamodb', region_name=REGION)

def get_device_model_object_name(tag, id):
    return tag + '/model-' + str(id) + '.h5'

def get_device_file_name(tag, id):
    return tag + '/device-' + str(id) + '.pickle'

def get_device_resource_object_name(tag, name, id):
    return tag + '/obj-' + name + '-' + str(id) + '.pickle'

def save_device_as_pickle(device: base_device.Device, tag, id, enc_idx):
    # store pickled device to S3
    s3 = boto3.resource('s3')
    tmp_device_path = ".tmp/device.pickle"
    with open(tmp_device_path, 'wb') as handle:
        pickle.dump(device, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    s3.meta.client.upload_file(tmp_device_path, 
                                BUCKET_NAME, 
                                tag + '/' + 'device-' + str(id) + '.pickle',
                                {'Metadata': {'enc-id': str(enc_idx)}})

def load_device_as_pickle(tag, id):
    s3 = boto3.resource('s3')
    tmp_device_path = ".tmp/device.pickle"
    s3.Bucket(BUCKET_NAME).download_file(get_device_file_name(tag, id), tmp_device_path)
    with open(tmp_device_path, 'rb') as handle:
        return pickle.load(handle)

def save_device_model(device: base_device.Device, tag, id, enc_idx):
    s3 = boto3.resource('s3')
    model = device._model_fn()
    model.set_weights(device._weights)
    init_model_path = '.tmp/init_model.h5'
    model.save(init_model_path)
    s3.meta.client.upload_file(init_model_path, 
                                BUCKET_NAME, 
                                get_device_model_object_name(tag, id),
                                {'Metadata': {'enc-id': str(enc_idx)}})

def load_device_model(device: base_device.Device, tag, id):
    s3 = boto3.resource('s3')
    tmp_model_path = '.tmp/loaded_model.h5'
    s3.Bucket(BUCKET_NAME).download_file(get_device_model_object_name(tag, id),
                     tmp_model_path)
    model = keras.models.load_model(tmp_model_path, compile=False)
    device._weights = model.get_weights()

def save_device_dataset(device: base_device.Device, tag, id, enc_idx):
    save_data_object(device._x_train, "x_train", tag, id, enc_idx)
    save_data_object(device._y_train_orig, "y_train_orig", tag, id, enc_idx)
    save_data_object(device._y_train, "y_train", tag, id, enc_idx)
    save_data_object(device.test_data_provider, "test_data_provider", tag, id, enc_idx)

def save_data_object(obj, name, tag, id, enc_idx):
    s3 = boto3.resource('s3')
    tmp_local_file_path = ".tmp/dataset.pickle"
    with open(tmp_local_file_path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    s3.meta.client.upload_file(tmp_local_file_path, 
                                BUCKET_NAME, 
                                get_device_resource_object_name(tag, name, id),
                                {'Metadata': {'enc-id': str(enc_idx)}})

def load_and_set_attr(device, attr_name, obj_name, tag, id):
    s3 = boto3.resource('s3')
    tmp_local_file_path = ".tmp/dataset.pickle"
    s3.Bucket(BUCKET_NAME).download_file(get_device_resource_object_name(tag, obj_name, id),
                     tmp_local_file_path)
    with open(tmp_local_file_path, 'rb') as handle:
        loaded_obj = pickle.load(handle)
    setattr(device, attr_name, loaded_obj)

def load_data_objects(device, tag, id):
    s3 = boto3.resource('s3')
    load_and_set_attr(device, '_x_train', 'x_train', tag, id)
    load_and_set_attr(device, '_y_train_orig', 'y_train_orig', tag, id)
    load_and_set_attr(device, '_y_train', 'y_train', tag, id)
    load_and_set_attr(device, 'test_data_provider', 'test_data_provider', tag, id)
    device.set_local_data(device._x_train, device._y_train_orig)

def save_device(device: base_device.Device, tag: str, enc_idx: int):
    id = device._id_num
    # save device states
    # device_state = DeviceInDB(tag, id)
    # device_state.update_loss_and_metric_in_bulk(device.hist_loss, device.hist_metric, enc_idx)
    # device_state.update_timestamps_in_bulk(device.timestamps)

    # save model
    save_device_model(device, tag, id, enc_idx)

    # save data
    save_device_dataset(device, tag, id, enc_idx)

    # strip model and data from the device and save device as pickle
    device._weights = ""
    device._x_train = ""
    device._y_train_orig = ""
    device._y_train = ""
    device.test_data_provider = ""

    # save device
    save_device_as_pickle(device, tag, id, enc_idx)

def load_device(tag, device_id, load_model=True, load_dataset=True):
    Path(".tmp").mkdir(parents=True, exist_ok=True)
    # load pickled device class
    device = load_device_as_pickle(tag, device_id)
    if load_model:
        load_device_model(device, tag, device_id)
    if load_dataset:
        load_data_objects(device, tag, device_id)

    return device
    