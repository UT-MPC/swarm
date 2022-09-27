# model stored in S3

import tensorflow.keras as keras
import boto3
from pathlib import Path, PurePath
from cfg_utils import OUTPUT_FOLDER

BUCKET_NAME = 'opfl-sim-models'

class ModelInDB():
    def __init__(self, tag, id, local_path):
        """
        tag: name of the swarm
        id: model id
        local_path: path in which the model is stored locally
        """
        self.s3 = boto3.client('s3')
        self.local_path = local_path
        self.s3_model_path = PurePath(tag + '/' + 'model-' + str(id) + '.h5')
        self.id = id
        self.local_model_enc_id = -1
    
    def download(self):
        """
        download model from S3 and store in local storage
        """
        self.s3.meta.client.download_file(BUCKET_NAME, self.s3_model_path, self.local_path)

    def get_enc_id(self):
        """
        get enc-id of the model stored in S3
        """
        metadata = self.s3.head_object(BUCKET_NAME, self.s3_model_path)
        self.local_model_enc_id = int(metadata['x-amz-meta-enc-id'])
        return self.local_model_enc_id

    def get_stored_model(self):
        """
        get model from local storage
        """
        model = keras.models.load_model(self.local_path)
        return model

    def upload_stored_model(self, enc_id):
        self.s3.meta.client.upload_file(BUCKET_NAME, self.local_path, self.s3_model_path)

    # @TODO implement rest of the class for sync. simulation