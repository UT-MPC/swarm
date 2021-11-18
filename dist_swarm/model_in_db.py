import boto3
from pathlib import Path, PurePath
from cfg_utils import OUTPUT_FOLDER

class ModelInDB():
    def __init__(self, tag, id):
        self.s3 = boto3.client('s3')
        self.s3_model_path = PurePath(tag + '/' + str(id))