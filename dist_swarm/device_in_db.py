import sys
sys.path.insert(0,'./grpc_components')
import boto3
from boto3.dynamodb.conditions import Key
import time
from decimal import Decimal

from dynamo_db import DEVICE_ID, EVAL_HIST_LOSS, EVAL_HIST_METRIC, HOSTNAME, \
            GOAL_DIST, LOCAL_DIST, DATA_INDICES, DEV_STATUS, TIMESTAMPS, ERROR_TRACE, ENC_IDX
from grpc_components.status import IDLE, RUNNING, ERROR, FINISHED

class DeviceInDB():
    def __init__(self, table_name, device_id):    
        self.dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
        self.table = self.dynamodb.Table(table_name)
        self.device_id = device_id

        self.fetch_status()

    def fetch_status(self):
        """
        query database again to get updated status
        """
        resp = self.table.query(
            KeyConditionExpression=Key(DEVICE_ID).eq(self.device_id))

        if len(resp['Items']) != 1:
                raise ValueError(
                    'device Id: {} is multiple or \
                        not found with number of item : {}'.format(self.device_id, len(resp['Items'])))

        self.device_status = resp['Items'][0]
        self.status = resp['Items'][0][DEV_STATUS]

    # Getters
    def get_data_indices(self):
        return [int(idx) for idx in self.device_status[DATA_INDICES]]

    def get_local_labels(self):
        return [int(idx) for idx in self.device_status[LOCAL_DIST]]
    
    def get_goal_labels(self):
        return [int(idx) for idx in self.device_status[GOAL_DIST]]

    def get_status(self):
        return self.status

    ########

    # Setters
    def update_loss_and_metric(self, loss, metric, enc_idx):
        resp = self.table.update_item(
                    Key={DEVICE_ID: self.device_id},
                    ExpressionAttributeNames={
                        "#loss": EVAL_HIST_LOSS,
                        "#metric": EVAL_HIST_METRIC,
                        "#enc_idx": ENC_IDX,
                    },
                    ExpressionAttributeValues={
                        ":loss": [Decimal(loss)],
                        ":metric": [Decimal(metric)],
                        ":enc_idx": enc_idx
                    },
                    UpdateExpression="SET #loss = list_append(#loss, :loss), \
                    #metric = list_append(#metric, :metric), #enc_idx = :enc_idx",
                )

    def update_loss_and_metric_in_bulk(self, loss_hist, metric_hist, enc_idx):
        resp = self.table.update_item(
                    Key={DEVICE_ID: self.device_id},
                    ExpressionAttributeNames={
                        "#loss": EVAL_HIST_LOSS,
                        "#metric": EVAL_HIST_METRIC,
                        "#enc_idx": ENC_IDX,
                    },
                    ExpressionAttributeValues={
                        ":loss": [Decimal(str(loss)) for loss in loss_hist],
                        ":metric": [Decimal(str(metric)) for metric in metric_hist],
                        ":enc_idx": enc_idx
                    },
                    UpdateExpression="SET #loss = :loss, #metric = :metric, #enc_idx = :enc_idx",
                )

    def update_timestamps_in_bulk(self, timestamps):
        resp = self.table.update_item(
                    Key={DEVICE_ID: self.device_id},
                    ExpressionAttributeNames={
                        "#timestamps": TIMESTAMPS
                    },
                    ExpressionAttributeValues={
                        ":timestamps": [Decimal(str(ts)) for ts in timestamps]
                    },
                    UpdateExpression="SET #timestamps = :timestamps"
                )

    def set_error(self, e):
        self.update_status(ERROR)
        resp = self.table.update_item(
                    Key={DEVICE_ID: self.device_id},
                    ExpressionAttributeNames={
                        "#error": ERROR_TRACE
                    },
                    ExpressionAttributeValues={
                        ":error": str(e)
                    },
                    UpdateExpression="SET #error = :error",
                )

    def update_status(self, status):
        resp = self.table.update_item(
                    Key={DEVICE_ID: self.device_id},
                    ExpressionAttributeNames={
                        "#status": DEV_STATUS
                    },
                    ExpressionAttributeValues={
                        ":status": status
                    },
                    UpdateExpression="SET #status = :status",
                )
        self.status = status

    def set_hostname(self, hostname):
        resp = self.table.update_item(
                    Key={DEVICE_ID: self.device_id},
                    ExpressionAttributeNames={
                        "#hostname": HOSTNAME
                    },
                    ExpressionAttributeValues={
                        ":hostname": hostname
                    },
                    UpdateExpression="SET #hostname = :hostname",
                )


