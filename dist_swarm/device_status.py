import boto3
from boto3.dynamodb.conditions import Key
from decimal import Decimal

from dynamo_db import DEVICE_ID, EVAL_HIST_LOSS, EVAL_HIST_METRIC, \
            GOAL_DIST, LOCAL_DIST, DATA_INDICES, DEV_STATUS, TIMESTAMPS, ERROR_TRACE, ENC_IDX

class DeviceStatus():
    def __init__(self, table_name, device_id):    
        self.dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
        self.table = self.dynamodb.Table(table_name)
        self.device_id = device_id

        self.fetch_status(device_id)

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

    # Getters
    def get_data_indices(self):
        return [int(idx) for idx in self.device_status[DATA_INDICES]]

    def get_local_labels(self):
        return [int(idx) for idx in self.device_status[LOCAL_DIST]]
    
    def get_goal_labels(self):
        return [int(idx) for idx in self.device_status[GOAL_DIST]]

    ########

    # Setters
    def update_loss_and_metric(self, loss_hist, metric_hist, enc_idx):
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



