from distutils.log import error
import boto3
from boto3.dynamodb.conditions import Key

from dynamo_db import ERROR_MSG, IS_FINISHED, IS_PROCESSED, TASK_DETAILS, TASK_ID, WORKER_HISTORY, WORKER_ID, WORKER_STATUS,\
                      WTIMESTAMP, ACTION_TYPE

class WorkerInDB():
    def __init__(self, table_name, worker_id):    
        self.dynamodb = boto3.resource('dynamodb', region_name='us-east-2')
        self.table = self.dynamodb.Table(table_name+'-worker-state')
        self.finished_tasks_table = self.dynamodb.Table(table_name+'-finished-tasks')
        self.worker_id = worker_id

        self.fetch_status()

    def fetch_status(self):
        """
        query database again to get updated status
        """
        resp = self.table.query(
            KeyConditionExpression=Key(WORKER_ID).eq(self.worker_id))

        if len(resp['Items']) != 1:
                raise ValueError(
                    'workerId: {} is multiple or \
                        not found with number of item : {}'.format(self.worker_id, len(resp['Items'])))

        self.device_status = resp['Items'][0]
        self.status = resp['Items'][0][WORKER_STATUS]

    def append_history(self, timestamp, action_type, task, error_msg=""):
        result = self.table.update_item(
            Key={
                WORKER_ID: self.worker_id,
            },
            UpdateExpression="SET #worker_history = list_append(#worker_history, :new_history)",
            ExpressionAttributeNames={
                "#worker_history": WORKER_HISTORY,
            },
            ExpressionAttributeValues={
                ':new_history': [{
                    WTIMESTAMP: timestamp,
                    ACTION_TYPE: action_type,
                    TASK_DETAILS: task,
                    ERROR_MSG: error_msg,
                }],
            },
            ReturnValues="NONE"
        )

    def update_status(self, status):
        resp = self.table.update_item(
                    Key={WORKER_ID: self.worker_id},
                    ExpressionAttributeNames={
                        "#status": WORKER_STATUS
                    },
                    ExpressionAttributeValues={
                        ":status": status
                    },
                    UpdateExpression="SET #status = :status",
                )
        self.status = status

    def update_finished_task(self, task_id, is_finished):
        resp = self.finished_tasks_table.put_item(
            Item={TASK_ID: task_id, IS_FINISHED: is_finished, IS_PROCESSED: False}
        )