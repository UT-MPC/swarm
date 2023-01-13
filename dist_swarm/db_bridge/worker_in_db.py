from distutils.log import error
import boto3
from boto3.dynamodb.conditions import Key
import logging

from dynamo_db import ERROR_MSG, IS_FINISHED, IS_PROCESSED, IS_TIMED_OUT, TASK_DETAILS, TASK_ID, WORKER_HISTORY, WORKER_ID, WORKER_STATUS,\
                      WTIMESTAMP, ACTION_TYPE, TIME
from grpc_components.status import RUNNING, STOPPED

class WorkerInDB():
    def __init__(self, swarm_name, worker_namespace, worker_id):    
        self.dynamodb = boto3.resource('dynamodb', region_name='us-east-2')
        self.table = self.dynamodb.Table(worker_namespace+'-worker-state')
        self.finished_tasks_table = self.dynamodb.Table(swarm_name+'-finished-tasks')
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
        try:
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
        except:
            pass

    def update_finished_task(self, task_id, is_finished, is_timed_out, measured_time):
        resp = self.finished_tasks_table.put_item(
            Item={TASK_ID: task_id, 
                  IS_FINISHED: is_finished, 
                  IS_TIMED_OUT: is_timed_out,
                  IS_PROCESSED: False, 
                  TIME: measured_time}
        )

class WorkerInRDS():
    def __init__(self, cursor, worker_ips):
        self.worker_ips = worker_ips
        self.cursor = cursor
        self.worker_db_ids = []  # ids of workers in RDS

        for ip in worker_ips:
            resp = self.cursor.get_column('worker_id', 'external_ip', ip)
            if len(resp) == 0:
                logging.error(f'get column for ip {ip} failed')
            else:
                self.worker_db_ids.append(resp[0][0])

    def get_stopped_workers(self):
        # return ids of stopped workers
        all_stopped = self.cursor.get_column('worker_id', 'worker_state', STOPPED)
        res = []
        for item in all_stopped:
            if item[0] in self.worker_db_ids:
                res.append(item[0])
        return res
    
    def get_running_workers(self):
        # return ids of stopped workers
        all_running = self.cursor.get_column('worker_id', 'worker_state', RUNNING)
        res = []
        for item in all_running:
            if item[0] in self.worker_db_ids:
                res.append(item[0])
        return res
    
    def get_worker_id(self, worker_ip):
        resp = self.cursor.get_column('worker_id', 'external_ip', worker_ip)
        if len(resp) == 0:
            logging.error(f'get worker id for ip {worker_ip} failed')
            return []
        else:
            return resp[0][0]

    def update_status(self, worker_id, status):
        self.cursor.update_record_by_col('worker_id', worker_id, 'worker_state', status)
        
class TaskInRDS():
    def __init__(self, cursor):
        self.cursor = cursor

    def get_not_processed_finished_tasks(self):
        return self.cursor.get_column_multi_and({'is_processed': 'FALSE', 'is_finished': 'TRUE'})

    def insert_newly_finished_task(self, task_id, is_timed_out, sim_time, real_time):
        self.cursor.insert_record_wo_quotes(['task_id', 'is_processed', 'is_finished', 
                                                'is_timed_out', 'sim_time', 'real_time'],
                                            [str(task_id), 'FALSE', 'TRUE', 
                                                'TRUE' if is_timed_out else 'FALSE', 
                                                str(sim_time), str(real_time)])

    def mark_processed(self, task_id):
        self.cursor.update_record_by_col('task_id', str(task_id), 'is_processed', 'TRUE')
