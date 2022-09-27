# overmind controller
# reads encounter data, creates tasks and build dependency graph
import sys
sys.path.insert(0,'..')
import json
import os
import time
from pathlib import PurePath
from time import time
from pandas import read_pickle, read_csv
import boto3
from dynamo_db import IS_PROCESSED, TASK_ID
from boto3.dynamodb.conditions import Key

from ovm_swarm_initializer import OVMSwarmInitializer
from get_device import get_device_class

# data frame column names for encounter data
TIME_START="time_start"
TIME_END="time_end"
CLIENT1="client1"
CLIENT2="client2"
ENC_IDX="encounter index"


class Task():
    def __init__(self, swarm_name, task_id, start, end, learner_id, neighbor_id_list):
        self.swarm_name = swarm_name
        self.task_id = task_id
        self.start = start
        self.end = end
        self.learner_id = learner_id
        self.neighbor_id_list = neighbor_id_list
        self.load_config = {str(learner_id): self.get_new_load_config()}
        self.dynamodb = boto3.resource('dynamodb', region_name='us-east-2')
        
        for nbgr in neighbor_id_list:
            self.load_config[str(nbgr)] = self.get_new_load_config()
        self.func_list = []

    def add_func(self, func_name, params):
        self.func_list.append({
            "func_name": func_name,
            "params": params,
        })

    def add_eval(self):
        self.func_list.append({"func_name": "!evaluate"})
    
    def get_config(self):
        return {
            "swarm_name": self.swarm_name,
            "task_id": self.task_id,
            "start": self.start,
            "end": self.end,
            "learner": self.learner_id,
            "neighbors": self.neighbor_id_list,
            "load_config": self.load_config,
            "func_list": self.func_list,
        }
        
    def set_load_config(self, id, load_model: bool, load_dataset: bool):
        pass

    def get_new_load_config(self, load_model: bool=True, load_dataset: bool=True):
        return {"load_model": load_model, "load_dataset": load_dataset}
        

class Overmind():
    def __init__(self):
        self.dynamodb = boto3.resource('dynamodb', region_name='us-east-2')

    def create_swarm(self, config_file, ip):
        with open(config_file, 'rb') as f:
            config_json = f.read()
        self.config = json.loads(config_json)
        self.device_config = self.config["device_config"]
        self.swarm_name = self.config['tag']

        initializer = OVMSwarmInitializer()
        initializer.initialize(config_file, ip)

    def build_dep_graph(self):
        # read encounter dataset
        enc_dataset_filename = self.device_config['encounter_config']['encounter_data_file']
        enc_dataset_path = PurePath(os.path.dirname(__file__) +'/../' + enc_dataset_filename)
        if enc_dataset_filename.split('.')[-1] == 'pickle':
            with open(enc_dataset_path,'rb') as pfile:
                enc_df = read_pickle(pfile)
        else:
            with open(enc_dataset_path,'rb') as pfile:
                enc_df = read_csv(pfile)
        last_end_time = 0
        last_run_time = 0
        
        # get dependency info
        device_class = get_device_class(self.device_config["device_strategy"])
        dependency = device_class.get_dependency()

        # read connection data and populate task list
        task_id = 0
        encounter_config = self.device_config['encounter_config']
        model_send_time = self.device_config['model_size_in_bits'] / encounter_config['communication_rate']
        computation_time = encounter_config['computation_time']
        oppcl_time = 2 * model_send_time + computation_time
        print(f"oppcl time: {oppcl_time}")

        dep_graph = {}  # (preceding task id, task id)
        last_tasks = {}  # (device id, last task idx) 
        last_times = {}  # (device id, last time device is done with oppcl)
        tasks = {}  # (task id, task object)
        indegrees = {}

        for index, row in enc_df.iterrows():
            device1_id = (int)(row[CLIENT1])
            device2_id = (int)(row[CLIENT2])
            start_time = max(row[TIME_START], last_times[device1_id] if device1_id in last_times else 0)
            start_time = max(start_time, last_times[device2_id] if device2_id in last_times else 0)
            end_time = start_time + oppcl_time
            if (row[TIME_END] - start_time >= oppcl_time):
                task_1_2 = Task(self.swarm_name, task_id, start_time, end_time, device1_id, [device2_id])
                task_2_1 = Task(self.swarm_name, task_id+1, start_time, end_time, device2_id, [device1_id])
                task_1_2.add_func("delegate", {"epoch": 1, "iteration": 1})
                task_2_1.add_func("delegate", {"epoch": 1, "iteration": 1})
                task_1_2.add_eval()
                task_2_1.add_eval()
                
                indegrees[task_id] = 0
                indegrees[task_id+1] = 0
                dep_graph[task_id] = []
                dep_graph[task_id+1] = []
                tasks[task_id] = task_1_2
                tasks[task_id+1] = task_2_1

                # we assume that all delegations are dependent on data, at least
                if device1_id in last_tasks:
                    for lt in last_tasks[device1_id]:
                        dep_graph[lt].append(task_1_2.task_id)
                        indegrees[task_1_2.task_id] += 1
                    if dependency["on_mutable"]:
                        for lt in last_tasks[device2_id]:
                            dep_graph[lt].append(task_1_2.task_id)
                            indegrees[task_2_1.task_id] += 1

                if device2_id in last_tasks:
                    for lt in last_tasks[device2_id]:
                        dep_graph[lt].append(task_2_1.task_id)
                        indegrees[task_2_1.task_id] += 1
                    if dependency["on_mutable"]:
                        for lt in last_tasks[device1_id]:
                            dep_graph[lt].append(task_2_1.task_id)
                            indegrees[task_2_1.task_id] += 1

                last_tasks[device1_id] = [task_1_2.task_id]
                last_tasks[device2_id] = [task_2_1.task_id]
                # if task is dependent on mutable state of the neighbor,
                # succeeding tasks should also be dependent on tasks where the device
                # participated as a neighbor
                if dependency["on_mutable"]:
                    last_tasks[device1_id].append(task_2_1.task_id)
                    last_tasks[device2_id].append(task_1_2.task_id)

                task_id += 2
                last_times[device1_id] = row[TIME_START] + oppcl_time
                last_times[device2_id] = row[TIME_START] + oppcl_time

        self.task_num = task_id
        self.dep_graph = dep_graph
        self.last_tasks = last_tasks
        self.tasks = tasks
        self.indegrees = indegrees

        self.finished_tasks_table = self.dynamodb.Table(self.swarm_name +'-finished-tasks')
    
    def run_swarm(self, polling_interval=10):
        while self.task_num > 0:
            resp = self.finished_tasks_table.query(
                KeyConditionExpression=Key(IS_PROCESSED).eq(False)
            )
            task_queue = []
            for newly_finished in resp['Items']:
                # "process" the finished task, which is
                # running tasks that are dependent on that task
                task_id = newly_finished[TASK_ID]
                for next_task in self.dep_graph[task_id]:
                    self.indegrees[next_task] -= 1
                    if self.indegrees[next_task] == 0:
                        task_queue.append(next_task)
            
            # call RunTask asynchronously to all in task queue


            time.sleep(polling_interval)

        

