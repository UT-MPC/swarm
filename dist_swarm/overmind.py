# overmind controller
# reads encounter data, creates tasks and build dependency graph
from sqlite3 import Date
import sys
import traceback
import grpc
sys.path.insert(0,'..')
sys.path.insert(0,'../grpc_components')
import logging

from dist_swarm.db_bridge.worker_in_db import TaskInRDS, WorkerInDB, WorkerInRDS
from grpc_components import simulate_device_pb2, simulate_device_pb2_grpc
from grpc_components.status import RUNNING, STOPPED
import json
import os
import threading
import time
from time import sleep
import pickle
import datetime
import typing
from pathlib import PurePath, Path
from pandas import read_pickle, read_csv
import boto3
from dynamo_db import IS_FINISHED, IS_PROCESSED, IS_TIMED_OUT, TASK_ID, TIME
from boto3.dynamodb.conditions import Key

from ovm_swarm_initializer import OVMSwarmInitializer
from get_device import get_device_class

# data frame column names for encounter data
TIME_START="time_start"
TIME_END="time_end"
CLIENT1="client1"
CLIENT2="client2"
ENC_IDX="encounter index"

MAX_ITERATIONS = 10_000_000

class Task():
    def __init__(self, swarm_name, task_id, start, end, learner_id, neighbor_id_list, worker_namespace,
                 timeout=2**8, real_time_mode=False, communication_time=0, dependant_on_mutable=False):
        self.swarm_name = swarm_name
        self.worker_namespace = worker_namespace
        self.task_id = task_id
        self.start = start
        self.end = end
        self.learner_id = learner_id
        self.neighbor_id_list = neighbor_id_list
        self.timeout = timeout
        self.load_config = {str(learner_id): self.get_new_load_config()}
        self.real_time_mode = real_time_mode
        self.communication_time = communication_time
        self.real_time_timeout = self.end - self.start - communication_time
        self.dependant_on_mutable = dependant_on_mutable

        self.skip = False  # skip the processing of this task
        
        for nbgr in neighbor_id_list:
            self.load_config[str(nbgr)] = self.get_new_load_config()
        self.func_list = []

    def update_real_time_timeout(self, communication_time):
        self.real_time_timeout = self.end - self.start - communication_time

    def determine_skip(self):
        if self.start >= self.end:
            self.skip = True

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
            "worker_namespace": self.worker_namespace,
            "task_id": self.task_id,
            "start": str(self.start),
            "end": str(self.end),
            "learner": self.learner_id,
            "neighbors": self.neighbor_id_list,
            "load_config": self.load_config,
            "func_list": self.func_list,
            "timeout": self.timeout,  # overmind controller assumes that server is dead when timeout is elasped
            "real_time_mode": str(self.real_time_mode),
            "real_time_timeout": str(self.real_time_timeout),
        }
        
    def set_load_config(self, id, load_model: bool, load_dataset: bool):
        pass

    def get_new_load_config(self, load_model: bool=True, load_dataset: bool=True):
        return {"load_model": load_model, "load_dataset": load_dataset}

    def reset_start_time(self, last_avail):
        self.start = max(self.start, last_avail[self.learner_id])
        if self.dependant_on_mutable:
            for nid in self.neighbor_id_list:
                self.start = max(self.start, last_avail[nid])
        self.real_time_timeout = self.end - self.start - self.communication_time
    
    def is_skipped(self, process_time=0):
        return self.end - self.start < process_time
        

class Overmind():
    def __init__(self):
        self.dynamodb = boto3.resource('dynamodb', region_name='us-east-2')

    def create_swarm(self, config_file, skip_init_tables=False):
        with open(config_file, 'rb') as f:
            config_json = f.read()
        self.config = json.loads(config_json)
        self.device_config = self.config["device_config"]
        self.swarm_name = self.config['tag']
        self.worker_nodes : typing.List[str] = self.config["worker_ips"]
        self.worker_namespace = self.config["worker_namespace"]
        self.number_of_devices = self.config["swarm_config"]["number_of_devices"]

        self.log_path = f'ovm_logs/{self.swarm_name}/{datetime.datetime.now()}'
        Path(self.log_path).mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=self.log_path + '/overmind.log', 
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',level=logging.INFO)

        self.initializer = OVMSwarmInitializer()
        self.initializer.initialize(config_file, not skip_init_tables)
        self.worker_db = WorkerInRDS(self.initializer.rds_cursor, self.worker_nodes)

        self.worker_ip_to_id = self.initializer.worker_ip_to_id
        self.worker_id_to_ip = {v: k for k, v in self.worker_ip_to_id.items()}

        self.tasks_db = TaskInRDS(self.initializer.rds_tasks_cursor)

    def build_dep_graph(self, rt_mode=False, dependency=None, oppcl_time=None):
        # read encounter dataset
        enc_dataset_filename = self.device_config['encounter_config']['encounter_data_file']
        enc_dataset_path = PurePath(os.path.dirname(__file__) +'/../' + enc_dataset_filename)
        if enc_dataset_filename.split('.')[-1] == 'pickle':
            with open(enc_dataset_path,'rb') as pfile:
                enc_df = read_pickle(pfile)
        else:
            with open(enc_dataset_path,'rb') as pfile:
                enc_df = read_csv(pfile)
        self.enc_df = enc_df
        last_end_time = 0
        last_run_time = 0
        
        # get dependency info
        if dependency is None:
            device_class = get_device_class(self.device_config["device_strategy"])
            dependency = device_class.get_dependency()
        self.dependant_to_mutable = dependency["on_mutable"]

        # read connection data and populate task list
        task_id = 0
        encounter_config = self.device_config['encounter_config']
        self.model_send_time = self.device_config['model_size_in_bits'] / encounter_config['communication_rate']
        self.computation_time = encounter_config['computation_time']
        self.communication_time = 2 * self.model_send_time 
        if not rt_mode:
            oppcl_time = 2 * self.model_send_time + self.computation_time
        else:
            oppcl_time = 0
        print(f"oppcl time: {oppcl_time}")

        dep_graph = {}  # (preceding task id, task id)
        last_tasks = {}  # (device id, last task idx) 
        last_times = dict.fromkeys(range(self.number_of_devices), 0)  # (device id, last time device is done with oppcl)
        tasks = {}  # (task id, task object)
        indegrees = {}

        for index, row in enc_df.iterrows():
            device1_id = (int)(row[CLIENT1])
            device2_id = (int)(row[CLIENT2])
            if max(device1_id, device2_id) >= self.number_of_devices or device1_id == device2_id:
                continue
            start_time = row[TIME_START]
            if not rt_mode:
                start_time_1 = max(start_time, last_times[device1_id])
                start_time_2 = max(start_time, last_times[device2_id])
                end_time_1 = start_time_1 + oppcl_time
                end_time_2 = start_time_2 + oppcl_time
            else:
                start_time_1 = start_time
                start_time_2 = start_time
                end_time_1 = row[TIME_END]
                end_time_2 = row[TIME_END]

            task_1_timeout = row[TIME_END] - start_time_1 < oppcl_time
            task_2_timeout = row[TIME_END] - start_time_2 < oppcl_time

            if (not task_1_timeout) and (not task_2_timeout):
                task_1_2 = Task(self.swarm_name, task_id, start_time_1, end_time_1, device1_id, [device2_id], self.worker_namespace,
                                real_time_mode=rt_mode, communication_time=self.communication_time)
                task_2_1 = Task(self.swarm_name, task_id+1, start_time_2, end_time_2, device2_id, [device1_id], self.worker_namespace,
                                real_time_mode=rt_mode, communication_time=self.communication_time)
                task_1_2.add_func("delegate", {"epoch": 1, "iteration": 1})
                task_2_1.add_func("delegate", {"epoch": 1, "iteration": 1})
                task_1_2.add_eval()
                task_2_1.add_eval()
                
                if not task_1_timeout:
                    indegrees[task_id] = 0
                    dep_graph[task_id] = []
                    tasks[task_id] = task_1_2

                if not task_2_timeout:
                    indegrees[task_id+1] = 0
                    dep_graph[task_id+1] = []
                    tasks[task_id+1] = task_2_1

                # we assume that all delegations are dependent on data, at least
                if (not task_1_timeout) and device1_id in last_tasks:
                    for lt in last_tasks[device1_id]:
                        dep_graph[lt].append(task_1_2.task_id)
                        indegrees[task_1_2.task_id] += 1

                    if dependency["on_mutable"]:
                        for lt in last_tasks[device2_id]:
                            dep_graph[lt].append(task_1_2.task_id)
                            indegrees[task_1_2.task_id] += 1

                if (not task_2_timeout) and device2_id in last_tasks:
                    for lt in last_tasks[device2_id]:
                        dep_graph[lt].append(task_2_1.task_id)
                        indegrees[task_2_1.task_id] += 1

                    if dependency["on_mutable"]:
                        for lt in last_tasks[device1_id]:
                            dep_graph[lt].append(task_2_1.task_id)
                            indegrees[task_2_1.task_id] += 1

                if not task_1_timeout:
                    last_tasks[device1_id] = [task_1_2.task_id]
                if not task_2_timeout:
                    last_tasks[device2_id] = [task_2_1.task_id]
                # if task is dependent on mutable state of the neighbor,
                # succeeding tasks should also be dependent on tasks where the device
                # participated as a neighbor
                if dependency["on_mutable"]:
                    last_tasks[device1_id].append(task_2_1.task_id)
                    last_tasks[device2_id].append(task_1_2.task_id)

                task_id += 2
                if not rt_mode:
                    if not task_1_timeout:
                        last_times[device1_id] = max(last_times[device1_id], start_time_1 + oppcl_time)
                    if not task_2_timeout:
                        last_times[device2_id] = max(last_times[device2_id], start_time_2 + oppcl_time)

        self.task_num = task_id
        self.dep_graph = dep_graph
        self.last_tasks = last_tasks
        self.tasks = tasks
        self.indegrees = indegrees

        self.finished_tasks_table = self.dynamodb.Table(self.swarm_name +'-finished-tasks')
    
    def send_run_task_request(self, worker_id, task):
        # task.reset_start_time(self.last_avail)
        # if task.is_skipped():  # possible error here
        #     self.processed_tasks.append(task.task_id)
        #     for next_task in self.dep_graph[task.task_id]:
        #         self.indegrees[next_task] -= 1
        #         self.tasks[next_task].reset_start_time(self.last_avail)
        #     return
        try:
            with grpc.insecure_channel(self.worker_id_to_ip[worker_id], options=(('grpc.enable_http_proxy', 0),)) as channel:
                stub = simulate_device_pb2_grpc.SimulateDeviceStub(channel)
                config = simulate_device_pb2.Config(config=json.dumps(task.get_config()))
                status = stub.RunTask.future(config)
                res = status.result()
        except Exception as e:
            logging.error("gRPC call returned with error")
            traceback.print_stack()
    
    def send_check_running_request(self, worker_id):
        try:
            with grpc.insecure_channel(self.worker_id_to_ip[worker_id], options=(('grpc.enable_http_proxy', 0),)) as channel:
                stub = simulate_device_pb2_grpc.SimulateDeviceStub(channel)
                status = stub.CheckRunning(simulate_device_pb2.Empty())
                return status.status == RUNNING
        except Exception as e:
            logging.error("gRPC call returned with error")
            traceback.print_stack()
            return False

    def revive_worker(self, worker_in_db, worker_id):
        if self.send_check_running_request(worker_id):
            worker_in_db.update_status(STOPPED)
            self.initializer._initialize_worker(self.swarm_name, worker_id)
    
    def run_swarm(self, polling_interval=5, rt_mode=False):
        CHECKPOINT_INTERVAL = 1000
        ## !!Warning, rt_mode here has an error
        logging.info("----- swarm run start -----")
        run_swarm_start_time = time.time()
        self.cached_devices_to_worker_nodes = {}  # (cached_device_id, worker node)
        self.cached_worker_nodes_to_devices = {}
        self.task_queue = []
        self.processed_tasks = []
        self.next_checkpoint = CHECKPOINT_INTERVAL
        for task_id in self.tasks:
            if self.indegrees[task_id] == 0:
                if rt_mode:
                    self.tasks[task_id].real_time_mode = True
                self.task_queue.append(task_id)

        worker_last_avail = {}  # last time that a worker was "idle"
        for wi in self.worker_db.worker_db_ids:
            worker_last_avail[wi] = 0

        iterations = 0
        self.successful_tasks = 0
        self.timed_out_tasks = 0

        self.allocated_tasks = {}
        self.last_avail = dict.fromkeys(range(self.number_of_devices), 0)

        # @TODO store which task is allocated to which and re-allocate when timeout
        while self.task_num > 0 and iterations < MAX_ITERATIONS:
            iterations += 1

            finished_not_processed_tasks = self.tasks_db.get_not_processed_finished_tasks()
            # print(f"not fin: {finished_not_processed_tasks}")
            # checkpointing
            if len(self.processed_tasks) >= self.next_checkpoint:
                self.save_checkpoint({'elasped_time': f'{time.time() - run_swarm_start_time}'}, len(self.processed_tasks))
                self.next_checkpoint += CHECKPOINT_INTERVAL

            # re-allocate failed tasks
            for task_item in finished_not_processed_tasks:
                if not task_item[3]:
                    self.task_queue.append(task_item[1])
                    logging.info(f"re-allocating failed task {task_item}")
                    # TODO delete task item from DB

            for task_item in finished_not_processed_tasks:
                # "process" the finished task, which is
                # decrementing indegrees of tasks that are dependent on that task
                task_id = task_item[1]
                learner_id = self.tasks[task_id].learner_id
                neighbor_ids = self.tasks[task_id].neighbor_id_list
                self.processed_tasks.append(task_id)

                # get end time of the task
                elasped_time = float(task_item[5]) + 2 * self.model_send_time
                freed_time = self.tasks[task_id].start + elasped_time
                if not task_item[4]:
                    if freed_time > self.tasks[task_id].end:
                        logging.info(f"DISCREPANCY!: when not timed out, timed out: freed_time: {freed_time}, end: {self.tasks[task_id].end}")
                    self.successful_tasks += 1
                    self.last_avail[learner_id] = max(self.last_avail[learner_id], freed_time)
                    if self.dependant_to_mutable:
                        for nid in neighbor_ids:
                            self.last_avail[nid] = max(self.last_avail[nid], freed_time)
                else:
                    self.timed_out_tasks += 1
                    # if this is "end" we assume that our contact prediction is very bad
                    # logging.info(f"freed_time: {freed_time}, end time: {self.tasks[task_id].end}")
                    freed_time = self.tasks[task_id].start
                

                for next_task in self.dep_graph[task_id]:
                    # print(f"next task {next_task}")
                    self.indegrees[next_task] -= 1
                    if rt_mode:
                        self.tasks[next_task].real_time_mode = True
                        self.tasks[next_task].reset_start_time(self.last_avail)
                            
                    if self.indegrees[next_task] == 0 and next_task not in self.processed_tasks:
                        self.tasks[next_task].determine_skip()
                        self.task_queue.append(next_task)

                self.tasks_db.mark_processed(task_id)
                self.task_num -= 1

            # get "Stopped" workers and check if one of them holds recent device state
            # worker_dbs = [WorkerInDB(self.swarm_name, self.worker_namespace, id) for id, ip in enumerate(self.worker_nodes)]
            # free_workers = [worker.worker_id for worker in worker_dbs if worker.status != RUNNING]
            task_id_to_worker = {}
            
            free_workers = self.worker_db.get_stopped_workers()
            # "revive" the worker if running for more than 30 iterations
            # for w in worker_last_avail:
            #     worker_last_avail[w] += 10
            # for w in free_workers:
            #     worker_last_avail[w] = 0
            # for w in worker_last_avail:
            #     if worker_last_avail[w] > 300:
            #         logging.info(f"reviving worker {w}")
            #         self.revive_worker(worker_dbs[w], w)
            #         worker_last_avail[w] = 0
            #         self.task_queue.insert(0, self.allocated_tasks[w])

            logging.info(f"free workers {free_workers}, task queue size {len(self.task_queue)}")

            ## start assigning tasks to worker nodes (populate task_id_to_worker)
            # first assign based on cached device state
            # print(f"cached: {self.cached_devices_to_worker_nodes}")
            for task_id in self.task_queue:
                # @TODO support mutable neighbors
                if self.tasks[task_id].learner_id in self.cached_devices_to_worker_nodes and \
                    task_id not in task_id_to_worker and \
                    self.cached_devices_to_worker_nodes[self.tasks[task_id].learner_id] in free_workers:  
                    target_worker_id = self.cached_devices_to_worker_nodes[self.tasks[task_id].learner_id]
                    task_id_to_worker[task_id] = target_worker_id
                    free_workers.remove(target_worker_id)
                    logging.info(f"using {target_worker_id} to reuse state {self.tasks[task_id].learner_id} in {task_id}")
            
            # delete assigned tasks from task queue
            for task_id in task_id_to_worker:
                self.task_queue.remove(task_id)

            # assign remaining tasks to "Stopped" worker nodes
            while len(free_workers) > 0 and len(self.task_queue) > 0:
                worker_id = free_workers.pop()
                task_id_to_worker[self.task_queue.pop()] = worker_id

            # print(f"{task_id_to_worker}")
            # print(f"{self.task_queue}")
            logging.info(f"tasks left: {self.task_num}")

            # call RunTask asynchronously 
            for task_id in task_id_to_worker:
                task_thread = threading.Thread(target=self.send_run_task_request, args=(task_id_to_worker[task_id], self.tasks[task_id]))
                task_thread.start()
                self.allocated_tasks[task_id_to_worker[task_id]] = task_id
                self.cached_devices_to_worker_nodes[self.tasks[task_id].learner_id] = task_id_to_worker[task_id]

            sleep(polling_interval)
        
        self.save_checkpoint({'elasped_time': f'{time.time() - run_swarm_start_time}'}, 'last')
        logging.info(f"Overmind run finished successfully with {iterations} iterations, elasped time {time.time() - run_swarm_start_time} sec.")

    def save_checkpoint(self, log_dict, checkpoint_num):
        # save data for current progress, which are the following
        # device states
        # finished tasks
        # elasped time (which is given in log_dict)

        # save device states
        self._save_dynamodb_table(self.swarm_name, 'device_table', checkpoint_num)

        # save finished tasks
        self._save_rds_table(self.initializer.rds_tasks_cursor, 'tasks_table', checkpoint_num)

        # save log_dict
        filepath = self.log_path + f'/etc_{checkpoint_num}.log'
        with open(filepath, 'wb') as handle:
            pickle.dump(log_dict, handle, protocol=pickle.HIGHEST_PROTOCOL) 

    def _save_dynamodb_table(self, table_name, pickle_file_name, checkpoint_num):
        table = self.dynamodb.Table(table_name)
        resp = table.scan()
        filepath = self.log_path + f'/{pickle_file_name}_{checkpoint_num}.log'
        self._save_as_pickle(resp, filepath)

    def _save_rds_table(self, cursor, filename, checkpoint_num):
        record = cursor.get_all_records()
        filepath = self.log_path + f'/rds_{filename}_{checkpoint_num}.log'
        self._save_as_pickle(record, filepath)

    def _save_as_pickle(self, obj, filepath):
        with open(filepath, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
