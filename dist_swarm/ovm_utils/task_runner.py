# task runner function for simulate_device class
from decimal import Decimal
from inspect import trace
from time import gmtime, strftime
import traceback
import sys
import copy
import logging
import time
from grpc_components.status import RUNNING, STOPPED

sys.path.insert(0,'..')

from ovm_utils.device_storing_util import load_device, save_device
from dist_swarm.db_bridge.device_in_db import DeviceInDB
from dist_swarm.db_bridge.worker_in_db import WorkerInRDS
from dynamo_db import TASK_END, TASK_FAILED, TASK_START, TASK_REALTIME_TIMEOUT

# @TODO implement timeout here
def run_task(worker_db, worker_status, worker_id, task_config, device_state_cache):
    swarm_name = task_config['swarm_name']
    worker_namespace = task_config['worker_namespace']
    if worker_status != STOPPED:
        logging.info(f"RPC call for a new task {task_config['task_id']} is made while worker is busy")
        # worker_in_db.update_finished_task(task_config['task_id'], False)
        new_history = {"timestamp": strftime("%Y-%m-%d %H:%M:%S", gmtime()), 
                       "action_type": TASK_FAILED, "task": task_config, "error_msg": traceback.format_exc()}
        return new_history, {}
    worker_db.update_status(worker_id, RUNNING)
    # worker_in_db.append_history(strftime("%Y-%m-%d %H:%M:%S", gmtime()), TASK_START, task_config)
    logging.info(f"Running task {task_config['task_id']} in worker {worker_id}")
    logging.info(f"cached is {device_state_cache['device_states'].keys()}")

    try:
        # change the state of current worker
        task_id = task_config['task_id']
        learner_id = task_config['learner']
        neighbor_ids = task_config['neighbors']
        func_list = task_config['func_list']
        device_load_config = task_config['load_config']
        timeout = task_config['timeout']
        worker_namespace = task_config['worker_namespace']
        real_time_mode = task_config['real_time_mode']
        real_time_timeout = float(task_config['real_time_timeout'])
        end = Decimal(task_config['end'])
        measured_time = 0

        # load device
        learner_load_config = {}
        if str(learner_id) in device_load_config:
            learner_load_config = device_load_config[str(learner_id)]
        # if device is already in cache, use that
        # @TODO do sanity check. Cached device is not up-to-date? reused in different swarm run?
        if str(learner_id) in device_state_cache['device_states']:
            learner = device_state_cache['device_states'][str(learner_id)]
            logging.info(f"device state {learner_id} loaded from cache")
        else:
            learner = load_device(swarm_name, learner_id, **learner_load_config)

        device_in_db = DeviceInDB(swarm_name, learner_id)
        neighbors = []
        for ngbr in neighbor_ids:
            load_config = {}
            if str(ngbr) in device_load_config:
                load_config = device_load_config[str(ngbr)]
            neighbors.append(load_device(swarm_name, ngbr, **load_config))
        
        realtime_timed_out = False
        # invoke function 
        for i in range(len(func_list)):
            if func_list[i]["func_name"][0] != '!':
                func = getattr(learner, func_list[i]["func_name"])
                
                # @TODO handle multiple neighbors and multiple function calls before eval()
                start = time.time()
                func(neighbors[0], **func_list[i]["params"])
                measured_time += time.time() - start
            elif func_list[i]["func_name"] == '!evaluate' and measured_time <= timeout:
                hist = learner.eval()
                if not real_time_mode or real_time_timeout >= measured_time:
                    device_in_db.update_loss_and_metric(hist[0], hist[1], task_id)
                    device_in_db.update_timestamp(end)
                    device_in_db.update_encounter_history(TASK_END)
                else:
                    device_in_db.update_encounter_history(TASK_REALTIME_TIMEOUT)
                    realtime_timed_out = True

        # save to cache before saving to DB
        # because save_device deletes model and data from the device
        device_state_cache['device_states'] = {}
        device_state_cache['device_states'][str(learner_id)] = copy.deepcopy(learner)

        # save device (s)
        save_device(learner, swarm_name, task_id)
        # @TODO save neighbor device states if instructed
        
        new_history = {"timestamp": strftime("%Y-%m-%d %H:%M:%S", gmtime()), 
                       "action_type": TASK_REALTIME_TIMEOUT 
                       if real_time_mode and real_time_timeout < measured_time else TASK_END, "task": task_config}
    except:
        traceback.print_exc()
        new_history = {"timestamp": strftime("%Y-%m-%d %H:%M:%S", gmtime()), 
                       "action_type": TASK_FAILED, "task": task_config, "error_msg": traceback.format_exc()}
        # worker_in_db.append_history(**new_history)
        worker_db.update_status(worker_id, STOPPED)
        return new_history

    logging.info(f"-- Task {task_id} successfully finished --")
    try:
        worker_db.update_status(worker_id, STOPPED)
        # worker_in_db.append_history(**new_history)
        # worker_in_db.update_finished_task(task_id, True, realtime_timed_out, Decimal(measured_time))
    except:
        logging.error(f"Task {task_id} returned an error while updating status: {traceback.format_exc()}")

    return new_history
