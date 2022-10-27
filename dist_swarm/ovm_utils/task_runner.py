# task runner function for simulate_device class
from decimal import Decimal
from inspect import trace
from time import gmtime, strftime
import traceback
import sys
import copy
import logging
from grpc_components.status import RUNNING, STOPPED

sys.path.insert(0,'..')

from ovm_utils.device_storing_util import load_device, save_device
from dist_swarm.db_bridge.device_in_db import DeviceInDB
from dist_swarm.db_bridge.worker_in_db import WorkerInDB
from dynamo_db import TASK_END, TASK_FAILED, TASK_START

# @TODO implement timeout here
def run_task(worker_status, worker_id, task_config, device_state_cache):
    swarm_name = task_config['swarm_name']
    worker_in_db = WorkerInDB(swarm_name, worker_id)
    if worker_status != STOPPED:
        logging.info(f"RPC call for a new task {task_config['task_id']} is made while worker is busy")
        worker_in_db.update_finished_task(task_config['task_id'], False)
        new_history = {"timestamp": strftime("%Y-%m-%d %H:%M:%S", gmtime()), 
                       "action_type": TASK_FAILED, "task": task_config, "error_msg": traceback.format_exc()}
        return new_history, {}
    worker_in_db.update_status(RUNNING)
    worker_in_db.append_history(strftime("%Y-%m-%d %H:%M:%S", gmtime()), TASK_START, task_config)
    logging.info(f"Running task {task_config['task_id']} in worker {worker_id}")
    logging.info(f"cached is {device_state_cache['device_states'].keys()}")

    try:
        # change the state of current worker
        task_id = task_config['task_id']
        learner_id = task_config['learner']
        neighbor_ids = task_config['neighbors']
        func_list = task_config['func_list']
        device_load_config = task_config['load_config']
        end = Decimal(task_config['end'])

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
        
        # invoke function 
        for i in range(len(func_list)):
            if func_list[i]["func_name"][0] != '!':
                func = getattr(learner, func_list[i]["func_name"])
                
                # @TODO handle multiple neighbors
                func(neighbors[0], **func_list[i]["params"])
            elif func_list[i]["func_name"] == '!evaluate':
                hist = learner.eval()
                device_in_db.update_loss_and_metric(hist[0], hist[1], task_id)
                device_in_db.update_timestamp(end)

        # save to cache before saving to DB
        # because save_device deletes model and data from the device
        device_state_cache['device_states'] = {}
        device_state_cache['device_states'][str(learner_id)] = copy.deepcopy(learner)

        # save device (s)
        save_device(learner, swarm_name, task_id)
        # @TODO save neighbor device states if instructed
        
        new_history = {"timestamp": strftime("%Y-%m-%d %H:%M:%S", gmtime()), 
                       "action_type": TASK_END, "task": task_config}
    except:
        traceback.print_exc()
        new_history = {"timestamp": strftime("%Y-%m-%d %H:%M:%S", gmtime()), 
                       "action_type": TASK_FAILED, "task": task_config, "error_msg": traceback.format_exc()}
        worker_in_db.append_history(**new_history)
        worker_in_db.update_status(STOPPED)
        return new_history

    logging.info(f"-- Task {task_id} successfully finished --")
    try:
        worker_in_db.update_status(STOPPED) 
        worker_in_db.append_history(**new_history)
        worker_in_db.update_finished_task(task_id, True)
    except:
        logging.error(f"Task {task_id} returned an error while updating status: {traceback.format_exc()}")

    return new_history
