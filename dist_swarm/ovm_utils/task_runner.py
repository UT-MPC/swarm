# task runner function for simulate_device class
from time import gmtime, strftime
import traceback
import sys

sys.path.insert(0,'..')

from ovm_utils.device_storing_util import load_device, save_device
from dist_swarm.db_bridge.device_in_db import DeviceInDB
from dist_swarm.db_bridge.worker_in_db import WorkerInDB
from dynamo_db import TASK_END, TASK_FAILED, TASK_START

# @TODO implement timeout here
def run_task(worker_id, task_config):
    swarm_name = task_config['swarm_name']
    worker_in_db = WorkerInDB(swarm_name, worker_id)
    worker_in_db.append_history(strftime("%Y-%m-%d %H:%M:%S", gmtime()), TASK_START, task_config)

    try:
        # change the state of current worker
        task_id = task_config['task_id']
        learner_id = task_config['learner']
        neighbor_ids = task_config['neighbors']
        func_list = task_config['func_list']
        device_load_config = task_config['load_config']
        end = task_config['end']

        # load device
        learner_load_config = {}
        if str(learner_id) in device_load_config:
            learner_load_config = device_load_config[str(learner_id)]
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
    worker_in_db.update_finished_task(task_id)
    return new_history
