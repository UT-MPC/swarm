import sys
sys.path.insert(0,'..')
import os
import json
from io import StringIO

from ovm_utils.task_runner import run_task
from grpc_components import simulate_device_pb2_grpc
from grpc_components.simulate_device_pb2 import Status
from grpc_components.status import IDLE, RUNNING, ERROR, FINISHED
from oppcl_device import OppCLDevice

# data frame column names for encounter data
TIME_START="time_start"
TIME_END="time_end"
CLIENT1="client1"
CLIENT2="client2"
ENC_IDX="encounter index"

class SimulateDeviceServicer(simulate_device_pb2_grpc.SimulateDeviceServicer):
    def __init__(self) -> None:
        super().__init__()
        self.worker_status = IDLE

    ### gRPC methods
    def SetWorkerState(self, request, context):
        self.worker_id = request.worker_id
        return Status(status=self.worker_status)

    def RunTask(self, request, context):
        config = json.load(StringIO(request.config))
        # call the function to run a single training task
        run_task(self.worker_id, config)
        # save the pointer to the thread
    
    def StopTask(self, request, context):
        raise NotImplementedError('')

    def GetStatus(self, request, context):
        return Status(status=self.worker_status)
    
    def SimulateOppCL(self, request, context):
        # deprecated
        # simulate a single oppcl client
        device = OppCLDevice(json.load(StringIO(request.config)))
        return device.run()
        
        
    ### helper methods
    # def run_training_task(config):
        

