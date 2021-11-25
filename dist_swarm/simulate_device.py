import sys
sys.path.insert(0,'..')
import os
import json
from io import StringIO

from dist_swarm.model_in_db import ModelInDB
from dist_device import DistDevice
import grpc_components

# data frame column names for encounter data
TIME_START="time_start"
TIME_END="time_end"
CLIENT1="client1"
CLIENT2="client2"
ENC_IDX="encounter index"

class SimulateDeviceServicer(grpc_components.simulate_device_pb2_grpc.SimulateDeviceServicer):
    ### gRPC methods
    def SimulateOppCL(self, request, context):
        device = DistDevice(json.load(StringIO(request.config)))
        return device.run()
        
        