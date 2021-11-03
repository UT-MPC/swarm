import sys
sys.path.insert(0,'grpc_components')
import logging
import json
import grpc
from grpc_components import simulate_device_pb2, simulate_device_pb2_grpc

def run():
    with open('configs/dist_swarm/controller_example.json', 'rb') as f:
        config_json = f.read()
    config_json = json.loads(config_json)

    with grpc.insecure_channel('localhost:50051') as channel:
        stub = simulate_device_pb2_grpc.SimulateDeviceStub(channel)
        config = simulate_device_pb2.Config(config=json.dumps(config_json))
        status = stub.InitDevice(config)
        print(status)
        empty = simulate_device_pb2.Empty()
        status = stub.StartOppCL(empty)
        print(status)

if __name__ == '__main__':
    logging.basicConfig()
    run()