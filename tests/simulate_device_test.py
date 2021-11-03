import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../grpc_components')
import unittest
from simulate_device import SimulateDeviceServicer
import json
import logging
from grpc_components.status import IDLE, RUNNING, ERROR, STOPPED

class SimulateDeviceServicerMethods(unittest.TestCase):
    def setUp(self):
        self.simulate_device_servicer = SimulateDeviceServicer()
        # read config file
        with open('../configs/dist_swarm/controller_example.json', 'rb') as f:
            config_json = f.read()
        self.config = json.loads(config_json)

    # def test_init_dbs_and_device(self):
    #     self.simulate_device_servicer._initialize_dbs(self.config)
    #     self.simulate_device_servicer._initialize_device(self.config, False)

    # def test_set_device_status(self):
    #     self.simulate_device_servicer._initialize_dbs(self.config)
    #     self.simulate_device_servicer._set_device_status(STOPPED)

    # def test_init_device_rpccall(self):
    #     self.simulate_device_servicer.InitDevice(json.dumps(self.config), None)

    def test_init_and_train(self):
        self.simulate_device_servicer.InitDevice(json.dumps(self.config), None)
        self.simulate_device_servicer.StartOppCL(None, None)

    # def test_init_device(self):
        
    #     res = self.simulate_device_servicer.InitDevice(json.dumps(self.config), None)
    #     self.assertEqual(res, STATUS.index(IDLE))

if __name__ == '__main__':
    unittest.main()