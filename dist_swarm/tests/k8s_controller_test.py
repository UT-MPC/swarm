import sys
sys.path.insert(0,'..')
import unittest
import json
import logging

from k8s_controller import K8sController
from eks_controller import EKSController

class K8sTests(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        eks_controller = EKSController()
        self.controller = K8sController(eks_controller)

    def test_init(self):
        self.controller.deploy_workers('default_tag', 3)

if __name__ == '__main__':
    unittest.main()