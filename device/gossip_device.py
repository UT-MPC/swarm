from audioop import add
import device.base_device
from dist_swarm.ovm_utils.dependency_utils import get_dependency_dict
from model_weight_utils import add_weights, multiply_weights

class OVMGossipDevice(device.base_device.Device):
    def __init__(self, *args):
        super().__init__(*args)

    def update_model(self, other):
        added_weights = add_weights(self._weights, other._weights)
        added_weights = multiply_weights(added_weights, 1/2)
        self._weights = added_weights

    def train_self(self, other):
        self.fit_to(self, 1)

    @staticmethod
    def get_dependency():
        return get_dependency_dict(on_immutable=True, on_mutable=True)

class OVMBroadcastGossipDevice(device.base_device.Device):
    def __init__(self, *args):
        super().__init__(*args)
        self._init_model_updates()
        self._update_client_num = 10
        self._round_number = 0
        self.last_eval_round = -1
        self.last_eval = self.eval()
    
    def _init_model_updates(self):
        self._updated_model_nums = 0
        self._updated_averaged_weights = None

    def update_model(self, other):
        # receive model update
        if self._updated_averaged_weights == None:
            self._updated_averaged_weights = other._weights
        else:
            self._updated_averaged_weights = multiply_weights(self._updated_averaged_weights, self._updated_model_nums)
            added_weights = add_weights(self._updated_averaged_weights, other._weights)
            added_weights = multiply_weights(added_weights, 1/(self._updated_model_nums+1))
            self._updated_averaged_weights = added_weights
            
        self._updated_model_nums += 1
        
        if self._updated_model_nums >= self._update_client_num:
            self._weights = self._updated_averaged_weights
            self._init_model_updates()
            self._round_number += 1

    def train_self(self, other):
        self.fit_to(self, 1)

    def eval(self):
        # return last_eval if we have updated evaluation results
        # if not, evaluate
        if self.last_eval_round < self._round_number:
            self.last_eval_round = self._round_number
            self.last_eval = super().eval()
        return self.last_eval

    @staticmethod
    def get_dependency():
        return get_dependency_dict(on_immutable=True, on_mutable=True)