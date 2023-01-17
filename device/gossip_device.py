from audioop import add
import device.base_device
from model_weight_utils import add_weights, multiply_weights

class OVMGossipDevice(device.base_device.Device):
    def __init__(self, *args):
        super().__init__(*args)

    def update_model(self, other):
        added_weights = add_weights(self._weights, other._weights)
        added_weights = multiply_weights(added_weights, 1/2)
        self._weights = added_weights

    def train_self(self):
        self.fit_to(self, 1)