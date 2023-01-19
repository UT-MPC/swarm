# devices for experiments -- TMC
import sys
sys.path.insert(0,'..')
import device.base_device
from dist_swarm.ovm_utils.dependency_utils import get_dependency_dict

class LocalDevice(device.base_device.Device):
    def __init__(self, *args):
        super().__init__(*args)

    def delegate(self, other, epoch, iteration):
        # not in fact delegation at all
        for _ in range(iteration):
            self._weights = self.fit_to(self, 1)

    def decide_delegation(self, other):
        return True

class GreedyValidationClient(device.base_device.Device):
    def __init__(self, *args):
        super().__init__(*args)

    def delegate(self, other, epoch, iteration, fit_to_batch=False):
        new_weights = self.fit_weights_to(self._weights, other, 1)
        new_weights = self.fit_weights_to(new_weights, self, 1)
        hist = self.eval_weights(new_weights)
        if self._last_hist[0] > hist[0]:
            self._weights = new_weights
            for _ in range(iteration-1):
                if fit_to_batch:
                    self._weights = self.fit_to_batch(other, 1)
                    self._weights = self.fit_to_batch(self, 1)
                else:
                    self._weights = self.fit_to(other, 1)
                    self._weights = self.fit_to(self, 1)

class GreedyWOSimDevice(device.base_device.Device):
    """
    does not consider similarity
    """
    def __init__(self, *args):
        super().__init__(*args)

    def delegate(self, other, epoch, iteration):
        for _ in range(iteration):
            self._weights = self.fit_to(other, epoch)
            self._weights = self.fit_to(self, epoch)

    def on_connect(self):
        # simply returns the name of the function to be invoked
        # when encountering other device
        return "delegate"

    @staticmethod
    def get_dependency():
        return get_dependency_dict(on_immutable=True, on_mutable=False)

class OracleClient(device.base_device.Device):
    def __init__(self, *args):
        super().__init__(*args)

    def delegate(self, other, epoch, iteration):
        if other._id_num == 100 + self.task_num:
            for _ in range(iteration):
                self._weights = self.fit_to(other, epoch)