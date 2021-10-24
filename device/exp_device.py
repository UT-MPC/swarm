# devices for experiments -- TMC

import device.base_device

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

    def delegate(self, other, epoch, iteration):
        new_weights = self.fit_weights_to(self._weights, other, 1)
        new_weights = self.fit_weights_to(new_weights, self, 1)
        hist = self.eval_weights(new_weights)
        if self._last_hist[0] > hist[0]:
            self._weights = new_weights
            for _ in range(iteration-1):
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

class OracleClient(device.base_device.Device):
    def __init__(self, *args):
        super().__init__(*args)

    def delegate(self, other, epoch, iteration):
        if other._id_num == 100 + self.task_num:
            for _ in range(iteration):
                self._weights = self.fit_to(other, epoch)