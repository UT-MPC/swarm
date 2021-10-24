# criteria for opportunistic incorporation using statistical distance
import numpy as np

import device.base_device
from data_process import get_kl_div, JSD

class SimpleOppStrategy(device.base_device.Device):
    """
    uses a simple similarity metric (PerCom 2021)
    """
    def __init__(self, *args):
        super().__init__(*args)

    def get_similarity(self, other):
        accum = 0.
        for k in self._desired_data_dist.keys():
            if k in other._local_data_dist:
                accum += min(self._desired_data_dist[k], other._local_data_dist[k])
        return accum

    def decide_delegation(self, other):
        return self.get_similarity(other) >= self._similarity_threshold

class KLOppStrategy(device.base_device.Device):
    """
    KL Divergence
    """
    def __init__(self, *args):
        super().__init__(*args)

    def get_similarity(self, other):
        return np.exp(-8*get_kl_div(other._local_data_dist, self._desired_data_dist, self._num_classes))
    
    def decide_delegation(self, other):
        kl_div = self.get_similarity(other)
        if kl_div != np.nan and kl_div != np.inf:
            return kl_div <= 0.95
        return False

class JSDOppStrategy(device.base_device.Device):
    def __init__(self, *args):
        super().__init__(*args)

    def get_similarity(self, other):
        # print("other: {}".format(other._local_data_dist.keys()))
        return JSD(other._local_data_dist, self._desired_data_dist, self._num_classes)
    
    def decide_delegation(self, other):
        jsd = self.get_similarity(other)
        if jsd != np.nan and jsd != np.inf:
            return jsd <= self._similarity_threshold
        return False

class LowJSDStrategy(device.base_device.Device):
    """
    lower threshold -> more strict criteria
    """
    def __init__(self, *args):
        super().__init__(*args)

    def get_similarity(self, other):
        # print("other: {}".format(other._local_data_dist.keys()))
        return JSD(other._local_data_dist, self._desired_data_dist, self._num_classes)
    
    def decide_delegation(self, other):
        jsd = self.get_similarity(other)
        if jsd != np.nan and jsd != np.inf:
            return jsd <= self._similarity_threshold
        return False

class SimpleOppDevice(SimpleOppStrategy):
    def __init__(self, *args):
        super().__init__(*args)

    def delegate(self, other, epoch, iteration):
        if not self.decide_delegation(other):
            return
        # print("greedy sim encorporate with {}".format(other._local_data_dist))
        for _ in range(iteration):
            self._weights = self.fit_to(other, 1)
            self._weights = self.fit_to(self, 1)

class KLOppDevice(KLOppStrategy):
    def __init__(self, *args):
        super().__init__(*args)

    def delegate(self, other, epoch, iteration):
        if not self.decide_delegation(other):
            return
        # print("greedy sim encorporate with {}".format(other._local_data_dist))
        for _ in range(iteration):
            self._weights = self.fit_to(other, 1)
            self._weights = self.fit_to(self, 1)

class JSDOppDevice(JSDOppStrategy):
    def __init__(self, *args):
        super().__init__(*args)
        self.fac_agg = 0

    def delegate(self, other, epoch, iteration):
        if not self.decide_delegation(other):
            return
        # print("greedy sim encorporate with {}".format(other._local_data_dist))
        for _ in range(iteration):
            self._weights = self.fit_to(other, epoch)
            self._weights = self.fit_to(self, epoch)
            self.fac_agg += 2
