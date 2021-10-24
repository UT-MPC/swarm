# devices that are used in ablation studies in TMC papers
import numpy as np

import device.opportunistic_device
from model_weight_utils import gradients, multiply_weights, add_weights
from data_process import get_even_prob, JSD

class JSDOppWeightedDevice(device.opportunistic_device.JSDOppDevice):
    def __init__(self, *args):
        super().__init__(*args)
        self.fac_agg = 0

    def delegate(self, other, epoch, iteration):
        if not self.decide_delegation(other):
            return
        # print("greedy sim encorporate with {}".format(other._local_data_dist))
        for _ in range(iteration):
            fac = np.exp(-8*JSD(get_even_prob(set(other._local_data_dist.keys())), self._desired_data_dist, self._num_classes))
            update = self.fit_to(other, epoch)
            grads = gradients(self._weights, update)
            agg = multiply_weights(grads, fac*self._hyperparams['apply-rate']*self._hyperparams['opportunistic-weighted-apply-rate'])
            self._weights = add_weights(self._weights, agg)
            # print("{}: {}".format(set(other._local_data_dist.keys()), fac))
            self.fac_agg += fac

            fac = np.exp(-8*JSD(get_even_prob(set(self._local_data_dist.keys())), self._desired_data_dist, self._num_classes))
            update = self.fit_to(self, epoch)
            grads = gradients(self._weights, update)
            agg = multiply_weights(grads, fac*self._hyperparams['apply-rate']*self._hyperparams['opportunistic-weighted-apply-rate'])
            self._weights = add_weights(self._weights, agg)
            self.fac_agg += fac
            print('weighted: {}'.format(self.fac_agg))