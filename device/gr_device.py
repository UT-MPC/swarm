# Gradient replay
import copy
import numpy as np
import model_distance as md
import tensorflow.keras.backend as K

import device.opportunistic_device
from data_process import get_even_prob, JSD
from model_weight_utils import gradients, avg_weights, multiply_weights, add_weights

class JSDGradientReplayDevice(device.opportunistic_device.JSDOppStrategy):
    def __init__(self, *args):
        super().__init__(*args)
        self._cache_comb = []  # list of (label_sets, gradients, weights)
        self.cache_comb_decay_total_updates = 0  
        self.init_weights = copy.deepcopy(self._weights)
        self.lr_fac_min = 1
        unknown_set = set(self._desired_data_dist.keys()).difference(set(self._local_data_dist.keys()))
        self.desired_prob = {}
        for l in unknown_set:
            self.desired_prob[l] = 1./len(unknown_set)
        self.local_weight = np.exp(-8*JSD(get_even_prob(set(self._local_data_dist)), self._desired_data_dist, self._num_classes))
        self.local_decay = 1
        self.local_apply_rate = 1

    def delegate(self, other, epoch, iteration=1):
        if not self.decide_delegation(other):
            return
        drift = md.l2_distance_w(self._weights, self.init_weights)
        xx = self._hyperparams['kappa']*(-(drift-self._hyperparams['offset']))
        lr_fac = np.exp(xx)/(np.exp(xx) + 1)
        self.lr_fac_min = min(self.lr_fac_min, lr_fac)
        lr = self.lr_fac_min * self._hyperparams['orig-lr']

        lr = self._hyperparams['orig-lr']

        new_weights = self.fit_w_lr_to(other, epoch, lr)
        grads = gradients(self._weights, new_weights)
        # if set(other._local_data_dist.keys()).issubset(set(self._desired_data_dist.keys())):
        # update cache
        found_subsets = [] # smaller or equal
        found_supersets = [] # bigger
        for c in range(len(self._cache_comb)):
            if (set(self._cache_comb[c][0])).issubset(set(other._local_data_dist.keys())):
                found_subsets.append(c)
            elif (set(self._cache_comb[c][0])).issuperset(set(other._local_data_dist.keys())) and \
                    len(set(self._cache_comb[c][0]).difference(set(other._local_data_dist.keys()))) != 0:
                found_supersets.append(c)

        if len(found_supersets) == 0:
            if len(found_subsets) != 0:
                for c in sorted(found_subsets, reverse=True):
                    del self._cache_comb[c]
            weight = np.exp(-8*JSD(get_even_prob(set(other._local_data_dist.keys())), self._desired_data_dist, self._num_classes))
            self._cache_comb.append([set(other._local_data_dist.keys()), grads, weight])

        else: # @TODO this is where I'm not too sure about
            for c in found_supersets:
                self._cache_comb[c][1] = avg_weights(self._cache_comb[c][1], grads)

        not_stale_list = []
        avg_lr = 0
        if len(self._cache_comb) > 0:
            agg_g = None
            for cc in self._cache_comb:
                agg_g = add_weights(agg_g, multiply_weights(cc[1], cc[2]))
                avg_lr += cc[2]
                cc[2] *= self._hyperparams['decay'] # @TODO add this to hyperparams
                if cc[2] > 0.005:
                    not_stale_list.append(cc)
            # remove stale gradients from the data structure
            avg_lr /= len(self._cache_comb)
            self._cache_comb = not_stale_list
            
            # aggregate weights
            agg_g = multiply_weights(agg_g, self._hyperparams['apply-rate'])

            # do training
            for _ in range(iteration):
                self._weights = add_weights(self._weights, agg_g)
                new_weights = self.fit_w_lr_to(self, epoch, lr)
                grads = gradients(self._weights, new_weights)
                self._weights = add_weights(self._weights, multiply_weights(grads, self.local_weight * self.local_apply_rate * self._hyperparams['apply-rate']))
                self.local_apply_rate *= self.local_decay

        # else:
        #     for _ in range(iteration):
        #         self._weights = self.fit_w_lr_to(self, 1, lr)

        K.clear_session()