import numpy as np
import copy
from dist_swarm.ovm_utils.dependency_utils import get_dependency_dict
from model_weight_utils import add_weights, multiply_weights
import tensorflow.keras as K

import device.base_device
import device.opportunistic_device

class FederatedDevice(device.base_device.Device):
    def __init__(self, *args):
        super().__init__(*args)
        self.encountered_clients = {}

    def delegate(self, other, *args):
        self.encountered_clients[other._id_num] = other

    def federated_round(self, epoch):
        updates = []
        for k in self.encountered_clients:
            updates.append(self.fit_to(self.encountered_clients[k], epoch))
        agg_weights = list()
        for weights_list_tuple in zip(*updates):
            agg_weights.append(np.array([np.average(np.array(w), axis=0) for w in zip(*weights_list_tuple)]))
        self._weights = agg_weights

    def is_federated(self):
        return True

    def eval(self):
        return [0]
    
    def federated_eval(self):
        if self._evaluation_metrics == 'loss-and-accuracy':
            return self.eval_loss_and_accuracy()
        elif self._evaluation_metrics == 'f1-score-weighted':
            return self.eval_f1_score()
        elif self._evaluation_metrics == 'split-f1-score-weighted':
            return self.eval_split_f1_score()
        else:
            raise ValueError('evaluation metrics is invalid: {}'.format(self._evaluation_metrics))

class FederatedJSDGreedyDevice(device.opportunistic_device.JSDOppStrategy):
    def __init__(self, *args):
        super().__init__(*args)
        self.other_x_trains = None
        self.other_y_trains = None

    def delegate(self, other, *args):
        if not self.decide_delegation(other):
            return
        if self.other_x_trains == None or self.other_y_trains == None:
            self.other_x_trains = copy.deepcopy(other._x_train)
            self.other_y_trains = copy.deepcopy(other._y_train)
        else:
            self.other_x_trains = np.concatenate(self.other_x_trains, other._x_train)
            self.other_y_trains = np.concatenate(self.other_y_trains, other._y_train)

    def train(self, epochs):
        model = self._get_model()
        self._train_config['epochs'] = 1
        self._train_config['x'] = np.concatenate(self.other_x_trains, self._x_train)
        self._train_config['y'] = np.concatenate(self.other_y_trains, self._y_train)
        self._train_config['verbose'] = 0
        self._train_config['shuffle'] = True
        model.fit(**self._train_config)
        weights = copy.deepcopy(model.get_weights())
        K.clear_session()
        del model
        self._weights = weights

class OVMFLServerDevice(device.base_device.Device):
    def __init__(self, *args):
        super().__init__(*args)
        self._init_model_updates()
        self._update_client_num = self._hyperparams['update-client-num']
        self._round_number = 0
        # evaluate here and put in last eval
        self.last_eval_round = -1
        # self.last_eval = self.eval()
        self.last_eval = [0, 0]

    def _init_model_updates(self):
        self._updated_model_nums = 0
        self._updated_averaged_weights = None

    def update_weights(self, other: device.base_device.Device):
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

    def eval(self):
        # return last_eval if we have updated evaluation results
        # if not, evaluate
        # if self.last_eval_round < self._round_number:
        #     self.last_eval_round = self._round_number
        #     self.last_eval = super().eval()
        return self.last_eval

    @staticmethod
    def get_dependency():
        return get_dependency_dict(on_immutable=True, on_mutable=True)

class OVMFLClientDevice(device.base_device.Device):
    def __init__(self, *args):
        super().__init__(*args)

    def fl_update(self, other: OVMFLServerDevice, epoch):
        self._weights = other._weights
        self._weights = self.fit_to(self, epoch)

    @staticmethod
    def get_dependency():
        return get_dependency_dict(on_immutable=True, on_mutable=True)

    def eval(self):
        return [0, 0]