import numpy as np
import copy
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