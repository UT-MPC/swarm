# devices for heterogeneous environment in terms of model sizes
from collections import Counter
import numpy as np
import time
import copy
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import scipy

import device.base_device
import device.exp_device

from model_weight_utils import SelectWeightsConv, SelectWeightsAdv, SelectWeightsNoWeighting, MomentumSelectWeights, MomentumSelectWeightsConv
from model_weight_utils import gradients, multiply_weights, enlarge_weights, add_weights

class HeteroDevice(device.exp_device.GreedyWOSimDevice):
    def __init__(self, *args):
        super().__init__(*args)
        self._num_requests = 0
        if 'device-power' not in self._hyperparams:
            self.device_power = self._get_number_from_dist(self._hyperparams['hetero-lower-bound'],
                                self._hyperparams['hetero-upper-bound'],
                                self._hyperparams['hetero-distribution'],
                                self._hyperparams['device-power-interval'])
        else:
            self.device_power = self._hyperparams['device-power']
        # set model size
        self.model_size = self._hyperparams['model-size']
        
    
    def delegate(self, other, epoch, iteration):
        """
        only fits to other's data
        """
        if other.device_power >= self.model_size:
            for _ in range(iteration):
                self._weights = self.fit_to(other, epoch)

    def _get_model(self):
        model = self._model_fn(size=self.model_size)
        model.set_weights(self._weights)
        self._compile_config['optimizer'] = self._opt_fn(lr=self._hyperparams['orig-lr'])
        model.compile(**self._compile_config)
        return model
    
    def _get_number_from_dist(self, lower, upper, dist, manual):
        vals = np.arange(lower, upper+1, 1)

        if dist == 'uniform':
            probs = np.ones(upper-lower+1)
        elif dist == '-x':
            probs = np.arange(upper, lower-1, 1)
        elif dist == '1/x':
            probs = 1/np.arange(lower, upper+1, 1)
        elif dist == '2-3':
            probs = np.array([2,1])
        elif dist == '5-10':
            probs = np.array([1,1])
            vals = np.array([5,10])
        elif dist == 'manual':
            return manual[self._hyperparams['repeated-number']]
        else:
            probs = np.ones(upper-lower+1)

        p = probs/np.sum(probs)
        # t = time.time()
        # np.random.seed(int(t))
        return np.random.choice(vals, 1, p=p)[0]

class MixedDropoutDevice(HeteroDevice):
    def __init__(self, *args):
        super().__init__(*args)
        self.requests = {}
        for i in range(self.model_size+1):
            self.requests[i] = 0
        self.batch_size = self._hyperparams['batch-size']
        self.optimizer_params = self._hyperparams['optimizer-params']
        self.optimizer_fn = keras.optimizers.Adam
        self.optimizer_weights = None
        if self._hyperparams['dataset'] == 'mnist':
            self.weight_selector = SelectWeightsAdv()
        elif self._hyperparams['dataset'] == 'cifar':
            self.weight_selector = SelectWeightsConv()
        elif self._hyperparams['dataset'] == 'svhn':
            self.weight_selector = SelectWeightsConv()
        self.comm_cost = 0
        self.comp_cost = 0
        self.flops = {1: 0, 2: 0,
                    3: 24273,
                    4: 32563,
                    5: 40953,
                    6: 49443,
                    7: 58033,
                    8: 66723,
                    9: 75513,
                    10: 84403}

        self.criteria = False

    def delegate(self, other, epoch, iteration):
        """
        only fits to other's data
        """
        # print('model_size: {}, others_device_power: {}'.format(self.model_size, other.device_power))
        self._num_requests += 1
        # if other.device_power >= self.model_size:
        #     for _ in range(iteration):
        #         self._weights = self.fit_to(other, epoch)
        #         self.add_comm_cost(self.model_size)
        #     self.requests[other.device_power] += 1
        if other.device_power >= self._hyperparams['downto']:
            # if self._num_requests > 3000 * 3:
                # print('device power: {}'.format(other.device_power))
            for _ in range(iteration):
                self.fit_to_submodel(other, epoch)
                self.add_comm_cost(other.device_power)
            self.requests[other.device_power] += 1

    def add_comm_cost(self, model_size, bits=64):
        submodel = self._model_fn(size=model_size)
        self.comm_cost += submodel.count_params() * bits
        K.clear_session()
        self.comp_cost += self.flops[model_size]
            
    def eval(self):
        hist = super().eval()
        if hist[1] > 0.9 and not self.criteria:
            self.requests['reached 0.9 at'] = str(self.requests)
            self.criteria = True
        hist.append(self.comm_cost)
        hist.append(self.comp_cost)
        return hist

    def sample_batch(self):
        X, y = self.train_data_provider.get_random(self.batch_size)
        return X, y

    def _get_optimizer(self, model):
        opt = self._opt_fn(**self._hyperparams['optimizer-params'])
        zero_grads = [tf.zeros_like(w) for w in model.trainable_variables]
        opt.apply_gradients(zip(zero_grads, model.trainable_variables))

        if self.optimizer_weights != None:
            opt.set_weights(self.optimizer_weights)
        else:
            self.optimizer_weights = opt.get_weights()

        del model
        return opt

    def fit_to_submodel(self, other, epoch, size=None, scale=1):
        # get submodel and select random (dropout) weights
        if size == None:
            size = other.device_power
        submodel = self._model_fn(size=size)
        # if self._hyperparams['dataset'] == 'mnist':
        #     sw = SelectWeights(self._weights, submodel.get_weights())
        # elif self._hyperparams['dataset'] == 'cifar':
        #     sw = SelectWeightsConv(self._weights, submodel.get_weights())
        sub_weights = self.weight_selector.select_weights(self._weights, submodel.get_weights())
        submodel.set_weights(sub_weights)

        # train submodel
        X = other._x_train
        y = other._y_train
        with tf.GradientTape() as tape:
            pred = submodel(X)
            loss = keras.metrics.categorical_crossentropy(y, pred)
            # loss = keras.metrics.mean_squared_error(y, pred)

        grads = tape.gradient(loss, submodel.trainable_variables)
        grads_val = [g.numpy() for g in grads]

        # scale gradients
        grads_val = multiply_weights(grads_val, scale)

        # convert gradients to the original size
        big_grads = self.weight_selector.get_target_from_selected(grads_val)
        model = self._get_model()
        opt = self._get_optimizer(model)
        opt.apply_gradients(zip(big_grads, model.trainable_variables))
        self._weights = model.get_weights()

        # save optimizer state
        self.optimizer_weights = opt.get_weights()
        K.clear_session()

class MomentumMixedDropoutDevice(MixedDropoutDevice):
    def __init__(self, *args):
        super().__init__(*args)
        if self._hyperparams['dataset'] == 'mnist':
            self.weight_selector = MomentumSelectWeights()
            self.orig_weight_selector = SelectWeightsAdv()
        elif self._hyperparams['dataset'] == 'cifar':
            self.weight_selector = MomentumSelectWeightsConv()
        elif self._hyperparams['dataset'] == 'svhn':
            self.weight_selector = MomentumSelectWeightsConv()

    def fit_to_submodel(self, other, epoch, size=None, scale=1):
        # get submodel and select random (dropout) weights
        if size == None:
            size = other.device_power
        submodel = self._model_fn(size=size)
        sub_weights = self.weight_selector.select_weights(self._weights, submodel.get_weights())
        submodel.set_weights(sub_weights)

        # train submodel
        X = other._x_train
        y = other._y_train
        CL = 0.001
        with tf.GradientTape() as tape:
            pred = submodel(X)
            for i in range(0,len(submodel.trainable_variables)):
                if i==0:
                    l2_loss = CL * tf.nn.l2_loss(submodel.trainable_variables[i])
                if i>=1:
                    l2_loss = l2_loss + CL * tf.nn.l2_loss(submodel.trainable_variables[i])
            loss = keras.metrics.categorical_crossentropy(y, pred)
            loss += l2_loss

        grads = tape.gradient(loss, submodel.trainable_variables)
        grads_val = [g.numpy() for g in grads]

        # scale gradients
        grads_val = multiply_weights(grads_val, scale)

        # convert gradients to the original size
        if self.optimizer_weights == None:
            big_grads = self.weight_selector.get_target_from_selected(grads_val)
        else:
            big_grads = self.weight_selector.get_target_from_selected_w_momentum(self.optimizer_weights[1:len(grads_val)+1], grads_val)
        big_grads_w_zero = self.weight_selector.get_target_from_selected(grads_val)

        model = self._get_model()
        opt = self._get_optimizer(model)
        opt.apply_gradients(zip(big_grads, model.trainable_variables))
        self._weights = model.get_weights()
        self.optimizer_weights = opt.get_weights()
        K.clear_session()

class AutoMomentumMixedDropoutDevice(MomentumMixedDropoutDevice):
    def __init__(self, *args):
        super().__init__(*args)
        self.window = []
        self.WINDOW_SIZE = 30
        lower = self._hyperparams['hetero-lower-bound']
        upper = self._hyperparams['hetero-upper-bound']
        self.device_power_to_idx = {}

        self.num_params = {}
        n = 0
        param_max = 0
        for i in range(lower, upper+1, 1):
            self.device_power_to_idx[i] = n
            n += 1
            m = self._model_fn(size=i)
            self.num_params[i] = m.count_params()
            param_max = max(param_max, self.num_params[i])
            del m
        self.model_num_params = param_max

    def delegate(self, other, epoch, iteration):
        if len(self.window) < self.WINDOW_SIZE:
            super().delegate(other, epoch, iteration)
        else:
            count = Counter(self.window)
            d_l = self._hyperparams['hetero-upper-bound']
            d_l_to_res = {}
            for d in self.device_power_to_idx:
                p = count[d] / self.WINDOW_SIZE
                target = self.model_num_params/self.num_params[d]
                mean = self.WINDOW_SIZE * p
                var = self.WINDOW_SIZE * p * (1-p)
                z = (target - mean)/np.sqrt(var)
                res = 1 - scipy.stats.norm.cdf(z)
                d_l_to_res[d] = res
                if res > 0.5 and d < d_l:
                    d_l = d
            print(d_l_to_res)
            print('selected: {}'.format(d_l))
            if d_l < other.device_power:
                super().delegate(other, epoch, iteration)
            self.window.pop(0)
            
        self.window.append(other.device_power)

class AutoMomentumMixedDropoutDevice_OLD(MomentumMixedDropoutDevice):
    def __init__(self, *args):
        super().__init__(*args)
        self.window = []
        self.WINDOW_SIZE = 30
        lower = self._hyperparams['hetero-lower-bound']
        upper = self._hyperparams['hetero-upper-bound']
        self.device_power_to_idx = {}

        num_params = {}
        n = 0
        param_max = 0
        for i in range(lower, upper+1, 1):
            self.device_power_to_idx[i] = n
            n += 1
            m = self._model_fn(size=i)
            num_params[i] = m.count_params()
            param_max = max(param_max, num_params[i])
            del m
        
        # precompute optimal dropout rates
        self.pp_min = np.sqrt(1 / np.array(list(num_params.values())))
        K = np.sum(self.pp_min)
        self.pp_min /= K

    def delegate(self, other, epoch, iteration):
        if len(self.window) < self.WINDOW_SIZE:
            super().delegate(other, epoch, iteration)
        else:
            count = Counter(self.window)
            d_l = self._hyperparams['hetero-upper-bound']
            d_to_min_and_prob = {}
            for d in self.device_power_to_idx:
                p_min = self.pp_min[self.device_power_to_idx[d]]
                prob = count[d] / self.WINDOW_SIZE
                d_to_min_and_prob[d] = (p_min, prob)
                if p_min < prob and d < d_l:
                    d_l = d
            print(d_to_min_and_prob)
            print('selected: {}'.format(d_l))
            if d_l < other.device_power:
                super().delegate(other, epoch, iteration)
            self.window.pop(0)
            
        self.window.append(other.device_power)

class NoDropoutDevice(MixedDropoutDevice):
    def __init__(self, *args):
        super().__init__(*args)

    def delegate(self, other, epoch, iteration):
        """
        only fits to other's data
        """
        for _ in range(iteration):
            self._weights = self.fit_to(other, epoch)
            self.add_comm_cost(self.model_size)
        self.requests[other.device_power] += 1

class DropoutOnlyOnDevice(MomentumMixedDropoutDevice):
    def __init__(self, *args):
        super().__init__(*args)

    def delegate(self, other, epoch, iteration):
        """
        only fits to other's data
        """
        # print('model_size: {}, others_device_power: {}'.format(self.model_size, other.device_power))
        self._num_requests += 1

        self.requests[other.device_power] += 1
        # if self._num_requests > 3000 * 3:
            # print('device power: {}'.format(other.device_power))
        for _ in range(iteration):
            self.fit_to_submodel(other, epoch, self._hyperparams['downto'])
            self.add_comm_cost(other.device_power)