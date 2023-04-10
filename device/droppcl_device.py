import tensorflow as tf
from droppcl_swarm import DROppCLSwarm
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import numpy as np
from collections import Counter
import scipy
import logging

import device.base_device
import device.hetero_device
import device.quantization_device
from model_weight_utils import quant, grad_to_vec, vec_to_grad, weights_to_vec
from model_weight_utils import MomentumSelectWeights, MomentumSelectWeightsConv

class DROppCLDevice(device.base_device.Device):
    def __init__(self, *args):
        super().__init__(*args)
        self.window = []
        self.WINDOW_SIZE = 30
        self.model_size = self._hyperparams['enc-exp-config']['client-sizes']['model-sizes'][self._id_num%5]
        self.device_power = self._hyperparams['enc-exp-config']['client-sizes']['device-powers'][self._id_num%5]
        self.num_params = self._hyperparams['enc-exp-config']['model-params']
        self.possible_device_powers = [int(l) for l in list(self.num_params.keys())]
        self.hetero_upper_bound = max(self.possible_device_powers)
        self.model_num_params = self.num_params[str(self.model_size)]

        self.optimizer_params = self._hyperparams['optimizer-params']
        self.optimizer_weights = None

        if self._hyperparams['dataset'] == 'mnist':
            self.weight_selector = MomentumSelectWeights()
        elif self._hyperparams['dataset'] == 'cifar':
            self.weight_selector = MomentumSelectWeightsConv()
        elif self._hyperparams['dataset'] == 'svhn':
            self.weight_selector = MomentumSelectWeightsConv()

        # downsample from init_weights to meet with model_size
        if self._model_fn is not None:
            my_model = self._model_fn(size=self.model_size)
            self._weights = self.weight_selector.select_weights(self._weights, my_model.get_weights())
        K.clear_session()

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

    def _get_model(self):
        tf.compat.v1.reset_default_graph()
        K.clear_session()
        model = self._model_fn(size=self.model_size)
        model.set_weights(self._weights)
        self._compile_config['optimizer'] = self._opt_fn(lr=self._hyperparams['orig-lr'])
        model.compile(**self._compile_config)
        return model

    def quantize(self, grads, bits):
        vec = grad_to_vec(grads)
        quantized = quant(vec, bits)
        return vec_to_grad(quantized, grads)

    def get_d_l(self, other):
        if other.device_power == 0:
            return 100000
        if len(self.window) < self.WINDOW_SIZE:
            d_l = other.device_power
            self.window.append(other.device_power)
        else:
            count = Counter(self.window)
            self.window.pop(0)
            self.window.append(other.device_power)
            d_l = self.hetero_upper_bound
            for d in self.possible_device_powers:
                p = count[d] / self.WINDOW_SIZE
                target = self.model_num_params / self.num_params[str(d)]
                mean = self.WINDOW_SIZE * p
                var = self.WINDOW_SIZE * p * (1-p)
                z = (target - mean)/np.sqrt(var)
                res = 1 - scipy.stats.norm.cdf(z)
                if res > 0.5 and d < d_l:
                    d_l = d
        return d_l

    def get_q_and_iteration(self, d_l, time_duration):
        computation_duration = self._hyperparams['computation-time']
        time_left = time_duration - computation_duration
        num_params = self.num_params[str(self.model_size)] * d_l / 10
        bits_to = time_left * self._hyperparams['communication-rate'] / (2 * num_params)
        if bits_to < 1:
            return (0, 0)
        n = int(np.log2(bits_to))

        if (n >= 6): # if no need of quantization for one round
            iteration_mx = int((time_left * self._hyperparams['communication-rate']) / (2 * num_params * pow(2, 6)))
            iteration = min(self._hyperparams['max-iteration'], iteration_mx)
            n = 6
        elif n <= self._hyperparams['minimum-bits-to']: # lower bound == 2^(min quantization bits-to)
            return (0, 0)
        else:
            iteration = 1

        return iteration, n

    def hetero_delegate(self, other, epoch, time_duration):
        ######### determine d_l
        d_l = self.get_d_l(other)
        if d_l > other.device_power:
            return (0, 0, 0)

        ######## determine quantization bits
        iteration, n = self.get_q_and_iteration(d_l, time_duration)
        
        ####### compute gradients and update the model
        for _ in range(iteration):
            self.fit_w_drop_quant(other, epoch, d_l, pow(2, n))
        
        return iteration, d_l, n

    def fit_w_drop_quant(self, other, epoch, d_l, q):
        """
        fit the model with mixed dropout or quantization
        no dropout if d_l == my model size
        no quantization if q == 64
        """
        logging.info('running training {}->{}'.format(self._id_num, other._id_num))
        if d_l >= self.model_size:
            d_l = self.model_size
        # logging.info('d_l: {}'.format(d_l))
        submodel = self._model_fn(size=d_l)
        sub_weights = self.weight_selector.select_weights(self._weights, submodel.get_weights())
        if q >= 64:
            quantized_weights = sub_weights
        else:
            quantized_weights = vec_to_grad(quant(weights_to_vec(sub_weights), q), sub_weights)
        submodel.set_weights(quantized_weights)

        X = other._x_train
        y = other._y_train
        CL = 0.001
        with tf.GradientTape() as tape:
            pred = submodel(X)
            for i in range(0,len(submodel.trainable_variables)):
                if i==0:
                    l2_loss = CL * tf.nn.l2_loss(submodel.trainable_variables[i])
                if i>=1:
                    l2_loss = l2_loss+ CL * tf.nn.l2_loss(submodel.trainable_variables[i])
            loss = keras.metrics.categorical_crossentropy(y, pred)
            loss += l2_loss

        grads = tape.gradient(loss, submodel.trainable_variables)
        if q == 64:
            quantized_grads_val = [g.numpy() for g in grads]
        elif q < 64:
            quantized_grads_val = self.quantize(grads, q)
        else:
            raise ValueError('quantization bits: {}, cannot be bigger than 64'.format(q))
        
        # convert gradients to the original size
        if self.optimizer_weights == None:
            big_grads = self.weight_selector.get_target_from_selected(quantized_grads_val)
        else:
            big_grads = self.weight_selector.get_target_from_selected_w_momentum(self.optimizer_weights[1:len(quantized_grads_val)+1], quantized_grads_val)

        model = self._get_model()
        opt = self._get_optimizer(model)
        opt.apply_gradients(zip(big_grads, model.trainable_variables))
        self._weights = model.get_weights()
        self.optimizer_weights = opt.get_weights()

        ########### @TODO change this
        # other.resample_local_data()

        tf.compat.v1.reset_default_graph()
        K.clear_session()
    
class DROppCLTestDevice(DROppCLDevice):
    def __init__(self, *args):
        super().__init__(*args)

    def hetero_delegate(self, other, epoch, time_duration):      
        ########## determine d_l
        d_l = self.get_d_l(other)
        if d_l > other.device_power:
            return (0, 0, 0)

        ######## determine quantization bits
        iteration, n = self.get_q_and_iteration(d_l, time_duration)
        if (n > 0):
            print('iter: {}, d_l: {}, n: {}'.format(iteration, d_l, n))
        return iteration, d_l, n

class DROppCLBaselineDevice(DROppCLDevice):
    def __init__(self, *args):
        super().__init__(*args)

    def hetero_delegate(self, other, epoch, time_duration):      
        ########## determine d_l
        if self.model_size > other.device_power:
            return (0, 0, 0)

        d_l = self.model_size
        ######## determine quantization bits
        iteration, n = self.get_q_and_iteration(d_l, time_duration)
        if n >= 6:
            for _ in range(iteration):
                self.fit_w_drop_quant(other, epoch, d_l, pow(2, 6))
            return iteration, d_l, n
        else:
            return (0, 0, 0)

class DROppCLOnlyDropoutDevice(DROppCLDevice):
    def __init__(self, *args):
        super().__init__(*args)

    def hetero_delegate(self, other, epoch, time_duration):      
        ########## determine d_l
        d_l = self.get_d_l(other)
        if d_l > other.device_power:
            return (0, 0, 0)

        ######## determine quantization bits
        iteration, n = self.get_q_and_iteration(d_l, time_duration)

        # only do training when can send with 64 bits 
        if n >= 6:
            for _ in range(iteration):
                self.fit_w_drop_quant(other, epoch, d_l, pow(2, 6))
        else:
            return (0, 0, 0)

        return iteration, d_l, n

class DROppCLOnlyQuantizationDevice(DROppCLDevice):
    def __init__(self, *args):
        super().__init__(*args)

    def hetero_delegate(self, other, epoch, time_duration):      
        ########## determine d_l
        d_l = self.model_size
        if d_l > other.device_power:
            return (0, 0, 0)

        ######## determine quantization bits
        iteration, n = self.get_q_and_iteration(d_l, time_duration)

        if n >= 6:
            for _ in range(iteration):
                self.fit_w_drop_quant(other, epoch, d_l, pow(2, n))
        elif n >= 1: # lower bound == 2^1
            for _ in range(iteration):
                self.fit_w_drop_quant(other, epoch, d_l, pow(2, n))
        else:
            return (0, 0, 0)

        return iteration, d_l, n