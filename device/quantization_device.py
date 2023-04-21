import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
# from qkeras.utils import model_quantize
import numpy as np
from collections import Counter
import scipy

import device.hetero_device
import device.base_device
from model_weight_utils import quant, grad_to_vec, vec_to_grad, weights_to_vec
import models

class QuantizationDevice(device.base_device.Device):
    def __init__(self, *args):
        super().__init__(*args)
        self.model_size = self._hyperparams['model-size']
        self.q_requests = {}
        for i in range(self._hyperparams['duration-lower-bound'], self._hyperparams['duration-upper-bound']+1, 1):
            self.q_requests[i] = 0
        self.batch_size = self._hyperparams['batch-size']
        self.optimizer_params = self._hyperparams['optimizer-params']
        self.optimizer_weights = None

        self._num_requests = 0
        if 'enc-duration' not in self._hyperparams:
            self.enc_duration = self._get_enc_duration()
        else:
            self.enc_duration = self._hyperparams['enc-duration']
        # set model size

    def _get_enc_duration(self):
        lower = self._hyperparams['duration-lower-bound']
        upper = self._hyperparams['duration-upper-bound']
        vals = np.arange(lower, upper+1, 1)
        
        dist = self._hyperparams['duration-distribution']
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
            return self._hyperparams['device-duration-interval'][self._hyperparams['repeated-number']]
        else:
            probs = np.ones(upper-lower+1)

        p = probs/np.sum(probs)
        return np.random.choice(vals, 1, p=p)[0]

    def delegate(self, other, epoch, iteration):
        """
        only fits to other's data
        """
        for _ in range(iteration):
            self.fit_w_quant(other, epoch, pow(2, self._hyperparams['bits']))

    # @TODO add this to the base class
    def fit(self, other, epoch, q):
        self.fit_w_quant_fn(other, epoch, self.no_quantize, q)

    def fit_w_quant(self, other, epoch, q):
        self.fit_w_quant_fn(other, epoch, self.quantize, q)

    def fit_w_quant_fn(self, other, epoch, quant_fn, q):
        model = self._get_model()
        # train model
        X = other._x_train
        y = other._y_train
        CL = 0.01
        with tf.GradientTape() as tape:
            pred = model(X)
            for i in range(0,len(model.trainable_variables)):
                if i==0:
                    l2_loss = CL * tf.nn.l2_loss(model.trainable_variables[i])
                if i>=1:
                    l2_loss = l2_loss+ CL * tf.nn.l2_loss(model.trainable_variables[i])
            loss = keras.metrics.categorical_crossentropy(y, pred)
            loss += l2_loss

        grads = tape.gradient(loss, model.trainable_variables)
        quantized_grads_val = quant_fn(grads, q)
        opt = self._get_optimizer(model)
        opt.apply_gradients(zip(quantized_grads_val, model.trainable_variables))
        self._weights = model.get_weights()

        # save optimizer state
        self.optimizer_weights = opt.get_weights()
        K.clear_session()

    def no_quantize(self, grads, bits):
        return grads

    def quantize(self, grads, bits):
        vec = grad_to_vec(grads)
        quantized = quant(vec, bits)
        return vec_to_grad(quantized, grads)

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

class MixedQuantizationDevice(QuantizationDevice):
    def __init__(self, *args):
        super().__init__(*args)
    def delegate(self, other, epoch, iteration):
        if other.enc_duration >= 6:
            for _ in range(iteration):
                self.fit(other, epoch, 64)
            self.q_requests[other.enc_duration] += 1
        elif other.enc_duration >= self._hyperparams['bits']:
            for _ in range(iteration):
                self.fit_w_quant(other, epoch, pow(2, other.enc_duration))
            self.q_requests[other.enc_duration] += 1

# class QuantizationNetworksDevice(QuantizationDevice):
#     def __init__(self, *args):
#         super().__init__(*args)

#     def delegate(self, other, epoch, iteration):
#         for _ in range(iteration):
#             self.fit_w_quant_net(other, epoch)

#     def fit_w_quant_net(self, other, epoch):
#         model = self._get_model()
#         q_model = model_quantize(model, models.get_Q_2nn_mnist_dict(self.model_size), 4)
#         # train model
#         X = other._x_train
#         y = other._y_train
#         CL = 0.01
#         with tf.GradientTape() as tape:
#             pred = q_model(X)
#             for i in range(0,len(q_model.trainable_variables)):
#                 if i==0:
#                     l2_loss = CL * tf.nn.l2_loss(q_model.trainable_variables[i])
#                 if i>=1:
#                     l2_loss = l2_loss+ CL * tf.nn.l2_loss(q_model.trainable_variables[i])
#             loss = keras.metrics.categorical_crossentropy(y, pred)
#             loss += l2_loss

#         grads = tape.gradient(loss, q_model.trainable_variables)
#         quantized_grads_val = self.no_quantize(grads, self._hyperparams['bits'])
#         opt = self._get_optimizer(model)
#         opt.apply_gradients(zip(quantized_grads_val, model.trainable_variables))
#         self._weights = model.get_weights()

class QuantizationParamDevice(QuantizationDevice):
    """
    quantize the model
    """
    def __init__(self, *args):
        super().__init__(*args)

    def delegate(self, other, epoch, iteration):
        if other.enc_duration >= 6:
            for _ in range(iteration):
                self.fit(other, epoch, 64)
            self.q_requests[other.enc_duration] += 1
        elif other.enc_duration >= self._hyperparams['bits']:
            for _ in range(iteration):
                self.fit_w_quantized_params(other, epoch, pow(2, other.enc_duration))
            self.q_requests[other.enc_duration] += 1

    def fit_w_quantized_params(self, other, epoch, bits):
        model = self._get_model()
        quantized_weights = vec_to_grad(quant(weights_to_vec(self._weights), bits), self._weights)
        model.set_weights(quantized_weights)
        # train model
        X = other._x_train
        y = other._y_train
        CL = 0.01
        with tf.GradientTape() as tape:
            pred = model(X)
            for i in range(0,len(model.trainable_variables)):
                if i==0:
                    l2_loss = CL * tf.nn.l2_loss(model.trainable_variables[i])
                if i>=1:
                    l2_loss = l2_loss+ CL * tf.nn.l2_loss(model.trainable_variables[i])
            loss = keras.metrics.categorical_crossentropy(y, pred)
            loss += l2_loss

        grads = tape.gradient(loss, model.trainable_variables)
        opt = self._get_optimizer(model)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        self._weights = model.get_weights()

        # save optimizer state
        self.optimizer_weights = opt.get_weights()
        K.clear_session()

class QuantizationGradParamDevice(QuantizationDevice):
    """
    quantize both model & gradients
    """
    def __init__(self, *args):
        super().__init__(*args)

    def delegate(self, other, epoch, iteration):
        if other.enc_duration >= 6:
            for _ in range(iteration):
                self.fit(other, epoch, 64)
            self.q_requests[other.enc_duration] += 1
        elif other.enc_duration >= self._hyperparams['bits']:
            for _ in range(iteration):
                self.fit_w_quantized_params(other, epoch, pow(2, other.enc_duration))
            self.q_requests[other.enc_duration] += 1

    def fit_w_quantized_params(self, other, epoch, bits):
        model = self._get_model()
        quantized_weights = vec_to_grad(quant(weights_to_vec(self._weights), bits), self._weights)
        model.set_weights(quantized_weights)
        # train model
        X = other._x_train
        y = other._y_train
        CL = 0.001
        with tf.GradientTape() as tape:
            pred = model(X)
            for i in range(0,len(model.trainable_variables)):
                if i==0:
                    l2_loss = CL * tf.nn.l2_loss(model.trainable_variables[i])
                if i>=1:
                    l2_loss = l2_loss+ CL * tf.nn.l2_loss(model.trainable_variables[i])
            loss = keras.metrics.categorical_crossentropy(y, pred)
            loss += l2_loss

        grads = tape.gradient(loss, model.trainable_variables)
        quantized_grads_val = self.quantize(grads, bits)
        opt = self._get_optimizer(model)
        opt.apply_gradients(zip(quantized_grads_val, model.trainable_variables))
        self._weights = model.get_weights()

        # save optimizer state
        self.optimizer_weights = opt.get_weights()
        K.clear_session()

class NoQuantizationDevice(QuantizationDevice):
    def __init__(self, *args):
        super().__init__(*args)
    
    def delegate(self, other, epoch, iteration):
        if other.enc_duration >= 6:
            for _ in range(iteration):
                self.fit_w_quant_fn(other, epoch, self.no_quantize, 64)

    def quantize(self, grads, bits):
        return grads

class GeneralQuantizationDevice(QuantizationDevice):
    def __init__(self, *args):
        super().__init__(*args)
        self.is_q_params = self._hyperparams['is-q-params']
        self.is_q_grad = self._hyperparams['is-q-grad']
        self.is_q_network = self._hyperparams['is-q-network']

    def delegate(self, other, epoch, iteration):
        # @TODO quantize model but has to use the same scheme as qkeras
        pass

class MomentumDROppCLDevice(device.hetero_device.MomentumMixedDropoutDevice):
    def __init__(self, *args):
        super().__init__(*args)
        self.requests = []

        if 'enc-duration' not in self._hyperparams:
                self.enc_duration = self._get_number_from_dist(self._hyperparams['duration-lower-bound'],
                                    self._hyperparams['duration-upper-bound'],
                                    self._hyperparams['duration-distribution'],
                                    self._hyperparams['device-duration-interval'])
        else:
            self.enc_duration = self._hyperparams['enc-duration']

    def quantize(self, grads, bits):
        vec = grad_to_vec(grads)
        quantized = quant(vec, bits)
        return vec_to_grad(quantized, grads)

    def delegate(self, other, epoch, iteration):
        if other.enc_duration >= self._hyperparams['bits'] and other.device_power >= self._hyperparams['downto']:
            self.requests.append((other.device_power, other.enc_duration))
            for _ in range(iteration):
                self.fit_to_submodel_w_q(other, epoch, None, q=pow(2, other.enc_duration))

    def fit_to_submodel_w_q(self, other, epoch, size=None, q=64):
        # get submodel and select random (dropout) weights
        if size == None:
            size = other.device_power
        submodel = self._model_fn(size=size)
        sub_weights = self.weight_selector.select_weights(self._weights, submodel.get_weights())
        if q >= 64:
            quantized_weights = sub_weights
        else:
            quantized_weights = vec_to_grad(quant(weights_to_vec(sub_weights), q), sub_weights)
        submodel.set_weights(quantized_weights)
        print(f'q: {q}')

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
        K.clear_session()

class AutoMomentumDROppCLDevice(MomentumDROppCLDevice):
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

class OnlyDropoutDevice(MomentumDROppCLDevice):
    def __init__(self, *args):
        super().__init__(*args)

    def delegate(self, other, epoch, iteration):
        if other.enc_duration >= 6 and self._hyperparams['downto']:
            self.requests.append((other.device_power, other.enc_duration))
            for _ in range(iteration):
                self.fit_to_submodel_w_q(other, epoch, None, q=pow(2, other.enc_duration))

class OnlyQuantDevice(MomentumDROppCLDevice):
    def __init__(self, *args):
        super().__init__(*args)

    def delegate(self, other, epoch, iteration):
        if other.enc_duration >= self._hyperparams['bits'] and other.device_power >= self.model_size:
            self.requests.append((other.device_power, other.enc_duration))
            for _ in range(iteration):
                self.fit_to_submodel_w_q(other, epoch, None, q=pow(2, other.enc_duration))


class NoDropoutNorQDevice(MomentumDROppCLDevice):
    def __init__(self, *args):
        super().__init__(*args)
    
    def delegate(self, other, epoch, iteration):
        print(f'NoDropoutNorQDevice met {other.device_power}')
        if other.enc_duration >= 6 and other.device_power >= self.model_size:
            self.requests.append((other.device_power, other.enc_duration))
            for _ in range(iteration):
                self.fit_to_submodel_w_q(other, epoch, None, q=pow(2, 6))