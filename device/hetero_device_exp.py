# devices for heterogeneous environment in terms of model sizes
import numpy as np
import time
import copy
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

import device.base_device
import device.exp_device

from model_weight_utils import SelectWeightsConv, SelectWeightsAdv, SelectWeightsNoWeighting, MomentumSelectWeights
from model_weight_utils import gradients, multiply_weights, enlarge_weights, add_weights

class HeteroDevice(device.exp_device.GreedyWOSimDevice):
    def __init__(self, *args):
        super().__init__(*args)
        self.device_power = self._get_device_power()
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

    def _get_device_power(self):
        lower = self._hyperparams['hetero-lower-bound']
        upper = self._hyperparams['hetero-upper-bound']
        vals = np.arange(lower, upper+1, 1)
        
        dist = self._hyperparams['hetero-distribution']
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
            return self._hyperparams['device-power-interval'][self._hyperparams['repeated-number']]
        else:
            probs = np.ones(upper-lower+1)

        p = probs/np.sum(probs)
        # t = time.time()
        # np.random.seed(int(t))
        return np.random.choice(vals, 1, p=p)[0]

class DropinDevice(HeteroDevice):
    def __init__(self, *args):
        super().__init__(*args)
        self._num_requests = 0
        self._num_delegation = 0

    def delegate(self, other, epoch, iteration):
        self._num_requests += 1
        if self._num_requests == 200 * 3 or self._num_requests == 450 * 3:
            self._enlarge_model()
        """
        only fits to other's data
        """
        if other.device_power >= self.model_size:
            self._num_requests += 1
            for _ in range(iteration):
                self._weights = self.fit_to(other, epoch)
            
    def _enlarge_model(self):
        new_model_size = self.model_size + 1
        new_model = self._model_fn(size=new_model_size)
        new_weights = new_model.get_weights()
        weights = self._weights
        enlarge_weights(new_weights, weights)
        self._weights = new_weights
        self.model_size = new_model_size

class DropInNOutDevice(DropinDevice):
    def __init__(self, *args):
        super().__init__(*args)

    def delegate(self, other, epoch, iteration):
        """
        only fits to other's data
        """
        self._num_requests += 1
        if self._num_requests == 200 * 3 or self._num_requests == 450 * 3:
            self._enlarge_model()
        if other.device_power >= self.model_size:
            for _ in range(iteration):
                self._weights = self.fit_to(other, epoch)
        else:
            for _ in range(iteration):
                submodel = self._model_fn(size=other.device_power)
                sw = SelectWeights(self._weights, submodel.get_weights())
                sub_weights = sw.get_selected()
                submodel.set_weights(sub_weights)
                sub_weights = self.fit_model_to(submodel, other, epoch)
                self._weights = sw.update_target(sub_weights)

class DropoutDevice(HeteroDevice):
    def __init__(self, *args):
        super().__init__(*args)
        self._num_requests = 0
        # self._hyperparams['orig-lr'] *= 10 @TODO this errors out. bc the weights become too big?

    def delegate(self, other, epoch, iteration):
        """
        only fits to other's data
        """
        print('device_power: {}'.format(other.device_power))
        self._num_requests += 1
        if other.device_power >= self.model_size:
            pass
            for _ in range(iteration):
                self._weights = self.fit_to(other, epoch)
        else:
            # if self._num_requests > 3000 * 3:
                # print('device power: {}'.format(other.device_power))
            for _ in range(iteration):
                submodel = self._model_fn(size=other.device_power)
                sw = SelectWeights(self._weights, submodel.get_weights())
                sub_weights = sw.get_selected()
                submodel.set_weights(sub_weights)
                new_sub_weights = self.fit_model_to(submodel, other, epoch, self._hyperparams['orig-lr'] * 10)
                grads = gradients(sub_weights, new_sub_weights)
                grads = multiply_weights(grads, other.device_power/self.model_size)
                new_sub_weights = add_weights(sub_weights, grads)
                self._weights = sw.update_target(new_sub_weights)

class DropOutVarDevice(DropoutDevice):
    def __init__(self, *args):
        super().__init__(*args)
        self.requests = {}
        for i in range(self.model_size+1):
            self.requests[i] = 0

    def delegate(self, other, epoch, iteration):
        """
        only fits to other's data
        """
        # print('model_size: {}, others_device_power: {}'.format(self.model_size, other.device_power))
        self._num_requests += 1
        if other.device_power >= self.model_size:
            self.requests[other.device_power] += 1
            for _ in range(iteration):
                self._weights = self.fit_to(other, epoch)
        elif other.device_power >= self._hyperparams['downto']:
            self.requests[other.device_power] += 1
            # if self._num_requests > 3000 * 3:
                # print('device power: {}'.format(other.device_power))
            for _ in range(iteration):
                submodel = self._model_fn(size=other.device_power)
                if self._hyperparams['dataset'] == 'mnist':
                    sw = SelectWeights(self._weights, submodel.get_weights())
                elif self._hyperparams['dataset'] == 'cifar':
                    sw = SelectWeightsConv(self._weights, submodel.get_weights())
                sub_weights = sw.get_selected()
                submodel.set_weights(sub_weights)
                new_sub_weights = self.fit_model_to(submodel, other, epoch, self._hyperparams['orig-lr'] * 25)
                grads = gradients(sub_weights, new_sub_weights)
                grads = multiply_weights(grads, other.device_power/self.model_size)
                new_sub_weights = add_weights(sub_weights, grads)
                self._weights = sw.update_target(new_sub_weights)

class MixedDropoutDevice(DropoutDevice):
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
        self.comm_cost = 0
        self.comp_cost = 0
        self.flops = {3: 24273,
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
        if hist[1] > 0.85 and not self.criteria:
            self.requests['reached 0.85 at'] = str(self.requests)
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
            # loss = keras.metrics.categorical_crossentropy(y, pred)
            loss = keras.metrics.mean_squared_error(y, pred)

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
            # loss = keras.metrics.categorical_crossentropy(y, pred)
            loss = keras.metrics.mean_squared_error(y, pred)

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

        # we act like zeros got applied
        # model = self._get_model()
        # opt = self._get_optimizer(model)
        # opt.apply_gradients(zip(big_grads_w_zero, model.trainable_variables))

        self.optimizer_weights = opt.get_weights()
        K.clear_session()


class MixedMultiOptDropoutDevice(MixedDropoutDevice):
    """
    use different optimizers for different dropouts
    """
    def __init__(self, *args):
        super().__init__(*args)

        self.optimizer_weights_dict = {}
        self.lr_avg = 0
        self.lr_num = 0

    def _get_suboptimizer(self, model, size):
        opt_params = copy.deepcopy(self._hyperparams['optimizer-params'])
        # opt_params['learning_rate'] *= (self.model_size/size) ** 2
        self.lr_avg = (self.lr_avg * self.lr_num + (size / self.model_size) * opt_params['learning_rate']) / (self.lr_num + 1)
        self.lr_num += 1

        opt = self._opt_fn(**opt_params)
        zero_grads = [tf.zeros_like(w) for w in model.trainable_variables]
        opt.apply_gradients(zip(zero_grads, model.trainable_variables))

        if size in self.optimizer_weights_dict:
            opt.set_weights(self.optimizer_weights_dict[size])
        else:
            self.optimizer_weights_dict[size] = opt.get_weights()

        del model
        return opt

    def delegate(self, other, epoch, iteration):
        """
        only fits to other's data
        """
        # print('model_size: {}, others_device_power: {}'.format(self.model_size, other.device_power))
        self._num_requests += 1
        if other.device_power >= self.model_size:
            for _ in range(iteration):
                self._weights = self.fit_to(other, epoch)
                self.add_comm_cost(self.model_size)
                self.lr_avg = (self.lr_avg * self.lr_num + self._hyperparams['optimizer-params']['learning_rate']) / (self.lr_num + 1)
                self.lr_num += 1
            self.requests[other.device_power] += 1
        elif other.device_power >= self._hyperparams['downto']:
            # if self._num_requests > 3000 * 3:
                # print('device power: {}'.format(other.device_power))
            for _ in range(iteration):
                self.fit_to_submodel(other, epoch)
                self.add_comm_cost(other.device_power)
            self.requests[other.device_power] += 1

        print('lr_avg: {}'.format(self.lr_avg))

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
            # loss = keras.metrics.categorical_crossentropy(y, pred)
            loss = keras.metrics.mean_squared_error(y, pred)

        grads = tape.gradient(loss, submodel.trainable_variables)
        grads_val = [g.numpy() for g in grads]

        # scale gradients
        grads_val = multiply_weights(grads_val, scale)

        # convert gradients to the original size
        big_grads = self.weight_selector.get_target_from_selected(grads_val)
        model = self._get_model()
        opt = self._get_suboptimizer(model, size)
        opt.apply_gradients(zip(big_grads, model.trainable_variables))
        self._weights = model.get_weights()

        # save optimizer state
        self.optimizer_weights_dict[size] = opt.get_weights()
        K.clear_session()


class MixedScaledDropoutDevice(MixedDropoutDevice):
    def __init__(self, *args):
        super().__init__(*args)

    def delegate(self, other, epoch, iteration):
        """
        only fits to other's data
        """
        # print('model_size: {}, others_device_power: {}'.format(self.model_size, other.device_power))
        self._num_requests += 1
        if other.device_power >= self.model_size:
            for _ in range(iteration):
                self._weights = self.fit_to(other, epoch)
                self.add_comm_cost(self.model_size)
            self.requests[other.device_power] += 1
        elif other.device_power >= self._hyperparams['downto']:
            # if self._num_requests > 3000 * 3:
                # print('device power: {}'.format(other.device_power))
            for _ in range(iteration):
                self.fit_to_submodel(other, epoch, scale=(self.model_size/other.device_power) ** 2 * 10)
                self.add_comm_cost(other.device_power)
            self.requests[other.device_power] += 1

class DropoutOnlyOnDevice(MixedDropoutDevice):
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

class DynamicMixedDropoutDevice(MixedDropoutDevice):
    def __init__(self, *args):
        super().__init__(*args)
        self.last_hists = []
        if 'downto' in self._hyperparams:
            self.downto = self._hyperparams['downto']
        self.init_slope = None

    def delegate(self, other, epoch, iteration):
        """
        only fits to other's data
        """
        # determine lower limit
        if len(self.last_hists) >= 10:
            h = np.array(self.last_hists)
            x = np.arange(len(h))
            x = x.reshape(-1,1)
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression().fit(x, h)
            slope = max(-reg.coef_[0], 0)
            if self.init_slope != None:
                if (self.init_slope) != 0:
                    # if self.init_slope > slope:
                    #     self.init_slope = slope
                    #     slope = -1
                    slope /= self.init_slope
                else:
                    slope = 1
                print('slope: {}'.format(slope))
                s = np.exp(-8 * slope) + 0.0000001
                self.downto = (self._hyperparams['downto']) + (self.model_size - self._hyperparams['downto']) * s
                self.downto = int(self.downto)

                print('history: {}'.format(self.last_hists))
                print('slope: {}, minimum: {}'.format(slope, self.downto))
            else:
                self.init_slope = slope
        # print('model_size: {}, others_device_power: {}'.format(self.model_size, other.device_power))
        if other.device_power >= self.downto:
            # if self._num_requests > 3000 * 3:
                # print('device power: {}'.format(other.device_power))
            for _ in range(iteration):
                self.fit_to_submodel(other, epoch)
                self.add_comm_cost(other.device_power)
            self.requests[other.device_power] += 1

    def eval(self):
        hist = super().eval()
        if len(self.last_hists) >= 10:
            self.last_hists.pop(0)
        self.last_hists.append(hist[0])
        return hist

class DynamicMixedDropoutDeviceOrig(MixedDropoutDevice):
    def __init__(self, *args):
        super().__init__(*args)
        self.increase_downto = 0
        self.prev_acc = 0.7

    def delegate(self, other, epoch, iteration):
        """
        only fits to other's data
        """
        # print('model_size: {}, others_device_power: {}'.format(self.model_size, other.device_power))
        if other.device_power >= self._hyperparams['downto'] + self.increase_downto:
            # if self._num_requests > 3000 * 3:
                # print('device power: {}'.format(other.device_power))
            for _ in range(iteration):
                self.fit_to_submodel(other, epoch)
                self.add_comm_cost(other.device_power)
            self.requests[other.device_power] += 1

    def eval(self):
        hist = super().eval()
        if (hist[1] - self.prev_acc > 0.1):
            if self.prev_acc != 0:
                self.increase_downto += 1
                print('acc: {}, inc downto: {}'.format(hist[1], self.increase_downto))
            self.prev_acc = hist[1]
        return hist

class DynamicMixedDropoutDeviceOrig(MixedDropoutDevice):
    def __init__(self, *args):
        super().__init__(*args)
        self.noMixed = False
        self.prev_acc = 0

    def delegate(self, other, epoch, iteration):
        """
        only fits to other's data
        """
        # print('model_size: {}, others_device_power: {}'.format(self.model_size, other.device_power))
        if self.noMixed:
            if other.device_power >= 10:
                for _ in range(iteration):
                    self._weights = self.fit_to(other, epoch)
                    self.add_comm_cost(other.device_power)
                self.requests[other.device_power] += 1
            else:
                model = self._get_model()
                opt = self._get_optimizer(model)
                if opt.__class__.__name__ != 'Adam':
                    raise ValueError('This approach only works on Adam optimzer!')
                optimizer_weights = opt.get_weights()
                model_weights = model.get_weights()
                opt.apply_gradients(zip(optimizer_weights[1:1+len(model_weights)], model.trainable_variables))
                
                self._weights = model.get_weights()
                # self.optimizer_weights = opt.get_weights()
        
        elif other.device_power >= self._hyperparams['downto']:
            # if self._num_requests > 3000 * 3:
                # print('device power: {}'.format(other.device_power))
            for _ in range(iteration):
                self.fit_to_submodel(other, epoch)
                self.add_comm_cost(other.device_power)
            self.requests[other.device_power] += 1

    def eval(self):
        hist = super().eval()
        if not self.noMixed and hist[1] > 0.4:
            self.noMixed = True
        return hist

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