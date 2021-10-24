from device.exp_device import GreedyValidationClient
import numpy as np
import time
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

import device.hetero_device

class HeteroDataDevice(device.hetero_device.HeteroDevice):
    def __init__(self, *args):
        super().__init__(*args)
        data_size = self._get_data_size()
        x_train, y_train = self.train_data_provider.get_random(data_size)
        self.set_local_data(x_train, y_train)

        self.batch_size = self._hyperparams['batch-size']
        self.optimizer_params = self._hyperparams['optimizer-params']
        self.optimizer_weights = None

        grad_dict = {} # (data_size, grad)

    def _get_data_size(self):
        dist = self._hyperparams['data-size-distribution']
        if dist == 'manual':
            return self._hyperparams['data-size-interval'][self._hyperparams['repeated-number']]

    def delegate(self, other, epoch, iteration):
        """
        only fits to other's data
        """
        if other.device_power >= self.model_size:
            for _ in range(iteration):
                grad = self.req_grad(other, epoch)

    def req_grad(self, other, epoch):
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
        return grads

    def update_model(self, grad):
        model = self._get_model()
        opt = self._get_optimizer(model)
        opt.apply_gradients(zip(grad, model.trainable_variables))
        self._weights = model.get_weights()

        # save optimizer state
        self.optimizer_weights = opt.get_weights()


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