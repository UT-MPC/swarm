import sys
sys.path.insert(0,'..')
import copy

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from sklearn.metrics import f1_score
from sklearn.utils import shuffle

import data_process as dp

class Device():
    """device that runs local training 
    """
    def __init__(self, 
                _id,
                model_fn, 
                opt_fn,
                init_weights,
                x_train, 
                y_train,
                train_data_provider,
                test_data_provider,
                target_labels,
                compile_config, 
                train_config,
                hyperparams):
        """
        params
            model: function to get keras model
            init_weights: initial weights of the model
            x_train, y_train: training set
            compile_config: dict of params for compiling the model
            train_config: dict of params for training
            eval_config: dict of params for evaluation
        """
        # @TODO make routine for making a dummy device
        self._id_num = _id
        self._model_fn = model_fn
        self._opt_fn = opt_fn
        if init_weights == None and model_fn != None:
            m = model_fn()
            init_weights = m.get_weights()
        self._weights = init_weights
        self._y_train_orig = y_train
        self.train_data_provider = train_data_provider
        self.test_data_provider = test_data_provider
        self._num_classes = test_data_provider.num_classes
        self._hyperparams = hyperparams
        if hyperparams != None:
            self._evaluation_metrics = hyperparams['evaluation-metrics']
            if 'similarity-threshold' in hyperparams:
                self._similarity_threshold = hyperparams['similarity-threshold']
            if 'low-similarity-threshold' in hyperparams:
                self._low_similarity_threshold = hyperparams['low-similarity-threshold']
            self.task_num = train_data_provider.task_num

            self.last_batch_num = {} # keeps the last batch num of the other client that this client was trained on
            self.total_num_batches = int(len(y_train) / hyperparams['batch-size'])
            if len(y_train) / hyperparams['batch-size'] - self.total_num_batches != 0:
                raise ValueError('batch-size has to divide local data size without remainders')

        self.optimizer_weights = None
        ratio_per_label = 1./(len(target_labels))
        self._desired_data_dist = {}
        for l in target_labels:
            self._desired_data_dist[l] = ratio_per_label

        self._compile_config = compile_config
        self._train_config = train_config

        self.set_local_data(x_train, y_train)

        # print("client {} initialize".format(_id))
        # print("--desired_data: {}".format(self._desired_data_dist.keys()))
        # print("--local_data: {}".format(np.unique(y_train)))

    def set_task(self, task):
        self.task = task

    def get_task(self):
        return self.task
    
    def set_local_data(self, x_train, y_train):
        bc = np.bincount(y_train)
        ii = np.nonzero(bc)[0]
        self._local_data_dist = dict(zip(ii, bc[ii]/len(y_train)))
        self._local_data_count = dict(zip(ii, bc[ii]))
        self._x_train = x_train
        self._y_train_orig = y_train
        self._y_train = keras.utils.to_categorical(y_train, self._num_classes)

    def resample_local_data(self):
        x, y = self.train_data_provider.peek(self._local_data_dist)
        self.set_local_data(x, y)
    
    def replace_local_data(self, ratio, new_x_train, new_y_train_orig):
        """
        decrease the existing local set except the given ratio of data
        and add new train data on top of it
        """
        # if self._id_num % 20 == 0:
        #     print("replace for ratio {} in client{}".format(ratio, self._id_num))
        new_y_train = keras.utils.to_categorical(new_y_train_orig, self._num_classes)

        if ratio > 1:
            self._x_train = new_x_train
            self._y_train = new_y_train

        # shuffle existing data
        data_size = len(self._x_train)
        p = np.random.permutation(data_size)
        replace_size = (int)(data_size * (1-ratio))
        self._x_train = self._x_train[p][:replace_size]
        self._y_train = self._y_train[p][:replace_size]

        self._x_train = np.concatenate((self._x_train, new_x_train), axis=0)
        self._y_train = np.concatenate((self._y_train, new_y_train), axis=0)

    def local_data_dist_similarity(self, client):
        overlap_ratio = 0.
        for i in range(self._num_classes):
            if i in self._local_data_dist.keys() and i in client._local_data_dist.keys():
                overlap_ratio += min(self._local_data_dist[i], client._local_data_dist[i])
        return overlap_ratio

    def desired_data_dist_similarity(self, client):
        overlap_ratio = 0.
        for i in range(self._num_classes):
            if i in self._desired_data_dist.keys() and i in client._desired_data_dist.keys():
                overlap_ratio += min(self._desired_data_dist[i], client._desired_data_dist[i])
        return overlap_ratio

    def others_local_data_to_my_desired_data_similarity(self, other):
        overlap_ratio = 0.
        for i in range(self._num_classes):
            if i in self._desired_data_dist.keys() and i in other._local_data_dist.keys():
                overlap_ratio += min(self._desired_data_dist[i], other._local_data_dist[i])
        return overlap_ratio

    def min_d2l_data_similarity(self, other):
        desired_to_local_sim_me_to_other = self.others_local_data_to_my_desired_data_similarity(other)
        desired_to_local_sim_other_to_me = other.others_local_data_to_my_desired_data_similarity(self)

        min_dl_sim = min(desired_to_local_sim_me_to_other, desired_to_local_sim_other_to_me)
        return min_dl_sim

    def train(self, epoch):
        if epoch == 0:
            return {}

        model = self._get_model()
        
        self._train_config['epochs'] = epoch
        self._train_config['x'] = self._x_train
        self._train_config['y'] = self._y_train
        self._train_config['verbose'] = 0

        hist = model.fit(**self._train_config)
        self._weights = copy.deepcopy(model.get_weights())
        K.clear_session()
        return hist

    def get_samples(self, data_size=1):
        label_conf = {}
        for l in self._local_data_dist:
            label_conf[l] = data_size
        sample_data_provider = dp.DataProvider(self._x_train, self._y_train_orig, 0)
        return sample_data_provider.peek(label_conf)

    def _train_with_others_data(self, other, epoch):
        if epoch == 0:
            return {}

        model = self._get_model()
        
        self._train_config['epochs'] = epoch

        # only pick the labels I want
        mask = np.zeros(other._y_train.shape[0], dtype=bool)
        for l in other._local_data_dist.keys():
            if l in self._local_data_dist.keys():
                one_hot = keras.utils.to_categorical(np.array([l]), 10)
                mask |= np.all(np.equal(other._y_train, one_hot), axis=1)

        if np.all(mask == False):
            return {}
        
        self._train_config['x'] = other._x_train
        self._train_config['y'] = other._y_train
        self._train_config['verbose'] = 0

        hist = model.fit(**self._train_config)
        self._weights = copy.deepcopy(model.get_weights())
        K.clear_session()
        return hist

    def request(self, client, epoch, iteration):
        raise NotImplementedError('')

    def receive(self, client, coeff):
        """receive the model from the other client. Does not affect the other's model"""
        if coeff > 1 or coeff < 0:
            raise ValueError("coefficient is not in range 0 <= c <= 1: c:{}".format(coeff))
        
        weights = [self._weights, client._weights]
        agg_weights = list()
        for weights_list_tuple in zip(*weights):
            agg_weights.append(np.array([np.average(np.array(w), axis=0, weights=[coeff, 1.-coeff]) for w in zip(*weights_list_tuple)]))
        self._weights = copy.deepcopy(agg_weights)

    def eval(self):
        if self._evaluation_metrics == 'loss-and-accuracy':
            return self.eval_loss_and_accuracy()
        elif self._evaluation_metrics == 'f1-score-weighted':
            return self.eval_f1_score()
        elif self._evaluation_metrics == 'split-f1-score-weighted':
            return self.eval_split_f1_score()
        else:
            raise ValueError('evaluation metrics is invalid: {}'.format(self._evaluation_metrics))

    def eval_loss_and_accuracy(self):
        model = self._get_model()
        xt, yt = self.test_data_provider.fetch(list(self._desired_data_dist.keys()), self._hyperparams['test-data-per-label'])                           
        hist = model.evaluate(xt, keras.utils.to_categorical(yt, self._num_classes), verbose=0)
        self._last_hist = hist
        K.clear_session()
        del model
        return hist

    def eval_f1_score(self, average='weighted'):
        model = self._get_model()
        xt, yt = self.test_data_provider.fetch(list(self._desired_data_dist.keys()), self._hyperparams['test-data-per-label'])
        y_pred = np.argmax(model.predict(xt), axis = 1)
        hist = f1_score(yt, y_pred, average=average)
        self._last_hist = hist
        K.clear_session()
        del model
        return hist

    def eval_split_f1_score(self, average='weighted'):
        model = self._get_model()
        hist = {}
        for labels in self._hyperparams['split-test-labels']:
            xt, yt = self.test_data_provider.fetch(labels, self._hyperparams['test-data-per-label'])
            y_pred = np.argmax(model.predict(xt), axis = 1)
            if str(labels) not in hist:
                hist[str(labels)] = []
            hist[str(labels)].append(f1_score(yt, y_pred, average=average))
        self._last_hist = hist
        K.clear_session()
        del model
        return hist

    def eval_weights(self, weights):
        model = self._get_model_from_weights(weights)
        xt, yt = self.test_data_provider.fetch(list(self._desired_data_dist.keys()), self._hyperparams['test-data-per-label'])                         
        hist = model.evaluate(xt, keras.utils.to_categorical(yt, self._num_classes), verbose=0)
        K.clear_session()
        del model
        return hist
    
    def _get_model(self):
        model = self._model_fn()
        model.set_weights(self._weights)
        self._compile_config['optimizer'] = self._opt_fn(lr=self._hyperparams['orig-lr'])
        model.compile(**self._compile_config)
        return model

    def _get_compressed_model(self):
        model = self._model_fn(compressed_ver=1)
        weights = copy.deepcopy(self._weights)
        COMPRESSED_LAYER_SIZE = 30
        a = np.arange(200)
        choice = np.random.choice(a, COMPRESSED_LAYER_SIZE)

        weights[-4] = weights[-4][:, choice]
        weights[-3] = weights[-3][choice]
        weights[-2] = weights[-2][choice, :]

        model.set_weights(weights)
        return model

    def _get_v2_compressed_model(self):
        model = self._model_fn(compressed_ver=2)
        weights = copy.deepcopy(self._weights)
        COMPRESSED_LAYER_SIZE = 50
        a = np.arange(200)
        choice = np.random.choice(a, COMPRESSED_LAYER_SIZE)

        weights[0] = weights[0][:, choice]
        weights[1] = weights[1][choice]
        weights[2] = weights[2][choice, :]

        COMPRESSED_LAYER_SIZE = 30
        a = np.arange(200)
        choice = np.random.choice(a, COMPRESSED_LAYER_SIZE)

        weights[-4] = weights[-4][:, choice]
        weights[-3] = weights[-3][choice]
        weights[-2] = weights[-2][choice, :]

        model.set_weights(weights)
        return model

    def _get_compressed_cnn_model(self):
        model = self._model_fn(compressed_ver=1)
        weights = copy.deepcopy(self._weights)
        COMPRESSED_LAYER_SIZE = 128
        a = np.arange(512)
        choice = np.random.choice(a, COMPRESSED_LAYER_SIZE)

        weights[-4] = weights[-4][:, choice]
        weights[-3] = weights[-3][choice]
        weights[-2] = weights[-2][choice, :]

        model.set_weights(weights)
        return model

    def _get_v2_compressed_cnn_model(self):
        model = self._model_fn(compressed_ver=2)
        weights = copy.deepcopy(self._weights)
        COMPRESSED_LAYER_SIZE = 64
        a = np.arange(512)
        choice = np.random.choice(a, COMPRESSED_LAYER_SIZE)

        weights[-4] = weights[-4][:, choice]
        weights[-3] = weights[-3][choice]
        weights[-2] = weights[-2][choice, :]

        model.set_weights(weights)
        return model

    def _get_model_from_weights(self, weights):
        model = self._model_fn()
        model.set_weights(weights)
        self._compile_config['optimizer'] = self._opt_fn(lr=self._hyperparams['orig-lr'])
        model.compile(**self._compile_config)
        return model
    
    def _get_model_w_lr(self, lr):
        model = self._model_fn()
        model.set_weights(self._weights)
        self._compile_config['optimizer'] = self._opt_fn(lr=lr)
        model.compile(**self._compile_config)
        return model

    def _get_dist_similarity(self, d1, d2):
        accum = 0.
        for k in d1.keys():
            if k in d2:
                accum += min(d1[k], d2[k])
        return accum

    def get_batch_num(self, other):
        if other._id_num in self.last_batch_num:
            self.last_batch_num[other._id_num] = (self.last_batch_num[other._id_num] + 1) % self.total_num_batches
        else:
            self.last_batch_num[other._id_num] = 0
        return self.last_batch_num[other._id_num]

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

    def fit_to(self, other, epoch):
        """
        fit the model to others data for "epoch" epochs
        one epoch only corresponds to a single batch
        """
        model = self._model_fn()
        model.set_weights(self._weights)
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

        # save optimizer state
        self.optimizer_weights = opt.get_weights()
        self._weights = model.get_weights()
        return model.get_weights()

        # model = self._get_model()
        
        # model.fit(**self.get_train_config(other, epoch, self.get_batch_num(other)))
        # weights = copy.deepcopy(model.get_weights())
        # K.clear_session()
        # del model
        # return weights

    def fit_to_labels_in_my_goal(self, other, epoch):

        model = self._get_model()
        self._train_config['epochs'] = 1
        x, y = dp.filter_data(other._x_train, other._y_train_orig, self._desired_data_dist.keys())
        self._train_config['x'] = x
        self._train_config['y'] = keras.utils.to_categorical(y, self._num_classes)
        if x.shape[0] < 1:
            raise ValueError("the filtered data size is 0!")
        self._train_config['verbose'] = 0
        model.fit(**self._train_config)
        weights = copy.deepcopy(model.get_weights())
        K.clear_session()
        del model
        return weights

    def fit_weights_to(self, weights, other, epoch):
        model = self._get_model_from_weights(weights)
        model.fit(**self.get_train_config(other, epoch))
        weights = copy.deepcopy(model.get_weights())
        K.clear_session()
        del model
        return weights

    def fit_w_lr_to(self, other, epoch, lr):
        model = self._get_model_w_lr(lr)
        model.fit(**self.get_train_config(other, epoch, self.get_batch_num(other)))
        weights = copy.deepcopy(model.get_weights())
        K.clear_session()
        del model
        return weights

    def get_train_config(self, client, steps, batch_num=None):
        tc = copy.deepcopy(self._train_config)
        tc['steps_per_epoch'] = steps
        tc['epochs'] = 1
        if batch_num == None:
            tc['x'] = client._x_train
            tc['y'] = client._y_train
        else:
            idx_start = batch_num * self._hyperparams['batch-size']
            if idx_start > client._x_train.shape[0]:
                raise ValueError('batch number is too large')
            idx_end = min(idx_start + self._hyperparams['batch-size'], client._x_train.shape[0])
            tc['x'] = client._x_train[idx_start:idx_end]
            tc['y'] = client._y_train[idx_start:idx_end]
        tc['verbose'] = 0
        return tc

    def decide_delegation(self, other):
        return True

    def is_federated(self):
        return False