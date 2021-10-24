# devices for task-aware opportunistic continual learning
import tensorflow.keras as keras
import numpy as np

import device.base_device

class Cifar100OracleDevice(device.base_device.Device):
    def __init__(self, *args):
        super().__init__(*args)

    def delegate(self, other, epoch, iteration):
        mammals = {'fox': 34, 'porcupine': 63, 'possum': 64, 'raccoon': 66, 'skunk': 75}
        flowers = {'orchid': 54, 'poppy': 63, 'rose': 70, 'sunflower': 82, 'tulip': 92}
        # print(other.get_task().split('-')[0])
        # print(other.get_task().split('-')[1])
        if other.get_task().split('-')[0] in mammals.keys() and other.get_task().split('-')[1] in flowers.keys():
            for _ in range(iteration):
                self._weights = self.fit_to(other, epoch)

class OnlyOtherGreedyDevice(device.base_device.Device):
    def __init__(self, *args):
        super().__init__(*args)

    def delegate(self, other, epoch, iteration):
        for _ in range(iteration):
            self._weights = self.fit_to(other, epoch)

class TaskAwareDevice(device.base_device.Device):
    def __init__(self, *args):
        super().__init__(*args)

    def fetch_model(self):
        return self._get_model()

    def get_logit_diff(self, other):
        np.set_printoptions(precision=3, suppress=True)
        model = self.fetch_model()
        extractor = keras.Model(inputs=model.inputs,
                        outputs=[layer.output for layer in model.layers])
        other_x_samples, other_y_samples = other.get_samples()
        this_x_samples, this_y_samples = self.get_samples()
        other_x_samples = other_x_samples[np.argsort(other_y_samples)]
        other_y_samples = np.sort(other_y_samples)
        this_x_samples = this_x_samples[np.argsort(this_y_samples)]
        this_y_samples = np.sort(this_y_samples)
        if not np.any(other_y_samples == this_y_samples):
            raise ValueError('something wrong with sampling!')
        other_features = extractor(other_x_samples)
        this_features = extractor(this_x_samples)
        sums = []
        for i in range(len(other_y_samples)):
            diff = np.linalg.norm(np.array(other_features[-1][i]) - np.array(this_features[-1][i]))
            sums.append(diff)
            # print("LABEL: {} ---------------------".format(other_y_samples[i]))
            # import matplotlib.pyplot as plt
            # plt.imshow(this_x_samples[this_x_idx], cmap='gray')
            # plt.show()
            # import matplotlib.pyplot as plt
            # plt.imshow(other_x_samples[i], cmap='gray')
            # plt.show()
            # print(np.array(this_features[-1][i]))
            # print(np.array(other_features[-1][i]))
            # print(diff)
            # print(np.linalg.norm(np.array(other_features[-3][i]) - np.array(this_features[-3][this_x_idx])))
        sums.sort()
        # print(sums)
        # MEDIAN_PERCENTAGE = 70
        # lr_filtered_len = int((len(sums) * (1 - MEDIAN_PERCENTAGE / 100))/2)
        # print(lr_filtered_len)
        if len(sums) > 5:
            sums = sums[-5:]

        # compute consistency
        dev = 0
        for l in np.unique(other_y_samples):
            mask = other_y_samples == l
            other_features = extractor(other_x_samples[mask])
            for out in other_features[-1]:
                # print(out)
                dev += np.array(out[0]-out[1])
        dev /= len(np.unique(other_y_samples))
        # print(dev)
        
        return sum(sums) / len(sums)

    def decide_delegation(self, other):
        # print('self-diff')
        self_diff = self.get_logit_diff(self)
        # self_diff = 1
        # print('other-diff')
        other_diff = self.get_logit_diff(other)
        # print('task: {}, self: {}, other:{}, ratio: {}'.format(other.get_task(), self_diff, other_diff, other_diff/self_diff))
        ratio = other_diff/self_diff
        return ratio < self._hyperparams['task-threshold']

    def delegate(self, other, epoch, iteration):
        if self.decide_delegation(other):
            for _ in range(iteration):
                self._weights = self.fit_to(other, epoch)

class CompressedTaskAwareDevice(TaskAwareDevice):
    def __init__(self, *args):
        super().__init__(*args)

    def fetch_model(self):
        return self._get_compressed_model()

    def decide_delegation(self, other):
        self_diff = self.get_logit_diff(self)
        other_diff = self.get_logit_diff(other)
        # print('clinum: {}, self: {}, other:{}, \nratio: {}'.format(other._id_num, self_diff, other_diff, other_diff/self_diff))

        ratio = other_diff/self_diff
        return ratio < 1.3

    def delegate(self, other, epoch, iteration):
        if self.decide_delegation(other):
            for _ in range(iteration):
                self._weights = self.fit_to(other, epoch)

class CompressedCNNTaskAwareDevice(TaskAwareDevice):
    def __init__(self, *args):
        super().__init__(*args)

    def fetch_model(self):
        return self._get_compressed_cnn_model()

    def decide_delegation(self, other):
        self_diff = self.get_logit_diff(self)
        other_diff = self.get_logit_diff(other)
        # print('clinum: {}, self: {}, other:{}, \nratio: {}'.format(other._id_num, self_diff, other_diff, other_diff/self_diff))

        ratio = other_diff/self_diff
        return ratio < 1.3

    def delegate(self, other, epoch, iteration):
        if self.decide_delegation(other):
            for _ in range(iteration):
                self._weights = self.fit_to(other, epoch)

class V2CompressedTaskAwareDevice(TaskAwareDevice):
    def __init__(self, *args):
        super().__init__(*args)

    def fetch_model(self):
        return self._get_v2_compressed_model()

    def decide_delegation(self, other):
        self_diff = self.get_logit_diff(self)
        other_diff = self.get_logit_diff(other)
        # print('clinum: {}, self: {}, other:{}, \nratio: {}'.format(other._id_num, self_diff, other_diff, other_diff/self_diff))

        ratio = other_diff/self_diff
        return ratio < 1.2

    def delegate(self, other, epoch, iteration):
        if self.decide_delegation(other):
            for _ in range(iteration):
                self._weights = self.fit_w_lr_to(other, epoch, self._hyperparams['orig-lr'])


class V2CompressedCNNTaskAwareDevice(TaskAwareDevice):
    def __init__(self, *args):
        super().__init__(*args)

    def fetch_model(self):
        return self._get_v2_compressed_cnn_model()

    def decide_delegation(self, other):
        self_diff = self.get_logit_diff(self)
        other_diff = self.get_logit_diff(other)
        # print('clinum: {}, self: {}, other:{}, \nratio: {}'.format(other._id_num, self_diff, other_diff, other_diff/self_diff))

        ratio = other_diff/self_diff
        return ratio < 1.2

    def delegate(self, other, epoch, iteration):
        if self.decide_delegation(other):
            for _ in range(iteration):
                self._weights = self.fit_w_lr_to(other, epoch, self._hyperparams['orig-lr'])