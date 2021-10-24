# devices that adopt knowledge distillation
import pickle
import tensorflow.keras as keras
import utils.distiller as distiller

import device.opportunistic_device
import device.exp_device

class TrainOnDataClient(device.opportunistic_device.JSDOppDevice):
    def __init__(self, *args):
        super().__init__(*args)
    
    def delegate(self, other, epoch, iteration):
        """
        only fits to other's data
        """
        for _ in range(iteration):
            self._weights = self.fit_to(other, epoch)

class TrainOnModelClient(device.opportunistic_device.JSDOppDevice):
    def __init__(self, *args):
        super().__init__(*args)
        # init teacher model
        self.teacher = self._get_model()
        with open('../pretrained/2nn_mnist_teacher.pickle', 'rb') as handle:
            teacher_weights = pickle.load(handle)
        self.teacher.set_weights(teacher_weights)
        # set public dataset
        self.x_public = self._hyperparams['x_public']
        self.y_public = self._hyperparams['y_public']
    
    def delegate(self, other, epoch, iteration):
        student = self._get_model()
        dstlr = distiller.Distiller(student=student, teacher=self.teacher)
        dstlr.compile(
            optimizer=keras.optimizers.Adam(),
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
            student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            distillation_loss_fn=keras.losses.KLDivergence(),
            alpha=0.1,
            temperature=10,
        )
        for _ in range(iteration):
            dstlr.fit(self.x_public, self.y_public, epochs=1)
            self._weights = student.get_weights()

class TrainOnDataAndModelClient(device.opportunistic_device.JSDOppDevice):
    def __init__(self, *args):
        super().__init__(*args)
        self.request_number = 0
    
    def delegate(self, other, epoch, iteration):
        if self.request_number % 2 == 0:
            super().delegate(other, epoch, iteration)
        else:
            for _ in range(iteration):
                self._weights = self.fit_to(other, epoch)
        self.request_number += 1