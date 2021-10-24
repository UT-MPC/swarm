import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

import device.hetero_device

class QuantizationDevice(device.hetero_device.HeteroDevice):
    def __init__(self, *args):
        super().__init__(*args)