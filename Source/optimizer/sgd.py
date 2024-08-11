# -*- coding: utf-8 -*-
"""Adam loss function"""

from tensorflow.keras.optimizers import SGD


def sgd_opt(lr):
    return SGD(learning_rate=lr,
               name="sgd")
