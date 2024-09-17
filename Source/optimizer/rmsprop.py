# -*- coding: utf-8 -*-
"""Adam loss function"""

from tensorflow.keras.optimizers import RMSprop


def rmsprop_opt(lr):
    return RMSprop(learning_rate=lr, name="rmsprop")
