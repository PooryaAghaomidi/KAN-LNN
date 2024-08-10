# -*- coding: utf-8 -*-
"""Adam loss function"""

from tensorflow.keras.optimizers import Adam


def adam_opt(lr):
    return Adam(learning_rate=lr,
                name="adam")
