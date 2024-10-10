# -*- coding: utf-8 -*-
"""Adam loss function"""

from tensorflow.keras.optimizers import Adamax


def adamax_opt(lr, clipvalue=0.5):
    return Adamax(learning_rate=lr,
                  clipvalue=clipvalue,
                  name="adamax")
