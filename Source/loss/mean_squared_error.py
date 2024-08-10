# -*- coding: utf-8 -*-
"""MSE loss"""

from tensorflow.keras.losses import MeanSquaredError


def mse_loss():
    return MeanSquaredError(reduction="sum_over_batch_size", name="mean_squared_error")
