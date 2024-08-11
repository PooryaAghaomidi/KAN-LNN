# -*- coding: utf-8 -*-
"""MSE loss"""

from tensorflow.keras.losses import MeanAbsoluteError


def mae_loss():
    return MeanAbsoluteError(name="mean_absolute_error")
