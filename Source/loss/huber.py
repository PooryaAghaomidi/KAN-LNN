# -*- coding: utf-8 -*-
"""MSE loss"""

from tensorflow.keras.losses import Huber


def huber_loss():
    return Huber(delta=1.0, reduction="sum_over_batch_size", name="huber_loss")
