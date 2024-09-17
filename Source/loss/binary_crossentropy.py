# -*- coding: utf-8 -*-
"""Categorical loss"""

from tensorflow.keras.losses import BinaryCrossentropy


def bc_loss(from_logits=False, label_smoothing=0.0):
    return BinaryCrossentropy(from_logits=from_logits,
                              label_smoothing=label_smoothing,
                              name="binary_crossentropy")
