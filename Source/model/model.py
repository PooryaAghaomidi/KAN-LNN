# -*- coding: utf-8 -*-
"""Unet model"""

from .ncps.tf import LTC, LTCCell
from .ncps.wirings import AutoNCP
from .tfkan.layers.dense import DenseKAN
from .tfkan.layers.convolution import Conv2DKAN
from tensorflow.keras import layers
from tensorflow.keras.models import Model


def build(input_shape, num_classes):
    inpt = layers.Input(shape=input_shape)

    x = Conv2DKAN(8, kernel_size=(3, 3), padding="same")(inpt)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2DKAN(16, kernel_size=(3, 3), padding="same")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2DKAN(32, kernel_size=(3, 3), padding="same")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2DKAN(64, kernel_size=(3, 3), padding="same")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2DKAN(128, kernel_size=(3, 3), padding="same")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    r = layers.Permute((3, 1, 2))(x)
    r = layers.Reshape((128, 16))(r)

    wiring = AutoNCP(128, 64)
    rnn_cell = LTCCell(wiring)

    lnn = layers.RNN(rnn_cell, return_sequences=True)(r)
    gavg = layers.GlobalAveragePooling1D()(lnn)

    fc = DenseKAN(32)(gavg)
    utpt = DenseKAN(num_classes)(fc)

    model = Model(inputs=inpt, outputs=utpt)

    print(model.summary())

    return model
