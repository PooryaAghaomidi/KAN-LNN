# -*- coding: utf-8 -*-
"""Unet model"""

from tensorflow.keras.models import load_model, Model
from .ncps.tf import LTC, LTCCell
from .ncps.wirings import AutoNCP
from .tfkan.layers.dense import DenseKAN
from .tfkan.layers.convolution import Conv2DKAN
from tensorflow.keras import layers
from tensorflow.keras.models import Model


def build(image_shape, signal_shape, num_classes, lnn_units, fin_path, kan_units):
    """############################### Image ####################################"""

    input_img = layers.Input(shape=image_shape)

    res = layers.TimeDistributed(layers.Resizing(int(image_shape[1]/4), int(image_shape[2]/4)))(input_img)
    x = layers.TimeDistributed(layers.Conv2D(8, kernel_size=(3, 3), padding="same"))(res)
    x = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2)))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.Conv2D(16, kernel_size=(3, 3), padding="same"))(x)
    x = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2)))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.Conv2D(32, kernel_size=(3, 3), padding="same"))(x)
    x = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2)))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.Conv2D(64, kernel_size=(3, 3), padding="same"))(x)
    x = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2)))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.Conv2D(128, kernel_size=(3, 3), padding="same"))(x)
    x = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2)))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)

    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
    x = layers.Flatten()(x)
    img_output = layers.Dense(lnn_units[1], activation="relu")(x)

    """################################# LNN #####################################"""

    input_sig = layers.Input(shape=signal_shape)

    wiring = AutoNCP(lnn_units[0], lnn_units[1])
    rnn_cell = LTCCell(wiring)

    lnn = layers.RNN(rnn_cell, return_sequences=True)(input_sig)
    lnn_output = layers.GlobalAveragePooling1D()(lnn)

    """################################# FIN #####################################"""

    fin_model_raw = load_model(fin_path)
    fin_inp = fin_model_raw.input
    fin_out = fin_model_raw.layers[-2].output
    fin_model = Model(fin_inp, fin_out)

    x = fin_model(input_sig)
    x = layers.GlobalAveragePooling1D()(x)
    fin_output = layers.Dense(lnn_units[1], activation="relu")(x)

    """############################### Concat ####################################"""

    contacted = layers.Concatenate()([img_output, lnn_output, fin_output])

    fc = DenseKAN(kan_units[0])(contacted)
    fc = DenseKAN(num_classes)(fc)
    output = layers.Softmax()(fc)

    model = Model(inputs=[input_img, input_sig], outputs=output)

    print(model.summary())

    return model
