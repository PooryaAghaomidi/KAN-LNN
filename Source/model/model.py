# -*- coding: utf-8 -*-
"""Unet model"""

import tensorflow as tf
from .ncps.tf import LTCCell
from .ncps.wirings import AutoNCP
from tensorflow.keras import layers
from .tfkan.layers.dense import DenseKAN
from tensorflow.keras.models import load_model, Model


def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, stride, padding='same')(x)
    x = layers.BatchNormalization()(x)  # Can be replaced with other normalization layers if desired
    x = layers.Activation('swish')(x)  # SiLU activation

    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, shortcut])
    x = layers.Activation('swish')(x)
    return x


def encoder(image_shape):
    inputs = layers.Input(shape=image_shape)

    res = layers.Resizing(image_shape[0]//2, image_shape[1]//2)(inputs)

    x = layers.Conv2D(16, kernel_size=3, strides=2, padding='same')(res)
    x = residual_block(x, 16)

    x = layers.Conv2D(32, kernel_size=3, strides=2, padding='same')(x)
    x = residual_block(x, 32)

    x = layers.Conv2D(64, kernel_size=3, strides=2, padding='same')(x)
    x = residual_block(x, 64)

    x = layers.LayerNormalization(axis=-1)(x)
    x = layers.Activation('swish')(x)

    outputs = layers.GlobalAveragePooling2D()(x)

    backbone = Model(inputs=inputs, outputs=outputs)
    backbone.summary()

    return backbone


def build(image_shape, signal_shape, num_classes, fin_path, base_model, kan_units, common_len):
    """############################### Image ####################################"""

    input_img = layers.Input(shape=image_shape)

    base_model_raw = load_model(base_model)
    base_inp = base_model_raw.input
    base_out = base_model_raw.layers[-5].output
    final_base_model = Model(base_inp, base_out)

    x = final_base_model(input_img)

    img_output = layers.Dense(common_len, activation="relu")(x)

    """################################# FIN #####################################"""

    input_sig = layers.Input(shape=signal_shape)

    fin_model_raw = load_model(fin_path)
    fin_inp = fin_model_raw.input
    fin_out = fin_model_raw.layers[-2].output
    fin_model = Model(fin_inp, fin_out)

    x = fin_model(input_sig)
    x = layers.BatchNormalization()(x)
    fin_output = layers.Dense(common_len, activation="relu")(x)

    """############################### Concat ####################################"""

    contacted = layers.Concatenate()([fin_output, img_output])
    x = layers.Dropout(0.0)(contacted)

    for kan_unit in kan_units:
        x = DenseKAN(kan_unit)(x)

    fc = DenseKAN(num_classes)(x)
    output = layers.Softmax()(fc)

    model = Model(inputs=[input_img, input_sig], outputs=output)

    print(model.summary())

    return model
