# -*- coding: utf-8 -*-
"""Unet model"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model, Model


def SE(inputs, ratio=8):
    channel_axis = -1
    num_filters = inputs.shape[channel_axis]
    se_shape = (1, 1, num_filters)

    x = layers.GlobalAveragePooling2D()(inputs)
    x = layers.Reshape(se_shape)(x)
    x = layers.Dense(num_filters // ratio, activation='relu', use_bias=False)(x)
    x = layers.Dense(num_filters, activation='sigmoid', use_bias=False)(x)

    x = layers.Multiply()([inputs, x])
    return x


def stem_block(inputs, num_filters, strides=1):
    ## Conv 1
    x = layers.Conv2D(num_filters, 3, padding="same", strides=strides)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(num_filters, 3, padding="same")(x)

    ## Shortcut
    s = layers.Conv2D(num_filters, 1, padding="same", strides=strides)(inputs)
    s = layers.BatchNormalization()(s)

    ## Add
    x = layers.Add()([x, s])
    x = SE(x)
    return x


def resnet_block(inputs, num_filter, strides=1):
    ## Conv 1
    x = layers.BatchNormalization()(inputs)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(num_filter, 3, padding="same", strides=strides)(x)

    ## Conv 2
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(num_filter, 3, padding="same", strides=1)(x)

    ## Shortcut
    s = layers.Conv2D(num_filter, 1, padding="same", strides=strides)(inputs)
    s = layers.BatchNormalization()(s)

    ## Add
    x = layers.Add()([x, s])
    x = SE(x)
    return x


def aspp_block(inputs, num_filters):
    x1 = layers.Conv2D(num_filters, 3, dilation_rate=6, padding="same")(inputs)
    x1 = layers.BatchNormalization()(x1)

    x2 = layers.Conv2D(num_filters, 3, dilation_rate=12, padding="same")(inputs)
    x2 = layers.BatchNormalization()(x2)

    x3 = layers.Conv2D(num_filters, 3, dilation_rate=18, padding="same")(inputs)
    x3 = layers.BatchNormalization()(x3)

    x4 = layers.Conv2D(num_filters, (3, 3), padding="same")(inputs)
    x4 = layers.BatchNormalization()(x4)

    y = layers.Add()([x1, x2, x3, x4])
    y = layers.Conv2D(num_filters, 1, padding="same")(y)
    return y


def attetion_block(x1, x2):
    num_filters = x2.shape[-1]

    x1_conv = layers.BatchNormalization()(x1)
    x1_conv = layers.Activation("relu")(x1_conv)
    x1_conv = layers.Conv2D(num_filters, 3, padding="same")(x1_conv)
    x1_pool = layers.MaxPooling2D((2, 2))(x1_conv)

    x2_conv = layers.BatchNormalization()(x2)
    x2_conv = layers.Activation("relu")(x2_conv)
    x2_conv = layers.Conv2D(num_filters, 3, padding="same")(x2_conv)

    x = layers.Add()([x1_pool, x2_conv])

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(num_filters, 3, padding="same")(x)

    x = layers.Multiply()([x, x2])
    return x


def encoder(image_shape):
    inputs = layers.Input(shape=image_shape)

    res = layers.Resizing(image_shape[0] // 2, image_shape[1] // 2)(inputs)

    c1 = stem_block(res, 32, strides=1)
    c2 = resnet_block(c1, 64, strides=2)
    c3 = resnet_block(c2, 128, strides=2)
    c4 = resnet_block(c3, 256, strides=2)

    """ Bridge """
    b1 = aspp_block(c4, 256)

    outputs = layers.GlobalAveragePooling2D()(b1)

    return Model(inputs=inputs, outputs=outputs)


def build_n1(image_shape):
    input_img = layers.Input(shape=image_shape)

    backbone = encoder(image_shape[1:])
    time_distributed_encoder = layers.TimeDistributed(backbone)(input_img)

    temporal_output = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(1e-4)))(time_distributed_encoder)
    # temporal_output = layers.Flatten()(temporal_output)

    # skip = layers.Flatten()(time_distributed_encoder)
    # skip = layers.Dense(64, activation='relu')(skip)
    # concat = layers.Concatenate()([temporal_output, skip])

    x = layers.Dropout(0.2)(temporal_output)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)

    output = layers.Dense(2, activation='softmax')(x)

    model = Model(inputs=input_img, outputs=output)

    print(model.summary())

    return model
