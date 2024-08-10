# -*- coding: utf-8 -*-
"""Unet model"""

from tensorflow.keras.models import load_model, Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model


def sample_z(args):
    mu, l_sigma = args
    eps = K.random_normal(shape=(K.shape(mu)[0], K.int_shape(mu)[1]), mean=0., stddev=1.)
    return mu + K.exp(l_sigma / 2) * eps


def residual_block(x, filters, kernel_size=3, activation='relu', dropout_rate=0.5):
    """Residual block with Conv1D layers."""
    skip = x

    x = layers.Conv1D(filters, kernel_size=kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)

    x = layers.Conv1D(filters, kernel_size=kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)

    x = layers.Add()([x, skip])
    x = layers.Activation(activation)(x)

    return x


def build_cvae(label_shape, signal_shape, num_classes, z_space, dense_units):
    """############################### ENCODER ####################################"""

    input_sig = layers.Input(shape=signal_shape)
    input_lbl = layers.Input(shape=label_shape)
    inputs = layers.concatenate([input_sig, input_lbl])

    # Initial Convolution and Pooling
    x = layers.Conv1D(32, kernel_size=3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Residual Blocks
    x = residual_block(x, 32)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = residual_block(x, 64)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = residual_block(x, 128)

    # Dense Layers
    x = layers.Flatten()(x)
    x = layers.Dense(dense_units[0], activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    """############################# LATENT SPACE #################################"""

    mu = layers.Dense(z_space, activation='linear')(x)
    sg = layers.Dense(z_space, activation='linear')(x)
    z = layers.Lambda(sample_z, output_shape=(z_space,))([mu, sg])
    zc = layers.concatenate([z, input_lbl])

    """############################### DECODER ####################################"""

    x = layers.Dense(dense_units[0], activation='relu')(zc)
    x = layers.Reshape((dense_units[0], 1))(x)

    # Residual Blocks in Decoder
    x = layers.Conv1DTranspose(128, kernel_size=3, padding='same')(x)
    x = residual_block(x, 128)
    x = layers.UpSampling1D(size=2)(x)

    x = layers.Conv1DTranspose(64, kernel_size=3, padding='same')(x)
    x = residual_block(x, 64)
    x = layers.UpSampling1D(size=2)(x)

    x = layers.Conv1DTranspose(32, kernel_size=3, padding='same')(x)
    x = residual_block(x, 32)
    x = layers.UpSampling1D(size=2)(x)

    outputs = layers.Conv1DTranspose(1, kernel_size=3, activation='sigmoid', padding='same')(x)

    """################################ RETURN ####################################"""

    model = Model([input_sig, input_lbl], outputs)
    encoder = Model([input_sig, input_lbl], mu)

    d_in = layers.Input(shape=(z_space + num_classes,))
    dh = layers.Dense(dense_units[0], activation='relu')(d_in)
    dh = layers.Reshape((dense_units[0], 1))(dh)

    dh = layers.Conv1DTranspose(128, kernel_size=3, padding='same')(dh)
    dh = residual_block(dh, 128)
    dh = layers.UpSampling1D(size=2)(dh)

    dh = layers.Conv1DTranspose(64, kernel_size=3, padding='same')(dh)
    dh = residual_block(dh, 64)
    dh = layers.UpSampling1D(size=2)(dh)

    dh = layers.Conv1DTranspose(32, kernel_size=3, padding='same')(dh)
    dh = residual_block(dh, 32)
    dh = layers.UpSampling1D(size=2)(dh)

    do = layers.Conv1DTranspose(1, kernel_size=3, activation='sigmoid', padding='same')(dh)

    decoder = Model(d_in, do)

    return model, encoder, decoder