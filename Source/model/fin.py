import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


def build_fin(input_shape, units=None, drp=0.1):
    inputs = layers.Input(shape=input_shape)

    x = layers.Dense(units=units[0], activation='relu')(inputs)
    x = layers.Dropout(drp)(x)

    x = layers.Dense(units=units[1], activation='relu')(x)
    x = layers.Dropout(drp)(x)

    x = layers.Dense(units=units[2], activation='relu')(x)
    x = layers.Dropout(drp)(x)

    x = layers.Dense(units=units[3], activation='relu')(x)
    x = layers.Dropout(drp)(x)

    x = layers.BatchNormalization()(x)

    outputs = layers.Dense(units=1, activation=None)

    return Model(inputs=inputs, outputs=outputs(x))
