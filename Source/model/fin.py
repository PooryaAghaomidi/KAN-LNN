import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


def build_fin(input_shape, units=None, drp=0.0):
    inputs = layers.Input(shape=(input_shape[0]))

    x = layers.Dense(units=units[0])(inputs)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(drp)(x)
    x = layers.Dense(units=units[1])(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(drp)(x)
    x = layers.Dense(units=units[2])(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(drp)(x)

    outputs = layers.Dense(units=1, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs)
    print(model.summary())

    return model
