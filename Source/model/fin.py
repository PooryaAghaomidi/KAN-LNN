from tensorflow.keras import layers
from tensorflow.keras.models import Model


def build_fin(input_shape, units=None):
    inputs = layers.Input(shape=input_shape)

    x = layers.Dense(units=units[0], activation='relu')(inputs)
    x = layers.Dense(units=units[1], activation='relu')(x)
    x = layers.Dense(units=units[2], activation='relu')(x)

    outputs = layers.Dense(units=1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    print(model.summary())

    return model
