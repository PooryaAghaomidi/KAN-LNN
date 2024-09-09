from .ncps.tf import LTCCell
from .ncps.wirings import AutoNCP
from tensorflow.keras import layers
from tensorflow.keras.models import Model


def build_fin(input_shape, conv_units, lnn_units, fc_units):
    inputs = layers.Input(shape=input_shape)

    # Multi-scale 1D convolutional layers
    x = layers.Conv1D(filters=conv_units[0], kernel_size=3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(filters=conv_units[1], kernel_size=5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(filters=conv_units[2], kernel_size=7, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Liquid Neural Network
    wiring = AutoNCP(lnn_units[0], lnn_units[1])
    rnn_cell = LTCCell(wiring)
    x = layers.RNN(rnn_cell, return_sequences=False)(x)

    # Common Dense Network
    x = layers.Dense(fc_units[0], activation='relu')(x)
    x = layers.Dropout(0.0)(x)

    # Split into separate outputs
    skew_branch = layers.Dense(fc_units[1], activation='relu')(x)
    skew_output = layers.Dense(1, name='skew_output')(skew_branch)

    kurt_branch = layers.Dense(fc_units[1], activation='relu')(x)
    kurt_output = layers.Dense(1, name='kurt_output')(kurt_branch)

    model = Model(inputs=inputs, outputs=[skew_output, kurt_output])
    print(model.summary())

    return model
