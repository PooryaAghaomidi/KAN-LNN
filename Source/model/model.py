# -*- coding: utf-8 -*-
"""Unet model"""

from tensorflow.keras import layers
from tensorflow.keras.models import load_model, Model


def build(image_shape, signal_shape, num_classes, fin_model, base_model, kan_units):
    """############################### Image ####################################"""

    input_img = layers.Input(shape=image_shape)

    base_model_raw = load_model(base_model)
    base_inp = base_model_raw.input
    base_out = base_model_raw.layers[-5].output
    final_base_model = Model(base_inp, base_out)

    for layer in final_base_model.layers:
        layer.trainable = True

    img_output = final_base_model(input_img)

    """################################# FIN #####################################"""

    input_sig = layers.Input(shape=signal_shape)

    fin_inp = fin_model.input
    fin_out = fin_model.layers[-8].output
    fin_model = Model(fin_inp, fin_out)

    for layer in fin_model.layers:
        layer.trainable = True

    fin_output = fin_model(input_sig)

    """############################### Concat ####################################"""
    # Concatenate image and signal feature outputs
    contacted = layers.Concatenate()([fin_output, img_output])

    # Add Dropout for regularization
    x = layers.Dropout(0.2)(contacted)  # Increased dropout rate for better generalization

    # Add Dense layers with higher complexity
    for kan_unit in kan_units:
        x = layers.Dense(kan_unit, activation='relu')(x)

    # Output Layer
    output = layers.Dense(num_classes, activation='softmax')(x)

    # Build the final model
    model = Model(inputs=[input_img, input_sig], outputs=output)

    print(model.summary())

    return model
