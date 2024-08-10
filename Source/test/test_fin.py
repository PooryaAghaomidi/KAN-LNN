# -*- coding: utf-8 -*-
""" Test the model """

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


def test_fin(model_path, X_test, y_test):
    print('\nGetting test results ...\n')

    model = load_model(model_path)
    y_pred = model.predict(X_test)

    x_axis = np.arange(len(y_test))
    plt.figure(figsize=(12, 4))
    plt.scatter(x_axis, y_test, label='True')
    plt.scatter(x_axis, y_pred, label='Predicted')
    plt.grid()
    plt.legend()
    plt.show()

    y_true_sample = y_test[200:300]
    y_pred_sample = y_pred[200:300]

    plt.figure(figsize=(12, 4))
    plt.plot(y_true_sample, label='True')
    plt.plot(y_pred_sample, label='Predicted')
    plt.grid()
    plt.legend()
    plt.show()
