# -*- coding: utf-8 -*-
""" Test the model """

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.models import load_model


def test_fin(model, data_test, skew_test, kurt_test):
    print('\nGetting test results ...\n')

    y_pred_kurt = []
    y_pred_skew = []
    y_test_kurt = []
    y_test_skew = []
    for idx, sig in tqdm(enumerate(data_test.iloc)):
        skew, kurt = model.predict(np.array(sig).reshape((1, 3000)), verbose=0)
        y_pred_kurt.append(kurt[0][0])
        y_pred_skew.append(skew[0][0])

        y_test_kurt.append(skew_test.iloc[idx][0])
        y_test_skew.append(kurt_test.iloc[idx][0])

    x_axis = np.arange(len(y_test_kurt))

    plt.figure(figsize=(12, 4))
    plt.scatter(x_axis, y_test_skew, label='True Skew')
    plt.scatter(x_axis, y_pred_skew, label='Predicted Skew')
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.scatter(x_axis, y_test_kurt, label='True Kurt')
    plt.scatter(x_axis, y_pred_kurt, label='Predicted Kurt')
    plt.grid()
    plt.legend()
    plt.show()

    y_true_sample = y_test_skew[200:300]
    y_pred_sample = y_pred_skew[200:300]

    plt.figure(figsize=(12, 4))
    plt.plot(y_true_sample, label='True Skew')
    plt.plot(y_pred_sample, label='Predicted Skew')
    plt.grid()
    plt.legend()
    plt.show()

    y_true_sample = y_test_kurt[200:300]
    y_pred_sample = y_pred_kurt[200:300]

    plt.figure(figsize=(12, 4))
    plt.plot(y_true_sample, label='True Kurt')
    plt.plot(y_pred_sample, label='Predicted Kurt')
    plt.grid()
    plt.legend()
    plt.show()
