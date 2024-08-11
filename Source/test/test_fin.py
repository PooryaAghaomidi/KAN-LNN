# -*- coding: utf-8 -*-
""" Test the model """

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from tensorflow.keras.models import load_model


def get_kurtosis(data):
    return stats.kurtosis(data)


def test_fin(model_path, test):
    print('\nGetting test results ...\n')

    model = load_model(model_path)

    test = test.drop(['3000'], axis=1)

    y_pred = []
    y_test = []
    for sig in tqdm(test.iloc):
        sig = np.array(sig)
        sig = (sig - sig.min()) / (sig.max() - sig.min())

        y_pred.append(int(model.predict(sig.reshape((1, 3000)), verbose=0)[0]))
        y_test.append(get_kurtosis(sig))

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
