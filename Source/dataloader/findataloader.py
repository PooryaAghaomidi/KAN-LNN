# -*- coding: utf-8 -*-
"""Data Loader"""

import numpy as np
import tensorflow as tf
from scipy import stats


def get_kurtosis(data):
    return stats.kurtosis(data)


class FINDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, shape, batch_size, shuffle=False):
        self.data = data
        self.shape = shape
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [np.array(self.data.iloc[k][:3000]) for k in indexes]
        x, y = self.__data_generation(list_IDs_temp)
        return x, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        x = np.empty((self.batch_size, int(self.shape[0])))
        y = np.empty((self.batch_size,))

        for idx, ID in enumerate(list_IDs_temp):
            x[idx, :] = (ID - ID.min()) / (ID.max() - ID.min())
            y[idx] = get_kurtosis(x[idx, :])

        return x, y
