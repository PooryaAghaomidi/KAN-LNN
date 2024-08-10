# -*- coding: utf-8 -*-
"""Data Loader"""

import math
import numpy as np
import tensorflow as tf
from ssqueezepy import ssq_stft
from tensorflow.keras.utils import to_categorical


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, image_shape, signal_shape, batch_size, cls_num, width, overlap, shuffle=False):
        self.data = data
        self.image_shape = image_shape
        self.signal_shape = signal_shape
        self.batch_size = batch_size
        self.cls_num = cls_num
        self.shuffle = shuffle
        self.width = width

        self.diff = self.width - overlap
        self.n = math.floor((int(signal_shape[0]) - overlap) / self.diff)

        self.on_epoch_end()

    def windowing(self, signal):
        x = []
        for j in range(self.n):
            x.append(signal[self.diff * j: self.width + self.diff * j])
        return x

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [np.array(self.data.iloc[k]) for k in indexes]
        x, y = self.__data_generation(list_IDs_temp)
        return x, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        x = np.empty((self.batch_size, self.n, int(self.image_shape[1]), int(self.image_shape[2]), int(self.image_shape[3])))
        s = np.empty((self.batch_size, int(self.signal_shape[0]), int(self.signal_shape[1])))
        y = np.empty((self.batch_size), dtype=int)

        for idx, ID in enumerate(list_IDs_temp):
            my_signal = ID[:3000]
            segments = self.windowing(my_signal)

            for idxx, segment in enumerate(segments):
                Twxo, TF, *_ = ssq_stft(segment)
                v = np.abs(TF)[:int(self.image_shape[1]), :]
                x[idx, idxx, :, :, 0] = (v - v.min()) / (v.max() - v.min())

            s[idx, :, 0] = (my_signal - my_signal.min()) / (my_signal.max() - my_signal.min())
            y[idx] = int(ID[3000])

        return [x, s], to_categorical(y, num_classes=self.cls_num)
