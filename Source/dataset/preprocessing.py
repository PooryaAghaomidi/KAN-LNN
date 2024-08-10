# -*- coding: utf-8 -*-
""" preprocessing """

import os
import wfdb
import scipy
import numpy as np
import pandas as pd
from tqdm import tqdm


def resample_sig(x, fs, fs_target):
    t = np.arange(x.shape[0]).astype("float64")

    if fs == fs_target:
        return x, t

    new_length = int(x.shape[0] * fs_target / fs)
    # Resample the array if NaN values are present
    if np.isnan(x).any():
        x = pd.Series(x.reshape((-1,))).interpolate().values
    resampled_x, resampled_t = scipy.signal.resample(x, num=new_length, t=t)
    assert (
            resampled_x.shape == resampled_t.shape
            and resampled_x.shape[0] == new_length
    )
    assert np.all(np.diff(resampled_t) > 0)

    return resampled_x, resampled_t


def preprocess(init_address, final_address, length=7500):
    files = [file[:-4] for file in os.listdir(init_address) if file.endswith('.dat')]

    classes = {'W': 0, '1': 1, '2': 2, '3': 3, '4': 3, 'R': 4}

    data = []
    for file in tqdm(files):
        signal = wfdb.rdrecord(init_address + file, channels=[0]).p_signal
        segments = wfdb.rdann(init_address + file, extension='st').sample
        stages = wfdb.rdann(init_address + file, extension='st').aux_note

        for idx, start in enumerate(segments):
            if start == 1:
                start = 0
            else:
                pass

            if stages[idx][0] != 'M':
                resampled_x, _ = resample_sig(signal[start:start + length, 0], 250, 100)
                data.append(np.append(resampled_x, classes[stages[idx][0]]))

    df = pd.DataFrame(np.array(data))
    df.to_csv(final_address)

    print(f"\nThe data has been saved in {final_address}.")
