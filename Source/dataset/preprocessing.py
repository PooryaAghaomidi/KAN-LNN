# -*- coding: utf-8 -*-
""" preprocessing """

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.utils import resample


def preprocess(init_address, final_address):
    train_df = pd.read_csv(init_address, header=None)

    df_1 = train_df[train_df[187] == 1]
    df_2 = train_df[train_df[187] == 2]
    df_3 = train_df[train_df[187] == 3]
    df_4 = train_df[train_df[187] == 4]
    df_0 = (train_df[train_df[187] == 0]).sample(n=20000, random_state=42)

    df_1_upsample = resample(df_1, replace=True, n_samples=20000, random_state=123)
    df_2_upsample = resample(df_2, replace=True, n_samples=20000, random_state=124)
    df_3_upsample = resample(df_3, replace=True, n_samples=20000, random_state=125)
    df_4_upsample = resample(df_4, replace=True, n_samples=20000, random_state=126)

    train_df = pd.concat([df_0, df_1_upsample, df_2_upsample, df_3_upsample, df_4_upsample])

    counter = 0
    Clean_data = np.empty((100000, 129))

    for i in tqdm(range(len(train_df))):
        row = train_df.iloc[i]
        My_signal = row[5:133]
        My_class = int(row[187])

        Clean_data[counter, :128] = np.array(My_signal)
        Clean_data[counter, 128] = My_class
        counter = counter + 1

    np.random.shuffle(Clean_data)
    np.save(final_address, Clean_data)
