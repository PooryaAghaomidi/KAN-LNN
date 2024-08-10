import numpy as np
import pandas as pd


def generate_data(data_num, data_length):
    return pd.DataFrame(np.random.random((data_num, data_length)))
