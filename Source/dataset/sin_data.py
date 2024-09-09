import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import chirp
import random

seed = 0
np.random.seed(seed)
random.seed(seed)


# Min-Max normalization function
def min_max_normalize(signal):
    min_val = np.min(signal)
    max_val = np.max(signal)
    return (signal - min_val) / (max_val - min_val)


# Signal generation function
def generate_synthetic_signal(signal_type, data_length):
    t = np.linspace(0, 1, data_length)

    if signal_type == 'sinusoidal':
        frequency = random.uniform(0.1, 15)  # Wider frequency range
        phase = random.uniform(0, 2 * np.pi)
        amplitude = random.uniform(0.1, 5)  # Wider amplitude range
        return amplitude * np.sin(2 * np.pi * frequency * t + phase)

    elif signal_type == 'chirp':
        start_freq = random.uniform(0.1, 3)  # More diverse start frequency
        end_freq = random.uniform(2, 20)  # More diverse end frequency
        return chirp(t, f0=start_freq, f1=end_freq, t1=1, method='linear')

    elif signal_type == 'random_normal':
        return np.random.normal(loc=random.uniform(-0.5, 0.5), scale=random.uniform(0.5, 2), size=data_length)

    elif signal_type == 'random_uniform':
        return np.random.uniform(-2, 2, size=data_length)  # Wider uniform range

    elif signal_type == 'step':
        step_point = random.randint(data_length // 4, 3 * data_length // 4)
        step_value = random.uniform(0.5, 2)  # Vary the step height
        return np.concatenate((np.zeros(step_point), step_value * np.ones(data_length - step_point)))

    else:
        raise ValueError("Invalid signal type")


# Main function to generate dataset
def generate_data(data_num, data_length):
    data_list = []
    skew_list = []
    kurt_list = []

    for _ in range(data_num):
        # Randomly select a signal type
        signal_type = random.choice(['chirp', 'random_uniform', 'step'])

        # Generate the signal
        signal = generate_synthetic_signal(signal_type, data_length)

        # Normalize the signal using Min-Max scaling
        signal = min_max_normalize(signal)

        # Compute skewness, kurtosis
        signal_skewness = skew(signal)
        signal_kurtosis = kurtosis(signal)

        # Add to the lists
        data_list.append(signal)
        skew_list.append(signal_skewness)
        kurt_list.append(signal_kurtosis)

    # Convert lists to DataFrame
    df = pd.DataFrame(data_list)
    skew_df = pd.DataFrame(skew_list, columns=['Skewness'])
    kurt_df = pd.DataFrame(kurt_list, columns=['Kurtosis'])

    return df, skew_df, kurt_df
