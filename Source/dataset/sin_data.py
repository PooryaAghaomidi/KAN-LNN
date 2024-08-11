import numpy as np
import pandas as pd


def generate_data(data_num, data_length):
    return pd.DataFrame(np.random.random((data_num, data_length)))


# def generate_random_ecg_waveform(length, sampling_rate=100):
#     """
#     Generate a synthetic ECG-like waveform with randomness in the amplitude, timing, and width,
#     and with a random heart rate between 60 and 90 beats per minute.
#
#     Parameters:
#     - length: The number of samples in the ECG signal.
#     - sampling_rate: The sampling rate in Hz (samples per second).
#
#     Returns:
#     - ecg_signal: A synthetic ECG signal as a numpy array.
#     """
#     # Random heart rate between 60 and 90 bpm
#     heart_rate = np.random.randint(60, 91)
#
#     # Time vector
#     t = np.linspace(0, length / sampling_rate, length)
#
#     # Heartbeat period in seconds
#     heartbeat_period = 60.0 / heart_rate
#
#     # Create an empty signal
#     ecg_signal = np.zeros_like(t)
#
#     # Number of heartbeats in the signal
#     num_heartbeats = int(length / (heartbeat_period * sampling_rate))
#
#     for i in range(num_heartbeats):
#         # Time offset for each heartbeat
#         time_offset = i * heartbeat_period
#
#         # Random variations in the waveform parameters with clamping for stability
#         p_amplitude = 0.1 + 0.05 * np.random.randn()
#         p_width = np.clip(0.01 + 0.005 * np.random.randn(), 0.005, 0.02)
#         p_center = time_offset + 0.2 + 0.05 * np.random.randn()
#
#         q_amplitude = -0.15 + 0.05 * np.random.randn()
#         q_width = np.clip(0.001 + 0.0005 * np.random.randn(), 0.0005, 0.005)
#         q_center = time_offset + 0.25 + 0.01 * np.random.randn()
#
#         r_amplitude = 1.0 + 0.3 * np.random.randn()
#         r_width = np.clip(0.002 + 0.001 * np.random.randn(), 0.001, 0.01)
#         r_center = time_offset + 0.3 + 0.02 * np.random.randn()
#
#         s_amplitude = -0.2 + 0.05 * np.random.randn()
#         s_width = np.clip(0.001 + 0.0005 * np.random.randn(), 0.0005, 0.005)
#         s_center = time_offset + 0.35 + 0.01 * np.random.randn()
#
#         t_amplitude = 0.3 + 0.1 * np.random.randn()
#         t_width = np.clip(0.015 + 0.005 * np.random.randn(), 0.005, 0.03)
#         t_center = time_offset + 0.45 + 0.05 * np.random.randn()
#
#         # P-wave (small upward deflection)
#         p_wave = p_amplitude * np.exp(-((t - p_center) ** 2) / p_width)
#
#         # QRS complex (sharp spike)
#         q_wave = q_amplitude * np.exp(-((t - q_center) ** 2) / q_width)
#         r_wave = r_amplitude * np.exp(-((t - r_center) ** 2) / r_width)
#         s_wave = s_amplitude * np.exp(-((t - s_center) ** 2) / s_width)
#
#         # T-wave (broader upward deflection)
#         t_wave = t_amplitude * np.exp(-((t - t_center) ** 2) / t_width)
#
#         # Combine the waves to form the ECG signal
#         ecg_signal += p_wave + q_wave + r_wave + s_wave + t_wave
#
#     # Add some random noise to the signal to increase variability
#     noise = 0.05 * np.random.randn(length)
#     ecg_signal += noise
#
#     return ecg_signal
#
#
# def generate_data(data_num, data_length):
#     """
#     Generate a DataFrame of synthetic ECG-like waveforms.
#
#     Parameters:
#     - data_num: The number of ECG signals to generate (rows).
#     - data_length: The length of each ECG signal (columns).
#
#     Returns:
#     - df: A pandas DataFrame containing the synthetic ECG signals.
#     """
#     print('\nGenerate synthetic ECG ...\n')
#
#     if os.path.isfile('dataset/synthetic.csv'):
#         df = pd.read_csv('dataset/synthetic.csv')
#     else:
#         ecg_signals = []
#         for _ in tqdm(range(data_num)):
#             v = generate_random_ecg_waveform(data_length)
#             v = (v - v.min()) / (v.max() - v.min())
#             ecg_signals.append(v)
#         df = pd.DataFrame(ecg_signals)
#         df.to_csv('dataset/synthetic.csv', index=False)
#     return df