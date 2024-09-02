import torch
import pandas as pd
from ssqueezepy import issq_stft
from imblearn.over_sampling import SMOTE, ADASYN


def smote_oversampling(data_path, save_path):
    df = pd.read_csv(data_path)
    df = df.drop(['Unnamed: 0'], axis=1)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    original_indices = set(df.index)
    synthetic_label = ['Original' if i in original_indices else 'Synthetic' for i in range(len(X_resampled))]

    df_resampled = pd.DataFrame(X_resampled, columns=df.columns[:-1])
    df_resampled['label'] = y_resampled
    df_resampled['synthetic'] = synthetic_label

    df_resampled.to_csv(save_path, index=False)

    print("SMOTE oversampling completed.")


def adasyn_oversampling(data_path, save_path):
    df = pd.read_csv(data_path)
    df = df.drop(['Unnamed: 0'], axis=1)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    adasyn = ADASYN(sampling_strategy='minority', random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)

    original_indices = set(df.index)
    synthetic_label = ['Original' if i in original_indices else 'Synthetic' for i in range(len(X_resampled))]

    df_resampled = pd.DataFrame(X_resampled, columns=df.columns[:-1])
    df_resampled['label'] = y_resampled
    df_resampled['synthetic'] = synthetic_label

    df_resampled.to_csv(save_path, index=False)

    print("SMOTE oversampling completed.")


def cvae_oversampling(data_path, model, device, latent_size, save_path):
    df = pd.read_csv(data_path)
    df = df.drop(['Unnamed: 0'], axis=1)

    labels = [int(i) for i in list(df['3000'])]
    counts = {'0': labels.count(0),
              '1': labels.count(1),
              '2': labels.count(2),
              '3': labels.count(3),
              '4': labels.count(4)}

    maximum_count = max(list(counts.values()))
    for idx, count in enumerate(counts.values()):
        iterations = maximum_count - count

        if iterations != 0:
            for i in range(iterations):
                z = torch.randn(1, latent_size).to(device)

                c_one_hot = torch.zeros(1, model.condition_dim).to(device)
                c_one_hot[0, idx] = 1.0

                generated_signal = model.decode(z, c_one_hot).cpu().numpy()

                real_part = generated_signal[0, :, :, 0]
                imag_part = generated_signal[0, :, :, 1]
                TF = real_part + imag_part * 1j
                signal = issq_stft(TF).flatten()

