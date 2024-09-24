import pandas as pd
from sklearn.utils import resample
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE, ADASYN
from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse


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


def smote_tomek_oversampling(data_path, save_path):
    df = pd.read_csv(data_path)
    df = df.drop(['Unnamed: 0'], axis=1)

    # Split the data into features (X) and labels (y)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Apply SMOTE followed by Tomek Links
    smote_tomek = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smote_tomek.fit_resample(X, y)

    # Create labels to distinguish between original and synthetic samples
    original_indices = set(df.index)
    synthetic_label = ['Original' if i in original_indices else 'Synthetic' for i in range(len(X_resampled))]

    # Create a new dataframe for the resampled data
    df_resampled = pd.DataFrame(X_resampled, columns=df.columns[:-1])
    df_resampled['label'] = y_resampled
    df_resampled['synthetic'] = synthetic_label

    # Save the resampled dataframe to a CSV file
    df_resampled.to_csv(save_path, index=False)

    print("SMOTE-Tomek oversampling and cleaning completed.")


def augment_signals(data_path, save_path, target_size=4000):
    # Load the dataset
    df = pd.read_csv(data_path)
    df = df.drop(['Unnamed: 0'], axis=1)  # Drop unnecessary column if present

    # Define the tsaug augmentation pipeline
    my_augmenter = (
        TimeWarp() * 5  # Random time warping 5 times in parallel
        + Quantize(n_levels=[10, 20, 30])  # Random quantize to 10-, 20-, or 30-level sets
        + Drift(max_drift=(0.1, 0.5)) @ 0.8  # 80% probability of applying drift
        + Reverse() @ 0.5  # 50% probability of reversing the sequence
    )

    # Group the data by class
    df_grouped = df.groupby('3000')

    augmented_dfs = []

    for label, group in df_grouped:
        # If the class has fewer than target_size, augment the class
        if len(group) < target_size:
            # Separate the signals and labels for the class
            X_class = group.iloc[:, :-1].values
            y_class = group.iloc[:, -1].values

            # Calculate how many more samples are needed
            signals_needed = target_size - len(group)

            # Resample the class to match the number of needed samples
            X_resampled, y_resampled = resample(X_class, y_class, n_samples=signals_needed, random_state=42)

            # Apply the tsaug augmentation
            X_augmented = my_augmenter.augment(X_resampled)

            # Create a new dataframe for the augmented samples
            df_augmented = pd.DataFrame(X_augmented, columns=df.columns[:-1])
            df_augmented['3000'] = label
            df_augmented['synthetic'] = 'Synthetic'

            # Append the original and augmented samples
            augmented_dfs.append(pd.concat([group, df_augmented]))
        else:
            # If the class already has enough samples, just append the original group
            augmented_dfs.append(group)

    # Combine all augmented groups
    df_augmented_final = pd.concat(augmented_dfs, axis=0)

    # Shuffle the final dataframe to mix original and augmented samples
    df_augmented_final = df_augmented_final.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save the augmented dataset
    df_augmented_final.to_csv(save_path, index=False)
    print(df_augmented_final.head())

    print(f"Data augmentation using tsaug completed. Saved to {save_path}.")
