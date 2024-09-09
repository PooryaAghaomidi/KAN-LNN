import pandas as pd


def n1_dataset(input_csv_path, output_csv_path, scaling_factor=1.0):
    # Load the dataset
    df = pd.read_csv(input_csv_path)

    # Rename the last column as 'label' for clarity
    df.rename(columns={df.columns[-1]: 'label'}, inplace=True)

    # Remove 'Unnamed: 0' column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Separate N1 signals (label == 1)
    n1_signals = df[df['label'] == 1]

    # Count of N1 signals
    n1_count = len(n1_signals)

    # Initialize a list to hold the sampled data
    balanced_data = [n1_signals]

    # Sample from the other classes based on the scaling factor
    for label in [0, 2, 3, 4]:
        class_signals = df[df['label'] == label]
        sample_size = int(n1_count * scaling_factor)  # Sample size for this class

        # If the available samples are less than the sample size, take all samples
        if len(class_signals) < sample_size:
            sampled_class_signals = class_signals  # Take all available samples
        else:
            sampled_class_signals = class_signals.sample(n=sample_size, random_state=42)

        balanced_data.append(sampled_class_signals)

    # Concatenate all the balanced data
    final_df = pd.concat(balanced_data).reset_index(drop=True)

    # Shuffle the final dataset
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save the final dataset to the output path
    final_df.to_csv(output_csv_path, index=False)
