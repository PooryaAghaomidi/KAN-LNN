from tqdm import tqdm
import scipy.stats as stats


def get_kurtosis(data):
    return stats.kurtosis(data)


def get_all_kurtosis(data):
    print('\nGenerating random data ...\n')
    kurtosis = []
    for sig in tqdm(data.iloc):
        kurtosis.append(get_kurtosis(sig))
    return kurtosis
