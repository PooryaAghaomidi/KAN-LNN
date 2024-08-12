import torch
import numpy as np
from torch.nn import functional as F


class CVAEDataGenerator:
    def __init__(self, data, sig_shape, lbl_shape, batch_size, my_device):
        self.data = data
        self.sig_shape = sig_shape
        self.lbl_shape = lbl_shape
        self.batch_size = batch_size
        self.device = my_device

    def data_generation(self, idx):
        start = idx * self.batch_size

        x_init = self.data.iloc[start:start + self.batch_size, :3000]
        y_init = self.data.iloc[start:start + self.batch_size, 3000]

        x = np.empty((self.batch_size, int(self.sig_shape)))

        for i, signal in enumerate(x_init.iloc):
            signal = np.array(signal)
            x[i, :] = (signal - signal.min())/(signal.max() - signal.min())

        y_init = np.array(y_init)
        y = F.one_hot(torch.tensor(y_init, dtype=torch.int64, device=self.device), num_classes=self.lbl_shape).float()

        return torch.tensor(x, dtype=torch.float32, device=self.device), y
