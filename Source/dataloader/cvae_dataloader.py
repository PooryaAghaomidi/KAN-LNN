import torch
import numpy as np
from ssqueezepy import ssq_stft
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

        x_init = self.data.iloc[start:start + self.batch_size, :int(self.sig_shape)]
        y_init = self.data.iloc[start:start + self.batch_size, 3000]

        x = np.empty((self.batch_size, int(self.sig_shape/2), int(self.sig_shape), 2))

        for i, signal in enumerate(x_init.iloc):
            Twxo, TF, *_ = ssq_stft(np.array(signal))
            v_real = TF.real[:int(self.sig_shape/2), :int(self.sig_shape)]
            v_imag = TF.imag[:int(self.sig_shape/2), :int(self.sig_shape)]
            x[i, :, :, 0] = (v_real - v_real.min())/(v_real.max() - v_real.min())
            x[i, :, :, 1] = (v_imag - v_imag.min())/(v_imag.max() - v_imag.min())

        y_init = np.array(y_init)
        y = F.one_hot(torch.tensor(y_init, dtype=torch.int64, device=self.device), num_classes=self.lbl_shape).float()

        return torch.tensor(x, dtype=torch.float32, device=self.device), y
