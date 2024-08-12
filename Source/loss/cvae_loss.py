import torch
from torch.nn import functional as F


class CVAELoss:
    def __init__(self):
        self.weight_1 = 1
        self.weight_2 = 1

    def cvae_loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return self.weight_1 * BCE + self.weight_2 * KLD
