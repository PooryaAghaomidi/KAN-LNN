import torch
from torch.nn import functional as F


class CVAELoss:
    def __init__(self, weight_1=1, weight_2=1):
        self.weight_1 = weight_1
        self.weight_2 = weight_2

    def cvae_loss_function(self, recon_x, x, mu, logvar):
        MSE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = (self.weight_1 * MSE + self.weight_2 * KLD)

        return total_loss
