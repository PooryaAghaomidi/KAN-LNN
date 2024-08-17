import torch
from torch.nn import functional as F


class CVAELoss:
    def __init__(self, weight_1, weight_2, weight_3, start_epoch, end_epoch):
        self.weight_1 = weight_1
        self.weight_2 = weight_2
        self.weight_3 = weight_3
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.current_epoch = 0

    def anneal_weights(self):
        # Calculate the annealing factor based on the current epoch
        annealing_factor = min(1, max(0, (self.current_epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)))
        self.weight_3 = annealing_factor * self.weight_3

    def cvae_loss_function(self, recon_x, x, mu, logvar, class_pred, true_labels):
        # Calculate MSE (Reconstruction Loss)
        MSE = torch.log(1 + F.mse_loss(recon_x, x, reduction='sum'))

        # Calculate KLD (KL-Divergence Loss)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Calculate Classification Loss (Categorical Cross-Entropy)
        classification_loss = F.cross_entropy(class_pred, true_labels, reduction='sum')

        # Apply dynamic weighting
        total_loss = (self.weight_1 * MSE +
                      self.weight_2 * KLD +
                      self.weight_3 * classification_loss)

        return total_loss

    def update_epoch(self, epoch):
        self.current_epoch = epoch
        self.anneal_weights()
