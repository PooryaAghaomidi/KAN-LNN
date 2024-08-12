import torch
from torch import nn


class CVAE(nn.Module):
    def __init__(self, feature_size, latent_size, class_size):
        super(CVAE, self).__init__()
        self.feature_size = feature_size
        self.class_size = class_size

        # Encoder
        self.fc1 = nn.Linear(feature_size + class_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc21 = nn.Linear(512, latent_size)
        self.fc22 = nn.Linear(512, latent_size)

        # Decoder
        self.fc3 = nn.Linear(latent_size + class_size, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.fc5 = nn.Linear(1024, feature_size)

        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(1024)
        self.batchnorm2 = nn.BatchNorm1d(512)

    def encode(self, x, c):
        inputs = torch.cat([x, c], 1)
        h1 = self.relu(self.batchnorm1(self.fc1(inputs)))
        h2 = self.relu(self.batchnorm2(self.fc2(h1)))
        z_mu = self.fc21(h2)
        z_var = self.fc22(h2)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        inputs = torch.cat([z, c], 1)
        h3 = self.relu(self.fc3(inputs))
        h4 = self.relu(self.fc4(h3))
        return self.fc5(h4)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar
