import torch
from torch import nn


class CVAE(nn.Module):
    def __init__(self, feature_size, latent_size, class_size):
        super(CVAE, self).__init__()
        self.feature_size = feature_size
        self.latent_size = latent_size
        self.class_size = class_size

        # Encoder
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=8, stride=2, padding=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.batch_norm3 = nn.BatchNorm1d(256)
        self.fc_mu = nn.Linear(256 * (feature_size // 8) + class_size, latent_size)
        self.fc_logvar = nn.Linear(256 * (feature_size // 8) + class_size, latent_size)
        self.dropout = nn.Dropout(0.3)

        # Decoder
        self.fc3 = nn.Linear(latent_size + class_size, 256 * (feature_size // 8))
        self.deconv1 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose1d(in_channels=64, out_channels=1, kernel_size=8, stride=2, padding=3)
        self.batch_norm4 = nn.BatchNorm1d(128)
        self.batch_norm5 = nn.BatchNorm1d(64)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

        # Weight Initialization
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def encode(self, x, c):
        x = x.unsqueeze(1)  # Add channel dimension: [batch_size, 1, feature_size]
        h1 = self.leaky_relu(self.batch_norm1(self.conv1(x)))
        h2 = self.leaky_relu(self.batch_norm2(self.conv2(h1)))
        h3 = self.leaky_relu(self.batch_norm3(self.conv3(h2)))

        h3_flat = h3.view(h3.size(0), -1)
        inputs = torch.cat([h3_flat, c], 1)
        inputs = self.dropout(inputs)

        z_mu = self.fc_mu(inputs)
        z_logvar = self.fc_logvar(inputs)
        return z_mu, z_logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z, c):
        inputs = torch.cat([z, c], 1)
        h4 = self.leaky_relu(self.fc3(inputs))
        h4 = h4.view(h4.size(0), 256, self.feature_size // 8)

        h5 = self.leaky_relu(self.batch_norm4(self.deconv1(h4)))
        h6 = self.leaky_relu(self.batch_norm5(self.deconv2(h5)))
        return self.deconv3(h6).squeeze(1)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z, c)
        return reconstruction, mu, logvar