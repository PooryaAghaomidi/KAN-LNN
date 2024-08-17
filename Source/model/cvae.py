import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention2D(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super(SelfAttention2D, self).__init__()
        self.in_proj = nn.Conv2d(d_embed, 3 * d_embed, kernel_size=1, bias=in_proj_bias)
        self.out_proj = nn.Conv2d(d_embed, d_embed, kernel_size=1, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):
        batch_size, channels, height, width = x.shape
        d_embed = channels

        # Apply in_proj to get q, k, v
        qkv = self.in_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # Reshape into [batch_size, n_heads, d_head, height * width]
        q = q.view(batch_size, self.n_heads, self.d_head, height * width)
        k = k.view(batch_size, self.n_heads, self.d_head, height * width)
        v = v.view(batch_size, self.n_heads, self.d_head, height * width)

        # Scaled Dot-Product Attention
        attn_weights = q.transpose(-1, -2) @ k / math.sqrt(self.d_head)

        if causal_mask:
            mask = torch.ones_like(attn_weights).triu(1)
            attn_weights = attn_weights.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)

        output = attn_weights @ v.transpose(-1, -2)
        output = output.view(batch_size, self.n_heads * self.d_head, height, width)

        return self.out_proj(output)


class CrossAttention2D(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super(CrossAttention2D, self).__init__()
        self.q_proj = nn.Conv2d(d_embed, d_embed, kernel_size=1, bias=in_proj_bias)
        self.k_proj = nn.Conv2d(d_cross, d_embed, kernel_size=1, bias=in_proj_bias)
        self.v_proj = nn.Conv2d(d_cross, d_embed, kernel_size=1, bias=in_proj_bias)
        self.out_proj = nn.Conv2d(d_embed, d_embed, kernel_size=1, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y):
        batch_size, channels, height, width = x.shape
        d_embed = channels

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(batch_size, self.n_heads, self.d_head, height * width)
        k = k.view(batch_size, self.n_heads, self.d_head, height * width)
        v = v.view(batch_size, self.n_heads, self.d_head, height * width)

        attn_weights = q.transpose(-1, -2) @ k / math.sqrt(self.d_head)
        attn_weights = F.softmax(attn_weights, dim=-1)

        output = attn_weights @ v.transpose(-1, -2)
        output = output.view(batch_size, self.n_heads * self.d_head, height, width)

        return self.out_proj(output)


class VAE_ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        residue = x
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)


class VAE_AttentionBlock2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention2D(1, channels)

    def forward(self, x):
        residue = x
        x = self.groupnorm(x)

        n, c, h, w = x.shape
        x = self.attention(x)

        x += residue
        return x


class CVAE(nn.Module):
    def __init__(self, in_channels, latent_dim, condition_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim

        # Adjust the first layer to match the concatenated input size
        input_channels = 7

        # Define the encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            VAE_ResidualBlock2D(32, 32),
            VAE_ResidualBlock2D(32, 32),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            VAE_ResidualBlock2D(32, 64),
            VAE_ResidualBlock2D(64, 64),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            VAE_ResidualBlock2D(64, 128),
            VAE_ResidualBlock2D(128, 128),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            VAE_ResidualBlock2D(128, 256),
            VAE_ResidualBlock2D(256, 256),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            VAE_ResidualBlock2D(256, 256),
            VAE_ResidualBlock2D(256, 256),

            VAE_ResidualBlock2D(256, 256),
            VAE_AttentionBlock2D(256),
            VAE_ResidualBlock2D(256, 256),

            nn.GroupNorm(32, 256),
            nn.SiLU(),
        )

        # Add a fully connected layer to map the output to latent dimensions
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(256 * 8 * 16, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 16, latent_dim)

        # Define the decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim + condition_dim, 8, kernel_size=1, padding=0),  # Start with a small number of channels
            nn.Conv2d(8, 256, kernel_size=3, padding=1),
            VAE_ResidualBlock2D(256, 256),
            VAE_AttentionBlock2D(256),
            nn.Upsample(scale_factor=2, mode='bilinear'),

            VAE_ResidualBlock2D(256, 256),
            VAE_ResidualBlock2D(256, 256),
            VAE_ResidualBlock2D(256, 256),
            VAE_ResidualBlock2D(256, 256),
            nn.Upsample(scale_factor=4, mode='bilinear'),  # Upsample to 32x32

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            VAE_ResidualBlock2D(128, 128),
            VAE_ResidualBlock2D(128, 128),
            VAE_ResidualBlock2D(128, 128),
            nn.Upsample(scale_factor=4, mode='bilinear'),  # Upsample to 64x64

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            VAE_ResidualBlock2D(64, 64),
            VAE_ResidualBlock2D(64, 64),
            VAE_ResidualBlock2D(64, 64),
            nn.Upsample(scale_factor=(4, 8), mode='bilinear'),  # Upsample to 128x256

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            VAE_ResidualBlock2D(32, 32),
            VAE_ResidualBlock2D(32, 32),
            VAE_ResidualBlock2D(32, 32),
            nn.GroupNorm(32, 32),
            nn.SiLU(),

            nn.Conv2d(32, 2, kernel_size=3, padding=1)  # Final layer to produce 2 channels
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 256 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, condition_dim),
            nn.Softmax(dim=1)
        )

    def encode(self, x, c):
        c_expanded = c.unsqueeze(1).unsqueeze(1)
        c_expanded = c_expanded.expand(-1, x.size(1), x.size(2), -1)

        # Concatenate input and condition along the channel dimension
        x_cond = torch.cat([x, c_expanded], dim=-1)
        x_cond = x_cond.permute(0, 3, 1, 2)

        # Pass through the encoder convolutional layers
        x_cond = self.encoder_conv(x_cond)

        # Flatten the output
        x_flat = self.flatten(x_cond)

        # Map to latent dimension
        mean = self.fc_mu(x_flat)
        log_variance = self.fc_logvar(x_flat)

        # Clamp the log variance
        log_variance = torch.clamp(log_variance, -30, 20)

        return mean, log_variance

    def decode(self, z, c):
        # Concatenate latent vector and condition
        z_cond = torch.cat([z, c], dim=1).unsqueeze(-1).unsqueeze(-1)

        # Pass through the decoder
        z_cond /= 0.18215
        for module in self.decoder:
            z_cond = module(z_cond)

        return z_cond.permute(0, 2, 3, 1)

    def reparameterize(self, mean, log_variance):
        stdev = torch.exp(0.5 * log_variance)
        epsilon = torch.randn_like(stdev)
        return mean + stdev * epsilon

    def forward(self, x, c):
        # Encode input to get mean and log variance
        mean, log_variance = self.encode(x, c)

        # Reparameterize to get latent vector z
        z = self.reparameterize(mean, log_variance)

        # Decode the latent vector to reconstruct the input
        reconstruction = self.decode(z, c)

        class_logits = self.classifier(reconstruction)

        return reconstruction, mean, log_variance, class_logits
