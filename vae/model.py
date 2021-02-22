import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.fc_enc = nn.Linear(image_size, h_dim)
        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_log_var = nn.Linear(h_dim, z_dim)
        self.fc_dec1 = nn.Linear(z_dim, h_dim)
        self.fc_dec2 = nn.Linear(h_dim, image_size)

    def encode(self, x):
        h = F.relu(self.fc_enc(x))
        return self.fc_mu(h), self.fc_log_var(h)

    # for factorized Gaussian posterior
    def reparametrize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.rand_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc_dec1(z))
        return F.sigmoid(self.fc_dec2(h))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparametrize(mu, log_var)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, log_var