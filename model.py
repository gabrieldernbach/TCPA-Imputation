import torch
from torch import nn
from torch.nn import functional as F


class LinSkip(nn.Module):
    def __init__(self, dim):
        super(LinSkip, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(4 * dim),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x):
        return x + self.layer(x)


class VAE(nn.Module):
    def __init__(self, ins=64, hidden=256):
        super(VAE, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(ins, hidden),
            LinSkip(hidden),
            LinSkip(hidden),
            LinSkip(hidden),
            LinSkip(hidden),
        )
        self.mean = LinSkip(hidden)
        self.log_variance = LinSkip(hidden)
        self.dec = nn.Sequential(
            LinSkip(hidden),
            LinSkip(hidden),
            LinSkip(hidden),
            LinSkip(hidden),
            nn.Linear(hidden, ins)
        )

    def sample(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        x = self.enc(x)
        mu = self.mean(x)
        log_var = self.log_variance(x)
        z = self.sample(mu, log_var)
        x_ = self.dec(z)
        return x_, mu, log_var

    def train_step(self, x0, xn, missing_mask):
        xn, mu, logvar = self.forward(xn)
        loss = self.criterion(x0, xn, mu, logvar, missing_mask)

        # reset values to start value
        xn[~missing_mask] = x0[~missing_mask]
        xn = xn.detach()
        return xn, loss

    def eval_step(self, x0, xn, missing_mask):
        with torch.no_grad():
            xn, mu, logvar = self.forward(xn)
            loss = self.criterion(x0, xn, mu, logvar, missing_mask)
        xn[~missing_mask] = x0[~missing_mask]
        return xn, loss

    @staticmethod
    def criterion(x0, xn, mu, logvar, missing_mask):
        mse = F.mse_loss(xn[missing_mask], x0[missing_mask], reduction="sum")
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return mse + kl



# todo:
#   model that allows for additional categorical input
#   categorical input has embedding layer, that mapps integers to vector
#   apply residual skip connections with variational bottleneck
#   at the end of the skip compare the predictions to each of the
#   embedding vectors by cosine similarity
#   use a nll_logit loss to measure how well this was "fixed"

class MixedVAE(nn.Module):
    def __init__(self, n_proteins, n_classes, embedding_dim=128, depth=4):
        super(MixedVAE, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_classes = n_classes

        self.dim = n_proteins + embedding_dim
        self.embedd = nn.Embedding(n_classes, embedding_dim)

        self.enc = nn.Sequential(*[LinSkip(self.dim) for _ in range(depth)])
        self.mean = LinSkip(self.dim)
        self.log_variance = LinSkip(self.dim)
        self.dec = nn.Sequential(*[LinSkip(self.dim) for _ in range(depth)])

    def sample(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, proteins, tumor_id):
        tumor_vec = self.embedd(tumor_id)
        x = torch.cat([proteins, tumor_vec], dim=1)

        x = self.enc(x)
        mu = self.mean(x)
        log_var = self.log_variance(x)
        z = self.sample(mu, log_var)
        x_ = self.dec(z)


        return x_, mu, log_var

    def train_step(self, x0, xn, missing_mask):
        xn, mu, logvar = self.forward(xn)
        loss = self.criterion(x0, xn, mu, logvar, missing_mask)

        # reset values to start value
        xn[~missing_mask] = x0[~missing_mask]
        xn = xn.detach()
        return xn, loss

    def eval_step(self, x0, xn, missing_mask):
        with torch.no_grad():
            xn, mu, logvar = self.forward(xn)
            loss = self.criterion(x0, xn, mu, logvar, missing_mask)
        xn[~missing_mask] = x0[~missing_mask]
        return xn, loss

    @staticmethod
    def criterion(x0, xn, mu, logvar, missing_mask):
        mse = F.mse_loss(xn[missing_mask], x0[missing_mask], reduction="sum")
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return mse + kl
