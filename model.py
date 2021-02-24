import torch
from torch import nn
from torch.nn import functional as F


class LinSkip(nn.Module):
    def __init__(self, dim):
        super(LinSkip, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.Linear(dim, 4 * dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(4 * dim),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x):
        return x + self.layer(x)


class VAE(nn.Module):
    def __init__(self, dim, depth=4):
        super(VAE, self).__init__()
        self.enc = nn.Sequential(*[LinSkip(dim) for _ in range(depth)])
        self.mean = LinSkip(dim)
        self.log_variance = LinSkip(dim)
        self.dec = nn.Sequential(*[LinSkip(dim) for _ in range(depth)])

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



class MixedVAE(nn.Module):
    def __init__(self, d_continuous, d_categories, d_embedding, depth=4):
        super(MixedVAE, self).__init__()
        self.d_continuous = d_continuous
        self.d_categories = d_categories
        self.dim = d_continuous + d_embedding
        self.cat_emb = nn.Parameter(torch.randn((d_categories, d_embedding)))

        self.enc = nn.Sequential(*[LinSkip(self.dim) for _ in range(depth)])
        self.mean = LinSkip(self.dim)
        self.log_variance = LinSkip(self.dim)
        self.dec = nn.Sequential(*[LinSkip(self.dim) for _ in range(depth)])

    def sample(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std * 0.005

    def forward(self, data):
        ins = torch.cat([data.xn, data.tne], dim=1)

        h = self.enc(ins)
        mu = self.mean(h)
        log_var = self.log_variance(h)
        z = self.sample(mu, log_var)
        outs = self.dec(z)

        data.xn = outs[:, :self.d_continuous]
        data.tne = outs[:, self.d_continuous:]

        cat_scores = self.cossim(data.tne, self.cat_emb)
        data.tn = torch.argmax(cat_scores, dim=1)
        return data, mu, log_var

    def train_step(self, data):
        if not isinstance(data.tne, torch.Tensor):
            data.tne = self.cat_emb[data.tn[:, 0]]

        data, mu, logvar = self.forward(data)
        loss = self.criterion(data, mu, logvar)

        # reset observed x
        data.xn = data.xn.detach().clone()
        data.xn[~data.xmask] = data.x0[~data.xmask]
        # reset observed t
        data.tne = data.tne.detach().clone()
        data.tne[~data.tmask.squeeze(1)] = self.cat_emb[data.t0[~data.tmask]]
        return data, loss

    def eval_step(self, data):
        if not isinstance(data.tne, torch.Tensor):
            data.tne = self.cat_emb[data.tn[:, 0]].detach()
        data, mu, logvar = self.forward(data)
        loss = self.criterion(data, mu, logvar)
        data.xn[~data.xmask] = data.x0[~data.xmask]
        data.tne[~data.tmask.squeeze(1)] = self.cat_emb[data.t0[~data.tmask]]
        return data, loss

    def criterion(self, data, mu, logvar):
        cat_scores = self.cossim(data.tne, self.cat_emb)
        ce = F.nll_loss(cat_scores[data.tmask.squeeze(1)], data.t0[data.tmask], reduction="sum")
        mse = F.mse_loss(data.xn[data.xmask], data.x0[data.xmask], reduction="sum")
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl + mse + ce

    @staticmethod
    def cossim(a, b):  # for illustration set a (64 x 128), b (17 x 128)
        inner = torch.einsum("ik, jk -> ij", a, b)  # 64 x 17
        Za = torch.sqrt(a**2).sum(dim=1)  # 64
        Zb = torch.sqrt(b**2).sum(dim=1)  # 17
        Z = Za[:, None] @ Zb[None, :]  # 64 x 17
        return inner / Z  # 64 x 17
