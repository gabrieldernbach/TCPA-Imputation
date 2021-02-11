import numpy as np
from torch.utils.data import Dataset, DataLoader

import torch
from torch import nn
import torch.nn.functional as F


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
            LinSkip(hidden)
        )

        self.mean = LinSkip(hidden)
        self.log_variance = LinSkip(hidden)

        self.dec = nn.Sequential(
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

    def train_step(self, x0, xn, mask):
        xn, mu, logvar = self.forward(xn)
        loss = self.criterion(x0, xn, mu, logvar, mask)
        # todo: (1)
        #  ideally this would include resetting masked indices in xn with the
        #  original values from x0. Also gradients must be freed.
        #  currently this is performed in the training step post processing in
        #  the training loop itself
        return xn, loss

    @staticmethod
    def criterion(x0, xn, mu, logvar, mask):
        mse = F.mse_loss(xn[mask], x0[mask], reduction="sum")
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return mse + kl


class MissingEntryDataset(Dataset):
    def __init__(self, data: torch.tensor, min_missing: float, max_missing: float):
        self.data = data
        self.min_missing = min_missing
        self.max_missing = max_missing

    def __getitem__(self, idx):
        x0 = self.data[idx]

        prob = np.random.uniform(self.min_missing, self.max_missing)
        mask = np.random.binomial(1, prob, x0.shape)
        mask = torch.tensor(mask).bool()

        # todo:
        #   Ideally we would already create `xn` where the values
        #   with active `mask` are set to random.
        #   However x0 seems to carry gradients!?
        return x0, mask

    def __len__(self):
        return len(self.data)


def main():
    adjacency, samples = torch.load("dag_nn.pt")
    ds = MissingEntryDataset(samples, min_missing=0.1, max_missing=0.9)
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    model = VAE(ins=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    for epoch in range(200):
        model.train()
        for x0, mask in dl:
            xn = x0.detach().clone()  # todo: why can this not be moved to the dataloader
            xn[mask] = torch.randn(mask.shape)[mask]
            for it in range(200):
                xn, loss = model.train_step(x0, xn, mask)
                print(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # reset observed values to original state
                # todo: see (1), this section should go in the model but behaviour of x0 is unexpected
                xn[~mask] = x0[~mask]
                xn = xn.detach()
                x0 = x0.detach()  # todo: why is this even necessary? It is not necessary for model updates

if __name__ == "__main__":
    main()