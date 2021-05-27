import torch
import torch.nn as nn


class FlowMlp(nn.Module):
    def __init__(self, ins, hidden, reps=10):
        super().__init__()
        self.reps = reps
        self.ins = nn.Linear(ins, hidden)
        self.block = nn.Sequential(
            nn.Linear(hidden, hidden*2),
            nn.LeakyReLU(),
            nn.Linear(hidden*2, hidden),
        )
        self.outs = nn.Linear(hidden, ins)

    def forward(self, x):
        x = self.ins(x)
        for _ in range(self.reps):
            x = self.block(x) + x
        return self.outs(x)


class TransposeLast(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.transpose(-1, -2)


class MixerLayer(nn.Module):
    def __init__(self, n_samp, n_dim):
        super().__init__()
        self.tokenmix = nn.Sequential(
            nn.LayerNorm(n_dim),
            TransposeLast(),
            nn.Linear(n_samp, n_samp * 2),
            nn.GELU(),
            nn.Linear(n_samp * 2, n_samp),
            TransposeLast(),
        )
        self.channelmix = nn.Sequential(
            nn.LayerNorm(n_dim),
            nn.Linear(n_dim, n_dim * 2),
            nn.GELU(),
            nn.Linear(n_dim * 2, n_dim)
        )

    def forward(self, x):
        x = self.tokenmix(x) + x
        x = self.channelmix(x) + x
        return x

class MlpMix(nn.Module):
    def __init__(self, reps, n_samp, hidden, ins):
        super().__init__()
        self.in_dim = nn.Linear(ins, hidden)
        self.in_samp = nn.Linear(1, n_samp)

        mls = [MixerLayer(n_samp, hidden) for _ in range(reps)]
        self.block = nn.Sequential(*mls)

        self.out_dim = nn.Linear(hidden, ins)
        self.out_samp = nn.Linear(n_samp, 1)

    def forward(self, x): # from batch x ins
        # to batch x hidden
        x = self.in_dim(x)
        # to batch x n_samp x hidden
        x = self.in_samp(x[..., None]).transpose(-2, -1)

        # stay batch x n_samp x hidden
        x = self.block(x)

        # to batch x hidden
        x = self.out_samp(x.transpose(-2, -1)).squeeze(-1)
        # batch x ins
        x = self.out_dim(x)
        return x


if __name__ == "__main__":
    X = torch.randn(512, 18)
    mlp = FlowMlp(ins=18, hidden=128, reps=100)
    O = mlp(X)
    assert X.shape == O.shape

    mlpmix = MlpMix(ins=18, n_samp=128, hidden=128, reps=50)
    O = mlpmix(X)
    assert X.shape == O.shape
