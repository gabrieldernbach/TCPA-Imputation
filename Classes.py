import os
import torch
import torch as tc
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import random
import torch.nn.functional as F
import pdb

class Mydataset(Dataset):
    def __init__(self, data, p):
        self.data = data
        self.tensor_size = self.data.shape[1]
        self.p = p
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        target = self.data[i, :]

        if isinstance(self.p,tuple):
            q = random.uniform(self.p[0], self.p[1])
        else:
            q = self.p

        mask = tc.bernoulli(tc.ones(self.tensor_size)*q) == 0
        masked_data = target.clone()
        masked_data[mask]= float('nan')
        return masked_data.float(), target.float()

def bn_linear(input_dim: int, output_dim: int):
    return nn.Sequential(
        nn.BatchNorm1d(input_dim),
        nn.Linear(input_dim, output_dim, bias=False)
    )

######################################################################
class ResBlock(nn.Module):
    def __init__(self, input_dim, width,  act_bool, activation=nn.ReLU()):
        super(ResBlock, self).__init__()

        self.alpha = nn.Parameter(torch.tensor(1.0, requires_grad=False))
        self.input_dim, self.act_bool = input_dim, act_bool
        self.layers = nn.Sequential(
            bn_linear(input_dim, width),
            activation,
            nn.Dropout(p=0.1),
            bn_linear(width, input_dim),
            activation if act_bool else nn.Identity()
        )

    def forward(self,x):
        residual = x.clone() if self.act_bool else tc.zeros(x.shape).to(device)
        return residual*self.alpha + self.layers(x)


class ResNet(nn.Module):
    def __init__(self, input_dim, width, depth, variational, SHAP= False):
        super(ResNet,self).__init__()
        self.SHAP = SHAP
        self.init_nan = nn.Parameter(torch.tensor(0.0, requires_grad=True))
        self.layers = nn.Sequential(
            *[ResBlock(input_dim, width, act_bool=True) for _ in range(depth)],
            ResBlock(input_dim, width, act_bool= False)
        )

    def forward(self, x, repeats = 10):
        y = x.clone()
        nans = tc.isnan(x)
        notnans = ~nans
        y[nans] = self.init_nan
        y_history = []

        for i in range(repeats):
            y = self.layers(y)
            y[notnans] = x[notnans]
            y_history.append(y)
        if self.SHAP: return(y)
        else:
            return(y, y_history, nans)


class VariationalResNet(ResNet):
    def __init__(self, input_dim, width, sample_width, depth, variational, SHAP= False):
        super(ResNet,self).__init__()
        self.SHAP = SHAP
        self.variational = variational
        self.init_nan = nn.Parameter(torch.tensor(0.0, requires_grad=True))

        self.enc = nn.Sequential(
            *[ResBlock(input_dim, width, act_bool=False) for _ in range(depth)]
        )

        self.mean_layer = nn.Linear(input_dim, sample_width)
        self.logvar_layer = nn.Linear(input_dim, sample_width)

        self.dec = nn.Sequential(
            nn.Linear(sample_width, input_dim),
            ResBlock(input_dim, width, act_bool=False)
        )

    def sample(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, repeats = 10):
        y = x.clone()
        nans = tc.isnan(x)
        notnans = ~nans
        y[nans] = self.init_nan
        y_history = []

        for i in range(repeats):
            y = self.enc(y)
            mean_ = self.mean_layer(y)
            log_var_ = self.logvar_layer(y)
            latent = self.sample(mean_, log_var_)
            y = self.dec(latent)
            y[notnans] = x[notnans]
            y_history.append(y)

        if self.SHAP:
            return y,mean_, log_var_
        else:
            return(y, y_history, nans, mean_, log_var_)
