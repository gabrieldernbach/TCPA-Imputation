import os
import torch
import torch as tc
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import random
import numpy as np
import math


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

        mask = tc.bernoulli(tc.ones(self.tensor_size)*q)==0
        masked_data = target.clone()
        masked_data[mask]= float('nan')
        return masked_data.float(), target.float()

def bn_linear(input_dim, output_dim):
    return nn.Sequential(
        nn.BatchNorm1d(input_dim),
        nn.Linear(input_dim, output_dim, bias=False)
    )

######################################################################
class ResBlock(nn.Module):
    def __init__(self, input_dim, width, activation, act_bool):
        super(ResBlock, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.input_dim, self.act_bool = input_dim, act_bool
        self.layers1 = nn.Sequential(
            bn_linear(input_dim, width[0]),
            activation,
            nn.Dropout(p=0.2),
            bn_linear(width[0], input_dim),
            activation if act_bool else nn.Identity()
        )
        self.layers2 = nn.Sequential(
            bn_linear(input_dim, width[1]),
            activation,
            nn.Dropout(p=0.2),
            bn_linear(width[1], input_dim),
            activation if act_bool else nn.Identity()
        )
        self.layers3 = nn.Sequential(
            bn_linear(input_dim, width[2]),
            activation,
            nn.Dropout(p=0.2),
            bn_linear(width[2], input_dim),
            activation if act_bool else nn.Identity()
        )
    def forward(self,x):
        residual = x.clone() if self.act_bool else tc.zeros(x.shape).cuda()
        return residual*self.alpha + self.layers1(x) + self.layers2(x)+ self.layers3(x)


class ResNet(nn.Module):
    def __init__(self, input_dim, width, depth, activation):
        super(ResNet,self).__init__()
        self.init_nan = nn.Parameter(torch.tensor(0.0, requires_grad=True))
        self.layers = nn.Sequential(
            *[ResBlock(input_dim, width, activation, act_bool=True) for _ in range(depth)],
            ResBlock(input_dim, width, activation, act_bool= False)

        )

    def forward(self, x, repeats = 10):
        y = x.clone()
        nans = tc.isnan(x)
        notnans = ~nans
        y[nans] = self.init_nan
        y_history = []

        for i in range(repeats):
            y = self.layers(y)
            y_history.append(y)
            y[notnans] = x[notnans]

        return(y, y_history, nans)



#####################################################################
class block(nn.Module):
    def __init__(self, input_dim, output_dim, relu, activation, normalize):
        super(block, self).__init__()
        self.relu = relu
        self.layers = nn.Sequential(
            bn_linear(input_dim, output_dim) if normalize else nn.Linear(input_dim, output_dim),
            activation
        ) if relu else nn.Linear(input_dim, output_dim)

    def forward(self,x):
        return self.layers(x)

class Net(nn.Module):
    def __init__(self, layer_sizes, relu, activation = nn.ReLU, normalize = False):
        super(Net,self).__init__()

        self.layers = nn.Sequential(
            *[block(inp, outp, relu[i], activation, normalize) for i, (inp,outp) in enumerate(zip(layer_sizes, layer_sizes[1:]))]
        )
    def forward(self, x, repeats = 10, init_nan= 0):
        y = x.clone()
        nans = tc.isnan(x)
        notnans = ~nans
        y[nans] = init_nan

        y_history = []
        for i in range(repeats):
            y = self.layers(y)
            y_history.append(y)
            y[notnans] = x[notnans]

        return(y, y_history, nans)



#################################################################################################


