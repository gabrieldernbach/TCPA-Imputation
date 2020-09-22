import os
import torch
import torch as tc
import torch.nn as nn
import torch.autograd
from sklearn.manifold import TSNE
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sn
import random
import numpy as np
from torch.distributions.uniform import Uniform
import itertools
from torch.utils.data import DataLoader, Dataset


data_with_names = pd.read_csv('../data2/TCPA_data_sel.csv')
ID, data = data_with_names.iloc[:,:2], tc.tensor(data_with_names.iloc[:,2:].values)

from torch.utils.data import Dataset, DataLoader

class DataSet(Dataset):
    def __init__(self, data, prot_i,prot_j):
        self.data = data
        self.len = data.size(0)
        self.proteins = (prot_i, prot_j)

    def __getitem__(self, idx):
        false_idx = random.choice(list(range(idx)) + list(range(idx, self.len)))
        i = int(tc.bernoulli(tc.tensor([0.5])))

        correct_data = self.data[idx,self.proteins]
        false_data = tc.cat((self.data[idx, self.proteins[i]].unsqueeze(0), self.data[false_idx, self.proteins[1-i]].unsqueeze(0)), dim=0)

        return correct_data.float(), false_data.float()

    def __len__(self):
        return self.len


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return (F.sigmoid(self.layers(x)))

discriminator = Discriminator(2,20,1)
discriminator.cuda()
optim_discriminator = tc.optim.Adam(discriminator.parameters(), lr= 0.001)

results = tc.zeros(size=(data.shape[1], data.shape[1]))
for i in range(data.shape[1]-1):
    for j in range(i+1, data.shape[1]):
        dataset = DataSet(data, i,j)
        dataloader = DataLoader(dataset, batch_size = 500, shuffle = True, drop_last=True)
        discriminator.train()

        for epoch in range(1000):
            for correct_data, false_data in dataloader:
                correct_data = correct_data.cuda()
                false_data =false_data.cuda()

                optim_discriminator.zero_grad()

                pos_discrimination = discriminator(correct_data)
                neg_discrimination = discriminator(false_data)

                loss = -(pos_discrimination.mean(dim=0) - tc.log(tc.exp(neg_discrimination).mean(dim=0)))
                loss.backward()
                optim_discriminator.step()
                print(i,j, epoch, loss)

        results[i,j] = loss.detach().cpu()

print(results)
np.save('../results/MINE_results.npy', results)


