import torch as tc
device = tc.device('cpu')
import torch.nn as nn
import RecursiveNet
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
import numpy as np
import random


class ShapleySet(Dataset):
    def __init__(self, data, p, q):
        tc.manual_seed(0)
        data = data[tc.randperm(data.shape[0]), :]
        self.data = data[3000:, :].float()
        tc.manual_seed(random.randint(1,100000))
        self.nsamples, self.nfeatures = self.data.shape
        self.R = self.init_randomsample()
        self.p = p
        self.q = q
        self.Set, self.Setp = None, None
        self.getSets()

    def init_randomsample(self):
        tensor_list = [self.data[tc.randperm(self.nsamples), i] for i in range(self.nfeatures)]
        return tc.stack(tensor_list).t()

    def getSets(self):
        self.Set = tc.distributions.bernoulli.Bernoulli(tc.tensor([0.5] * self.nfeatures)).sample()
        #self.Set[self.q] = 0  # muss das sein?
        self.Setp = self.Set.clone()
        self.Set[self.p], self.Setp[self.p] = 0, 1

    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx):
        assert self.Set is not None, ('did not sample')
        target = self.data[idx, :]
        random_values = self.R[idx, :]

        masked_data = tc.where(self.Set == 1, target, random_values)
        masked_dataP = tc.where(self.Setp == 1, target, random_values)

        return target, masked_data, self.Set, masked_dataP, self.Setp


class Shapley():
    def __init__(self, data, net):
        self.data = data
        self.nsamples, self.nfeatures = data.shape
        self.net = net.eval()
        self.shapleyvalues = tc.zeros(self.nsamples, self.nfeatures)

    def calc_shapleypq(self, p, q, steps, device):
        self.shapleyset = ShapleySet(self.data, p, q)
        self.shapleyloader = DataLoader(self.shapleyset, batch_size=self.nsamples)
        self.net.to(device)
        meandiff = tc.zeros(1)
        criterion = F.mse_loss
        proceed = True
        convergencechecker = [1000,1] # random numbers
        for t in range(1, 1+steps):
            for target, masked_data, Set, masked_dataP, SetP in self.shapleyloader:
                target, masked_data, Set, masked_dataP, SetP = target.to(device), masked_data.to(device), Set.to(
                    device), masked_dataP.to(device), SetP.to(device)
                with tc.no_grad():
                    pred, predP = self.net(masked_data, Set), self.net(masked_dataP, SetP)
                loss, lossP = criterion(pred, target), criterion(predP, target)

                meanloss, meanlossP = loss.to(device), lossP.to(device)
                meandiff = (t - 1) / t * meandiff + 1 / t * (meanloss - meanlossP)
            convergencechecker.append(meandiff)
            if max(convergencechecker[-10:-1]) - min(convergencechecker[-10:-1]) < 0.0002:
                print('converged after', len(convergencechecker))
                print(p, q, meandiff)
                break
        self.shapleyvalues[p, q] = meandiff
        np.save('results/shapley_values', self.shapleyvalues.cpu().detach().numpy())
        np.savetxt('results/shapley_values', self.shapleyvalues.cpu().detach().numpy(),delimiter=',')

    def calc_shapleyAll(self, device, steps = 1000):
        for p in range(self.nfeatures):
            for q in range(self.nfeatures):
                self.calc_shapleypq(p, q, steps, device)

