import torch as tc
device = tc.device('cpu')
import torch.nn as nn
import BoostNet
import RecursiveNet
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd


class ShapleySet(Dataset):
    def __init__(self, data):
        self.data = data
        self.nsamples, self.nfeatures = data.shape
        self.R = self.init_randomsample()
        self.i = 0
        self.P = None

    def init_randomsample(self):
        tensor_list = [self.data[tc.randperm(self.nsamples), i] for i in range(self.nfeatures)]
        return tc.stack(tensor_list).t()

    def getSets(self):
        Set = tc.distributions.bernoulli.Bernoulli(tc.tensor([0.5] * self.nfeatures)).sample()
        SetP = Set.clone()
        Set[self.P], SetP[self.P] = 0, 1
        return Set, SetP

    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx):
        Set, SetP = self.getSets()

        target = self.data[idx, :]
        random_values = self.R[idx, :]

        masked_data = tc.where(Set == 1, target, random_values)
        masked_dataP = tc.where(SetP == 1, target, random_values)

        if self.i % self.nsamples == 0:
            self.R = self.init_randomsample()
        self.i += 1

        return target.float(), masked_data.float(), Set.float(), masked_dataP.float(), SetP.float()


class Shapley():
    def __init__(self, data, net, threshold=0.0001):
        self.data = data
        self.nsamples, self.nfeatures = data.shape
        self.net = net
        self.shapleyset = ShapleySet(data)
        self.shapleyvalues = tc.zeros(self.nsamples, self.nfeatures)
        self.threshold = threshold

    def calc_shapleyP(self, P):
        self.shapleyset.P = P
        self.shapleyloader = DataLoader(self.shapleyset, batch_size= self.nsamples)
        t = 1
        meandiff = tc.zeros(self.nfeatures)
        criterion = F.mse_loss
        proceed = True

        while proceed:
            for target, masked_data, Set, masked_dataP, SetP in self.shapleyloader:
                target, masked_data, Set, masked_dataP, SetP = target.to(device), masked_data.to(device), Set.to(
                    device), masked_dataP.to(device), SetP.to(device)
                pred, predP = self.net(masked_data, Set), self.net(masked_dataP, SetP)
                with tc.no_grad():
                    loss, lossP = criterion(pred, target, reduction='none'), criterion(predP, target, reduction='none')

                meanloss, meanlossP = loss.mean(dim=0).to(device), lossP.mean(dim=0).to(device)
                meandiff = (t - 1) / t * meandiff + 1 / t * (meanloss - meanlossP)
                t += 1
                if t == 1000:
                    proceed = False
                    self.shapleyvalues[P, :] = meandiff

    def calc_shapleyAll(self):
        for P in range(self.nfeatures):
            print(P)
            self.calc_shapleyP(P)


