import torch as tc
import torch.nn as nn
import RecursiveNet
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
import numpy as np
import random
from joblib import Parallel, delayed


class ShapleySet(Dataset):
    def __init__(self, data, p, probability):
        self.probability = probability
        tc.manual_seed(0)
        data = data[tc.randperm(data.shape[0]), :]
        self.data = data[3000:, :].float()
        tc.manual_seed(random.randint(1,100000))
        self.nsamples, self.nfeatures = self.data.shape
        self.R = self.init_randomsample()
        self.p = p
        self.Set, self.Setp = None, None
        self.getSets()

    def init_randomsample(self):
        tensor_list = [self.data[tc.randperm(self.nsamples), i] for i in range(self.nfeatures)]
        return tc.stack(tensor_list).t()

    def getSets(self):
        self.Set = tc.distributions.bernoulli.Bernoulli(tc.tensor([self.probability] * self.nfeatures)).sample()
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
    def __init__(self, data, protein_names, net):
        self.data = data
        self.nsamples, self.nfeatures = data.shape
        self.net = net.eval()
        self.shapleyvalues = tc.zeros(self.nfeatures, self.nfeatures)
        self.shapleylist = []
        self.protein_names = protein_names

    def calc_shapleypq(self, p, steps, device, probability):
        self.shapleyset = ShapleySet(self.data, p, probability)
        self.shapleyloader = DataLoader(self.shapleyset, batch_size=self.nsamples)
        self.net.to(device)
        meandiff = tc.zeros(1).to(device)
        criterion = F.mse_loss
        proceed = True
        convergencechecker = [0,1] # random numbers
        counter = tc.ones(self.nfeatures)
        for t in range(1, 1+steps):
            self.shapleyset.getSets()
            for target, masked_data, Set, masked_dataP, SetP in self.shapleyloader:
                target, masked_data, Set, masked_dataP, SetP = target.to(device), masked_data.to(device), Set.to(
                    device), masked_dataP.to(device), SetP.to(device)
                with tc.no_grad():
                    pred, predP = self.net(masked_data, Set), self.net(masked_dataP, SetP)

                loss, lossP = criterion(pred, target, reduction = 'none'), criterion(predP, target, reduction = 'none')
                #print((Set[0,:] == 0).shape) # use for specific shapley loss
                specific = (Set[0,:] == 0).float()
                meanloss, meanlossP = loss.mean(dim=0), lossP.mean(dim=0)
                meandiff = (counter - 1) / counter * meandiff + specific / counter * (meanloss - meanlossP)
                counter += specific
                convergencechecker.append(meandiff)
                #print(p, meandiff.shape)
                print(counter)
            if all(abs(convergencechecker[-2] - convergencechecker[-1])<0.0001):
                print(p, 'converged at', len(convergencechecker))
                break
        self.shapleyvalues[p, :] = meandiff
        pandasframe = pd.DataFrame(data = {'masked_protein': self.protein_names, 'shapley': meandiff.cpu().detach()})
        pandasframe.to_csv('results/shapley/batched_shapley_values_{}_{:.2f}_{}_specific.csv'.format(self.protein_names[p], probability, len(convergencechecker)-1), index=False)
        #np.savetxt('results/shapley/batched_shapley_values{}_{}.csv'.format(self.protein_names[p], probability),
        #       pandasframe, delimiter=',')

    def calc_shapleyAll(self, device, steps, probabilities):
        for probability in probabilities:
            Parallel(n_jobs=1)(delayed(self.calc_shapleypq)(p, steps, device, probability) for p in range(self.nfeatures))

        #np.save('results/shapley/shapley_values{}{}'.format(p), self.shapleylist.cpu().detach().numpy())
