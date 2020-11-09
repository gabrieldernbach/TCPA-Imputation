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
from collections import deque
import torch.autograd.functional.jacobian as jacobian

def get_jacobian(net, x, noutputs):
    x = x.squeeze()
    n = x.size()[0]
    x = x.repeat(noutputs, 1)
    x.requires_grad_(True)
    y = net(x)
    y.backward(torch.eye(noutputs))
    return x.grad.data

class RunningVariance:
    def __init__(self):
        self.n = 0
        self.ss = 0
        self.variance = 0
        self.mu_old = 0
        self.mu = None

    def add_value(self, x):
        x.cpu().detach().numpy()
        self.n += 1
        self.mu = (self.n - 1) / self.n * self.mu_old + (1 / self.n) * x
        self.ss = self.ss + (x - self.mu_old) * (x - self.mu)
        self.mu_old = self.mu
        self.variance = self.ss / self.n

class ShapleySet(Dataset):
    def __init__(self, data, probability):
        self.probability = probability
        tc.manual_seed(0)
        data = data[tc.randperm(data.shape[0]), :]
        self.data = data[3000:, :].float()
        tc.manual_seed(random.randint(1,100000))
        self.nsamples, self.nfeatures = self.data.shape
        self.R = self.init_randomsample()
        self.Set = None
        self.getSets()

    def init_randomsample(self):
        tensor_list = [self.data[tc.randperm(self.nsamples), i] for i in range(self.nfeatures)]
        return tc.stack(tensor_list).t()

    def getSets(self):
        self.Set = tc.distributions.bernoulli.Bernoulli(tc.tensor([self.probability] * self.nfeatures)).sample()

    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx):
        assert self.Set is not None, ('did not sample')
        target = self.data[idx, :]
        random_values = self.R[idx, :]

        masked_data = tc.where(self.Set == 1, target, random_values)

        return target, masked_data, self.Set #, masked_dataP, self.Setp

class Shapley():
    def __init__(self, data, protein_names, net):
        self.data = data
        self.nsamples, self.nfeatures = data.shape
        self.net = net.eval()
        self.jacobian = tc.zeros(self.nfeatures, self.nfeatures)
        self.shapleylist = []
        self.protein_names = protein_names

    def calc_shapleypq(self, q, steps, device, probability):

        self.shapleyset = ShapleySet(self.data, probability)
        self.shapleyloader = DataLoader(self.shapleyset, batch_size=self.nsamples)
        self.net.to(device)
        meangrad = tc.zeros(self.nfeatures).to(device)
        running_variance = RunningVariance()
        criterion = F.mse_loss
        proceed = True
        convergencechecker = deque([0,1,2,3,4,5,6,7,8,9,10]) # random numbers
        convergencecounter = 0
        counter = tc.ones(self.nfeatures).to(device)
        for t in range(1, 1+steps):
            self.shapleyset.getSets()
            self.net.zero_grad()
            prediction_sum = tc.zeros(1730, self.nfeatures).to(device)
            for i in range(1):
                for target, masked_data, Set in self.shapleyloader:
                    target, masked_data, Set = target.to(device), masked_data.to(device), Set.to(device)
                    #compute gradient with respect to input
                    masked_data = masked_data.requires_grad_()
                    pred = self.net(masked_data, Set)
                    prediction_sum += pred
            prediction_sum[:,q].sum(dim=0).backward()
            gradient = masked_data.grad.mean(dim=0)
            specific = Set.mean(dim=0)
            meangrad = (counter - 1) / counter * meangrad + specific / counter * (gradient)
            running_variance.add_value(gradient)
            counter += specific
            convergencechecker.append(meangrad)
            convergencechecker.popleft()
            convergencecounter += 1
            print(q, meangrad.mean())
            if all(abs(convergencechecker[-10] - convergencechecker[-1])<0.0001):
                print(q, 'converged at', convergencecounter)
                break
        pandasframe = pd.DataFrame(data = {'predicting_protein': self.protein_names, 'gradient': meangrad.cpu().detach(), 'variance': running_variance.variance.detach().cpu()})
        pandasframe.to_csv('results/gradient/batched_shapley_values_{}_{:.2f}_{}_gradient.csv'.format(self.protein_names[q], probability, convergencecounter), index=False)


    def calc_shapleyAll(self, device, steps, probabilities):
        for probability in probabilities:
            Parallel(n_jobs=1)(delayed(self.calc_shapleypq)(q, steps, device, probability) for q in range(self.nfeatures))

        #np.save('results/shapley/shapley_values{}{}'.format(p), self.shapleylist.cpu().detach().numpy())
