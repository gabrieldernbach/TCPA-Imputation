import os
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import random
import pandas as pd
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model  import LinearRegression as LinReg

#todo: mask timebatch
class AR():
    def __init__(self, vector_size, p=1):
        self.vector_size = vector_size
        tc.manual_seed(0)
        self.init_vector = tc.rand(vector_size)
        self.diagonal = tc.eye(vector_size) * tc.rand(vector_size)
        self.eigenvectors = tc.tensor(linalg.orth(tc.rand(vector_size, vector_size) + 5 * tc.eye(vector_size)))
        self.function = tc.mm(tc.mm(self.eigenvectors, self.diagonal), tc.inverse(self.eigenvectors))
        self.time_series = [(self.init_vector)]
        self.values = None
        print(self.function)
    def step(self,n):
        for i in range(n):
            next_value = tc.matmul(self.function, self.time_series[-1]) + 0.1*tc.randn(self.vector_size)
            self.time_series.append(next_value)
        self.values = tc.stack(self.time_series, dim=0)


class Timedataset(Dataset):
    def __init__(self, vector_size, n, warm_up = 100):
        self.n = n
        self.warm_up = warm_up
        self.vector_size = vector_size
        ar = AR(vector_size)
        ar.step(n)

        self.datalist = ar.values[warm_up:,:]
        self.shuffled_data = self.datalist[tc.randperm(self.datalist.shape[0]), :]
        self.randomlist = tc.stack([self.datalist[tc.randperm(self.datalist.shape[0]),i] for i in range(self.datalist.shape[1])]).t()

    def __len__(self):
        return self.n-self.warm_up+1

    def __getitem__(self, idx):
        Mask = tc.distributions.bernoulli.Bernoulli(tc.tensor([0.2] * self.vector_size)).sample().float()

        target = self.datalist[idx,:]
        masked_value = self.randomlist[idx,:]
        masked = tc.where(Mask==1, target, masked_value)
        return target, masked


#############
def bn_linear(input_dim: int, output_dim: int):
    return nn.Sequential(
        nn.BatchNorm1d(input_dim),
        nn.Linear(input_dim, output_dim, bias=False)
    )

#class ResBlock as smallest Unit
class ResBlock(nn.Module):
    def __init__(self, input_dim, width,  act_bool, activation=nn.LeakyReLU()):
        super(ResBlock, self).__init__()

        self.alpha = nn.Parameter(tc.tensor(1.0, requires_grad=True)) # delete?
        self.input_dim, self.act_bool = input_dim, act_bool
        self.layers = nn.Sequential(
            bn_linear(input_dim, width),
            activation,
            nn.Dropout(p=0.1),
            bn_linear(width, input_dim),
            activation if act_bool else nn.Identity()
        )

    def forward(self,x):
        assert tc.is_floating_point(x), 'input is not float'
        residual = x.clone()
        return residual*self.alpha + self.layers(x)

class VAE(nn.Module):
    def __init__(self, input_dim, width, sample_width, depth, variational=True, nonlinear = True):
        super(VAE,self).__init__()
        #specify if Autoencoder is variational, linear or nonlinear and if IWAE is applied with k iterations
        self.variational = variational
        self.lin = 'nonlinear' if nonlinear else 'linear'
        self.activation = nn.LeakyReLU() if nonlinear else nn.Identity()

        self.encoder = nn.Sequential(
            *[ResBlock(input_dim, width, act_bool=False, activation = self.activation) for _ in range(depth)]
        )
        self.mean_layer, self.logvar_layer = nn.Linear(input_dim, sample_width), nn.Linear(input_dim, sample_width)

        self.decoder = nn.Sequential(
            nn.Linear(sample_width, input_dim),
            *[ResBlock(input_dim, width, act_bool=False) for _ in range(depth)]
        )

    def sample(self, mean, log_var):
        std = tc.exp(0.5 * log_var)
        eps = tc.randn_like(std)
        results = mean + eps*std
        return results

    def forward(self, x):
        encoding = self.encoder(x)

        #compute mean and log variance
        mean_l, log_var_l = self.mean_layer(encoding), self.logvar_layer(encoding)

        if self.variational:
            latent = self.sample(mean_l, log_var_l)
        else:
            latent = mean_l

        reconstruction = self.decoder(latent)
        # return mean und log variance for kl loss in VAE framework
        return reconstruction, mean_l, log_var_l

def cp_kl_loss(mean_, log_var):
    return -0.5 * tc.mean(1 + log_var - mean_.pow(2) - log_var.exp())

class GAN(nn.Module):
    def __init__(self, input_dim):
        super(GAN,self).__init__()
        self.layers = nn.Sequential(
            ResBlock(input_dim=input_dim, width=5*input_dim, act_bool = True),
            ResBlock(input_dim=input_dim, width=5 * input_dim, act_bool=True),
            nn.Linear(input_dim, 1)
        )

    def forward(self,x):
        return self.layers(x)


def trainVAE(gan, model1, model2, trainloader, lr):
    model1.train(), model2.train()
    optimizer1 = tc.optim.Adam(model1.parameters(), lr=lr)
    optimizer2 = tc.optim.Adam(model2.parameters(), lr=lr)

    optimizer1.zero_grad(), optimizer2.zero_grad()
    criterion = F.mse_loss

    for ground_truth, masked in trainloader:
        if ground_truth.shape[0] <2:
            continue
        time_data_batch1,meant1, vart1 = model1(masked)
        time_data_batch2, meant2, vart2 = model2(masked)

        reconstruction1, meanr1, varr1 = model2(time_data_batch1)
        reconstruction2, meanr2, varr2 = model1(time_data_batch2)

        # compute KL-divergence and MSE Loss
        GANloss = - (gan(time_data_batch1) + gan(time_data_batch2))
        kl_loss = cp_kl_loss(meant1, vart1) + cp_kl_loss(meant2, vart2) + cp_kl_loss(meanr1, varr1) + cp_kl_loss(meanr2, varr2)
        MSEloss = criterion(reconstruction1, ground_truth) + criterion(reconstruction2, ground_truth) -\
                  tc.min(tc.tensor(20.0), criterion(time_data_batch1, ground_truth) + criterion(time_data_batch2, ground_truth))
        loss = GANloss.mean() + kl_loss + MSEloss
        loss.backward()
        optimizer1.step()
        optimizer2.step()
    print("VAE:", "GAN", GANloss.mean().item(), "kl_loss", kl_loss.item(), "MSEloss", MSEloss.item())

def trainGAN(gan, model1, model2, trainloader, lr):
    model1.train(), model2.train(), gan.train()
    optimizer = tc.optim.Adam(gan.parameters(), lr=lr)

    optimizer.zero_grad()

    for ground_truth, masked in trainloader:
        if ground_truth.shape[0] <2:
            continue
        with tc.no_grad():
            time_data_batch1, meant1, vart1 = model1(masked)
            time_data_batch2, meant2, vart2 = model2(masked)

        correct = gan(ground_truth)
        false1 = gan(time_data_batch1)
        false2 = gan(time_data_batch2)

        # compute KL-divergence and MSE Loss
        loss = tc.max(false1.mean(), tc.zeros_like(false1.mean())) + tc.max(false2.mean(), tc.zeros_like(false1.mean())) - 2 * tc.min(correct.mean(), tc.zeros_like(correct.mean()))
        loss.backward()
        optimizer.step()
    print("loss", loss)

def test(model1, model2, testloader):
    model1.eval() , model2.eval()
    for ground_truth, masked in testloader:
        with tc.no_grad():
            time_data1,_,_ = model1(ground_truth)
            time_data2,_,_ = model2(ground_truth)

            model1, model2 = LinReg(), LinReg()
            model1.fit(ground_truth.numpy(), time_data1.cpu().numpy())
            model2.fit(ground_truth.numpy(), time_data2.cpu().numpy())

            print(np.corrcoef(model1.coef_.ravel(), model2.coef_.ravel()))
            print(model1.coef_)
            break






