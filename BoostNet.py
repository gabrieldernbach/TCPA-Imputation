import os
import torch as tc
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from functions import plot_results
import torch.nn as nn
from Classes import ResBlock
import random

data_with_names = pd.read_csv('../data2/TCPA_data_sel.csv')
ID, data = data_with_names.iloc[:,:2], tc.tensor(data_with_names.iloc[:,2:].values)

class ProteinSet(Dataset):
    def __init__(self, data, p):
        self.data = data
        self.nsamples, self.nfeatures = data.shape
        self.R = self.init_randomsample()

        if isinstance(p,tuple):
            self.p = random.uniform(p[0], p[1])
        else:
            self.p = p


    def init_randomsample(self):
        tensor_list = [self.data[tc.randperm(self.nsamples),i] for i in range(self.nfeatures)]
        return tc.stack(tensor_list).t()

    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx):
        Mask = tc.distributions.bernoulli.Bernoulli(tc.tensor([self.p] * self.nfeatures)).sample().float()
        target = self.data[idx, :].float()
        random_values = self.R[idx, :].float()
        masked_data = tc.where(Mask == 1.0, target, random_values)

        return target, masked_data, Mask

#proteinset = ProteinSet(data,0.5)
#proteinloader = DataLoader(proteinset, batch_size=2000)
#for epoch in range(5):
#    for i,_ in enumerate(proteinloader):
#        print(i)


def get_dataloaders(data, p_train, p_test):
    tc.manual_seed(0)
    data = data[tc.randperm(data.shape[0]),:]
    train_dataset = ProteinSet(data[:4000,:], p = p_train)
    test_dataset = ProteinSet(data[4000:,:], p = p_test)
    trainloader = DataLoader(train_dataset, batch_size = 2000, shuffle = True)
    testloader = DataLoader(test_dataset, batch_size = 1000, shuffle = True)
    return trainloader, testloader

#print(get_dataloaders(data,0.3,0.1))
class BoostNet(nn.Module):
    def __init__(self, input_dim, width, sample_width, depth, variational):
        super(BoostNet,self).__init__()
        self.variational = variational

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
        std = tc.exp(0.5 * log_var)
        eps = tc.randn_like(std)
        return mean + eps * std

    def forward(self, x, Mask):
        y = x.clone()

        y = self.enc(y)
        mean_ = self.mean_layer(y)
        log_var_ = self.logvar_layer(y)
        latent = self.sample(mean_, log_var_)
        y = self.dec(latent)

        y = tc.where(Mask==1, x, y)
        return y, mean_, log_var_

class StackedNet(nn.Module):
    def __init__(self, input_dim, width, sample_width, depth, variational, repeats):
        super(StackedNet,self).__init__()
        self.repeats = repeats
        self.boostnets = nn.ModuleList([BoostNet(input_dim, width, sample_width, depth, variational) for i in range(self.repeats)])

    def forward(self,x, Mask):
        y = x.clone() #only for my readability
        for i,boostnet in enumerate(self.boostnets):
            y, mean_, log_var_ = boostnet(y, Mask)
        return y



class Train_env:
    def __init__(self,data, load_model=True):
        self.data = data
        if os.path.isfile('Stacked_net.pt') and load_model:
            print('Stackednet found, loading net')
            self.net = torch.load('Stacked_net.pt')
        else: self.net = None
        self.result_shape = None

    def train_network(self, width, sample_width, depth, variational,  train_dist, test_dist, lr , n_epochs=2, test_every=1, trainbool = True, repeats=5):
        self.variational = variational
        self.test_result = []
        self.repeats = repeats
        self.trainloader, self.testloader = get_dataloaders(self.data, train_dist, test_dist)

        if self.net is None:
            self.net = StackedNet(input_dim=self.data.shape[1], width=width, sample_width=sample_width, depth=depth, variational = self.variational, repeats = self.repeats)

        for i in range(n_epochs):
            if i%test_every == 0:
                print(i)
                self.test_it()
                print(self.test_result)
                #plot_results(self.test_result)
            if trainbool:
                print('train',i)
                self.train_it(lr)
                tc.save(self.net, 'Boost_net.pt')

        self.test_result = tc.cat(self.test_result, dim = 1)
        return self.test_result

    def train_it(self, lr):
        device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
        self.net.to(device).train()
        optimizer = tc.optim.Adam(self.net.parameters(), lr)
        criterion = F.mse_loss

        for i, (train_target, train_masked_data, Mask) in enumerate(self.trainloader):
            optimizer.zero_grad()

            train_target, train_masked_data, Mask = train_target.to(device), train_masked_data.to(device),  Mask.to(device)
            print('train', Mask.shape)

            loss = 0
            for boostnet in self.net.boostnets:
                prediction, mean_, log_var_ = boostnet(train_masked_data, Mask)
                kl = -0.5 * tc.mean(1 + log_var_ - mean_.pow(2) - log_var_.exp())

                loss += criterion(prediction[Mask==0], train_target[Mask==0]) + kl
                train_masked_data = prediction

            loss.backward()
            optimizer.step()

    def test_it(self):
        device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
        self.net.to(device).eval()
        criterion = F.mse_loss
        mse_losses = []

        for i, (test_target, test_masked_data, Mask) in enumerate(self.testloader):
            test_target, test_masked_data, Mask = test_target.to(device), test_masked_data.to(device), Mask.to(device)
            for boostnet in self.net.boostnets:
                prediction, mean_, log_var_ = boostnet(test_masked_data, Mask)
                mse_losses.append(tc.tensor([criterion(prediction[Mask==0], test_target[Mask==0])]).unsqueeze(0))

        all_mse_losses = tc.cat(mse_losses, dim=0).mean(dim=0)
        self.test_result.append(all_mse_losses.unsqueeze(1))



train_dist = ((0.1,0.9))
test_dist = (0.50)
activations = [nn.ReLU()]
n_epochs = 1000
test_every = 10
width =512
sample_width = 512
depth= 3
variational = True
lr = 0.0001
repeats = 3

train_env = Train_env(data, load_model=False)  # specify network
train_env.train_network(width, sample_width, depth, variational, train_dist, test_dist, lr=lr, n_epochs=n_epochs, test_every=test_every, repeats = repeats) # specify training and test
