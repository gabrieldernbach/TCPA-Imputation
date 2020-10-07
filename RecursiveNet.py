import os
import torch as tc
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from functions import plot_results
import torch.nn as nn
from Classes import bn_linear # ResBlock
import random

#device = tc.device('cpu') # tc.device('cuda' if tc.cuda.is_available() else 'cpu')

data_with_names = pd.read_csv('../data2/TCPA_data_sel.csv')
ID, data = data_with_names.iloc[:,:2], tc.tensor(data_with_names.iloc[:,2:].values)

class ProteinSet(Dataset):
    def __init__(self, data, p):
        self.data = data.float()
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
        target = self.data[idx, :]
        random_values = self.R[idx, :]
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
    train_dataset = ProteinSet(data[:3000,:], p = p_train)
    test_dataset = ProteinSet(data[3000:,:], p = p_test)
    trainloader = DataLoader(train_dataset, batch_size = 2000, shuffle = True)
    testloader = DataLoader(test_dataset, batch_size = 1000, shuffle = True)
    return trainloader, testloader

class ResBlock(nn.Module):
    def __init__(self, input_dim, width,  act_bool, activation=nn.ReLU()):
        super(ResBlock, self).__init__()

        self.alpha = nn.Parameter(tc.tensor(1.0, requires_grad=True)).to(device)
        self.input_dim, self.act_bool = input_dim, act_bool
        self.layers = nn.Sequential(
            bn_linear(input_dim, width),
            activation,
            nn.Dropout(p=0.1),
            bn_linear(width, input_dim),
            activation if act_bool else nn.Identity()
        )

    def forward(self,x):
        residual = x.clone() if self.act_bool else tc.zeros_like(x)
        #print(residual.device, self.alpha.device, next(self.layers.parameters()).device)
        return residual*self.alpha + self.layers(x)

class RecursiveNet(nn.Module):
    def __init__(self, input_dim, width, sample_width, depth, variational):
        super(RecursiveNet,self).__init__()
        self.variational = variational

        self.enc = nn.Sequential(
            *[ResBlock(input_dim, width, act_bool=False) for _ in range(depth)]
        )

        self.mean_layer = nn.Linear(input_dim, sample_width)
        self.logvar_layer = nn.Linear(input_dim, sample_width)

        self.dec = nn.Sequential(
            nn.Linear(sample_width, input_dim),
            *[ResBlock(input_dim, width, act_bool=False) for _ in range(depth)]
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
        self.recursivenet = RecursiveNet(input_dim, width, sample_width, depth, variational)

    def forward(self,x, Mask):
        y = x.clone() #only for my readability
        cumulative_y = []
        for i in range(self.repeats):
            y, mean_, log_var_ = self.recursivenet(y, Mask)
            cumulative_y.append(y)
        meany = tc.stack(cumulative_y[3:],dim=0).mean(dim=0)
        return meany


class Train_env:
    def __init__(self,data, load_model=True, device=tc.device('cpu')):
        self.data = data
        if os.path.isfile('recursivenet.pt') and load_model:
            print('recursivenet found, loading net')
            self.net = tc.load('recursivenet.pt')
        else: self.net = None
        self.result_shape = None
        self.device=device

    def train_network(self, width, sample_width, depth, variational,  train_dist, test_dist, lr , n_epochs=2, test_every=1, trainbool = True, repeats=5):
        self.variational = variational
        self.test_result = []
        self.repeats = repeats
        self.trainloader, self.testloader = get_dataloaders(self.data, train_dist, test_dist)

        if self.net is None:
            self.net = StackedNet(input_dim=self.data.shape[1], width=width, sample_width=sample_width, depth=depth, variational = self.variational, repeats = self.repeats)
        else:
            self.net.repeats = repeats
        for i in range(n_epochs):
            if i%test_every == 0:
                print(i)
                self.test_it()
                print(self.test_result[-1])
                #plot_results(self.test_result)
            if trainbool:
                self.train_it(lr)
                tc.save(self.net, 'recursivenet.pt')

        #self.test_result = tc.cat(self.test_result, dim = 1)
        #return self.test_result

    def train_it(self, lr):
        self.net.to(self.device).train()
        optimizer = tc.optim.Adam(self.net.parameters(), lr)
        criterion = F.mse_loss

        for i, (train_target, train_masked_data, Mask) in enumerate(self.trainloader):
            optimizer.zero_grad()
            train_target, train_masked_data, Mask = train_target.to(self.device), train_masked_data.to(self.device),  Mask.to(self.device)

            loss = 0
            for i in range(self.net.repeats):
                prediction, mean_, log_var_ = self.net.recursivenet(train_masked_data, Mask)
                train_masked_data = prediction

            kl = -0.5 * tc.mean(1 + log_var_ - mean_.pow(2) - log_var_.exp())
            loss += criterion(prediction[Mask == 0], train_target[Mask == 0]) + kl

            loss.backward()
            optimizer.step()

    def test_it(self):
        self.net.to(self.device).eval()
        criterion = F.mse_loss

        mse_losses=[]
        for i, (test_target, test_masked_data, Mask) in enumerate(self.testloader):
            test_target, test_masked_data, Mask = test_target.to(self.device), test_masked_data.to(self.device), Mask.to(self.device)

            repeat_losses = []
            for j in range(self.net.repeats):
                self.net.to(self.device)
                prediction, mean_, log_var_ = self.net.recursivenet(test_masked_data, Mask)
                repeat_losses.append(tc.tensor([criterion(prediction[Mask==0], test_target[Mask==0])]))
            mse_losses.append(tc.cat(repeat_losses, dim=0))

            mean_prediction = self.net(test_masked_data, Mask)
            mean_loss = criterion(mean_prediction[Mask==0], test_target[Mask==0])
            print(mean_loss)
        all_mse_losses = tc.stack(mse_losses,dim=0).mean(0)
        self.test_result.append(all_mse_losses.unsqueeze(1))

