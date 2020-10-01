import os
import torch
import torch as tc
import torch.autograd
import numpy as np
import torch.nn.functional as F
from functions import get_dataloaders, plot_results
from Classes import VariationalResNet, Mydataset
from itertools import product
import pdb

class Train_env:
    def __init__(self,data, load_model=True, device = tc.device('cpu')):
        self.data = data
        self.device = device
        if os.path.isfile('recursive_net.pt') and load_model:
            print('net found, loading net')
            self.net = torch.load('recursive_net.pt')
            self.net.SHAP=False
        else: self.net = None
        self.grid_search_results = None
        self.result_shape = None
        self.grid_search_dict = None

    def train_network(self, width, sample_width, depth, variational,  train_dist, test_dist, lr = 0.001, n_epochs=2, test_every=1, trainbool = True, repeats=5):
        self.variational = variational
        self.test_result = []
        self.repeats = repeats
        self.trainloader, self.testloader = get_dataloaders(self.data, train_dist, test_dist)

        if self.net is None:
            self.net = VariationalResNet(self.data.shape[1], width=width, sample_width=sample_width, depth=depth, variational = self.variational)

        for i in range(n_epochs):
            if i%test_every == 0:
                print(i)
                self.test_it()
                print(self.test_result)
                plot_results(self.test_result)
            if trainbool:
                self.train_it(lr)
                torch.save(self.net, 'recursive_net.pt')

        self.test_result = tc.cat(self.test_result, dim = 1)
        return self.test_result

    def train_it(self, lr):
        self.net.to(self.device).train()
        optimizer = tc.optim.Adam(self.net.parameters(), lr)
        criterion = F.mse_loss

        for i, (train_data, train_target) in enumerate(self.trainloader):
            optimizer.zero_grad()

            train_data = train_data.to(self.device)
            train_target = train_target.to(self.device)

            prediction,_, _, mean_, log_var_ = self.net(train_data)

            if self.variational:
                kl = -0.5 * torch.mean(1 + log_var_ - mean_.pow(2) - log_var_.exp())

            else: kl=0
            loss = criterion(prediction, train_target) + kl
            loss.backward()
            optimizer.step()

    def test_it(self):
        self.net.to(self.device).eval()
        criterion = F.mse_loss
        mse_losses = []

        for i, (test_data, test_target) in enumerate(self.testloader):
            test_data = test_data.to(self.device)
            test_target = test_target.to(self.device)

            prediction, pred_history, nans,_,_ = self.net(test_data, repeats = self.repeats)
            print(nans.shape, nans.sum())
            loss = criterion(prediction, test_target, reduction = 'none').mean(dim=1)

            print(loss)
            mse_losses.append(tc.tensor([criterion(pred[nans], test_target[nans]) for pred in pred_history]).unsqueeze(0))

        all_mse_losses = tc.cat(mse_losses, dim=0).mean(dim=0)
        self.test_result.append(all_mse_losses.unsqueeze(1))

    def grid_search(self, **kwargs):
        func = self.train_network
        self.grid_search_dict = kwargs

        #grid_shape = (np.array([len(kwargs[key]) for key in kwargs]))

        r = (([[(key, i) for i in value] for key, value in kwargs.items()]))
        r = list(product(*reversed(r)))

        results = [func(**dict(i)).unsqueeze(2) for i in r]
        #print(results[0])
        grid_shape = np.array([*list(results[0].shape[:-1]), *[len(kwargs[key]) for key in kwargs]])

        print(results)
        print(tc.cat(results, dim=2).shape)
        results= np.array(tc.cat(results, dim = 2)).flatten().reshape(grid_shape, order='F')
        print('results:', results)
        print(results.shape)
        self.grid_search_results= results














        




