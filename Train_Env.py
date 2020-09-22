import os
import torch
import torch as tc
import torch.nn as nn
import torch.autograd
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np
import math
import torch.nn.functional as F
from functions import get_dataloaders, plot_results
from Classes import *
from itertools import product
import shap

class Train_env:
    def __init__(self,data, load_model=True):
        self.data = data
        if os.path.isfile('recursive_net.pt') and load_model:
            print('net found')
            self.net = torch.load('recursive_net.pt')
            self.net.SHAP=False
        else: self.net = None
        self.grid_search_results = None
        self.result_shape = None
        self.grid_search_dict = None

    def train_network(self, width, depth, variational,  train_dist, test_dist, lr = 0.001, n_epochs=2, test_every=1, trainbool = True, test_repeats=5):
        self.variational = variational
        self.test_result = []
        self.test_repeats = test_repeats
        self.trainloader, self.testloader = get_dataloaders(self.data, train_dist, test_dist)

        if self.net is None:
            self.net = ResNet(self.data.shape[1], width=width, depth=depth, variational = self.variational) #if self.net is None else None#später self.os is none

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
        device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
        self.net.to(device).train()
        optimizer = tc.optim.Adam(self.net.parameters(), lr)
        criterion = F.mse_loss

        for i, (train_data, train_target) in enumerate(self.trainloader):
            optimizer.zero_grad()

            train_data = train_data.to(device)
            train_target = train_target.to(device)

            prediction,_, _= self.net(train_data)
            loss = criterion(prediction, train_target)
            loss.backward()
            optimizer.step()

    def test_it(self):
        device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
        self.net.to(device).eval()
        criterion = F.mse_loss
        mse_losses = []

        for i, (test_data, test_target) in enumerate(self.testloader):
            test_data = test_data.to(device)
            test_target = test_target.to(device)

            prediction, pred_history, nans = self.net(test_data, repeats = self.test_repeats)
            print(nans.shape, nans.sum())
            loss = criterion(prediction, test_target, reduction = 'none').mean(dim=1)
            mse_losses.append(tc.tensor([criterion(pred[nans], test_target[nans]) for pred in pred_history]).unsqueeze(0))

        all_mse_losses = tc.cat(mse_losses, dim=0).mean(dim=0)
        #print('all_mse_result:', all_mse_losses)
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

    def SHAP_values(self):
        device = tc.device('cpu')
        self.net.cpu().eval()
        self.net.SHAP=True
        _,testloader = get_dataloaders(self.data, 0.9, (0.1,0.9))
        batch = next(iter(testloader))
        data, target = batch
        background = target[:500]  # masked or unmasked data?

        test_data = tc.ones(self.data.shape).float()
        #test_data = F.one_hot(tc.tensor([1]),num_classes = self.data.shape[1]).float()
        #test_data = data[200:202]
        e = shap.GradientExplainer(self.net, background)
        shap_values = e.shap_values(test_data)
        tensor = tc.tensor(shap_values).squeeze()
        print('shap_values', shap_values)
        np.save('results/SHAP.npy',np.array(tensor))

        return np.array(tensor)










        




