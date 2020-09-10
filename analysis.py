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
from functions import *
from classes import *


data = pd.read_csv('../data2/TCPA_data_sel.csv')
ID, data = data.iloc[:,:2], tc.tensor(data.iloc[:,2:].values)

def loop(data, p, n_epochs, test, activation, width, depth, trainbool = True, test_every = 400, net = None, repeats=5):
    if net is None:
        net = ResNet(data.shape[1], width=width, depth=depth, activation=activation)
        #net = Net([data.shape[1], 300,500,300,data.shape[1]], [True,True,True,False], activation = activation, normalize = normalize)
    trainloader, testloader = get_dataloaders(data, p)
    lr = 0.001
    results = []
    for i in range(n_epochs):
        if i == 5: lr = 0.0001
        if i == 8: lr = 0.00002
        if i%test_every == 0 and test:
            results.append(test_it(net, testloader, repeats=repeats))
            plot_results(results)
        if trainbool:
            train_it(net, trainloader, lr)

        if i%test_every == 0:
            pass
    return (results, net)


class Analysis:
    def __init__(self, learn_Bernoulli, super_epochs, activations, widths, depths, repeats):
        self.training_epochs = 1
        self.net = None
        self.saved_net = None
        self.repeats = repeats
        self.acts, self.widths, self.depths = activations, widths, depths
        self.results = np.zeros(shape = (len(np.arange(0.1,1,0.1)), len(learn_Bernoulli), super_epochs, len(activations), len(widths),len(depths), repeats))
        self.learn_Bernoulli = learn_Bernoulli
        self.super_epochs = super_epochs

    def train_networks(self, epochs = 4000, test_bool = True, end_test_bool = True, test_every = 400):
        self.training_epochs = epochs

        for l, depth in enumerate(self.depths):
            for k, width in enumerate(self.widths):
                for j, activation in enumerate(self.acts):
                    for i in range(self.super_epochs):
                        print(i)
                        for h, p in enumerate(self.learn_Bernoulli):
                            plt.close()

                            _, self.net = loop(data, p, n_epochs=epochs, test=test_bool, activation=activation, width=width, depth=depth, trainbool=True,
                                               test_every = test_every, net= self.net, repeats= self.repeats)

                            self.saved_net = self.net
                            if end_test_bool:
                                print('end_test')
                                for z, p in enumerate(np.arange(0.1,1,0.1)):
                                    resultlist, _ = loop(data, p, n_epochs=1, test=True, activation=activation,
                                                   width=width, depth=depth,
                                                   test_every=1, trainbool=False, net = self.net, repeats=self.repeats)

                                    myresult = np.array(resultlist[0])
                                    self.results[z, h, i, j, k, l, :] = myresult



                    self.net = None


class Importance_measure:
    def __init__(self, net, data, names):
        self.data = data
        self.names = names
        self.net = net
        print(self.net)
        self.unimportance = tc.zeros(data.shape[1])


    def compute_importance(self, p, n_epochs, repeats):
        _, testloader = get_dataloaders(self.data, p)

        for i in range(n_epochs):
            loss, nans = importance_test(self.net, testloader, repeats=repeats)
            loss, nans = loss.cpu(), nans.cpu()
            print(tc.ones(nans.shape).shape, loss.repeat(nans.shape[1],1).permute(1,0).shape)
            unimportance = tc.ones(nans.shape) * loss.repeat(nans.shape[1],1).permute(1,0)
            unimportance[nans] = tc.tensor(0)

            self.unimportance += unimportance.mean(dim=0).squeeze().detach()

        self.unimportance /= self.unimportance.mean()





