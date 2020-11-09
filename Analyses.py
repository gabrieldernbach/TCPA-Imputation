import os
import sys
import torch as tc
import torch.nn as nn
import torch.autograd
#import BoostNet as BN
import RecursiveNet as RN
import DataShapley as DS
import numpy as np
import DataShapley_allq as DSq
import DataShapley_allq_specific as DSs
import DataShapley_ProteinConcentration as DSpc

import pandas as pd

data_with_names = pd.read_csv('../data2/TCPA_data_sel.csv')
ID, data = data_with_names.iloc[:,:2], tc.tensor(data_with_names.iloc[:,2:].values)
#[print(key, list(ID['Tumor']).count(key)) for key in ID['Tumor'].unique()]    #info
protein_names = data_with_names.columns[2:]

mode = "DataShapley_ProteinConcentration"
#mode = "RecursiveNet"
device = tc.device('cuda:0')


if mode == "Variational":
    grid_search_dict = {
        'width':[20],
        'depth': [4,2,3],
        'variational': [True],
        'train_dist': [(0.1,0.9)],
        'test_dist': [0.2]
    }
    train_dist = ((0.1,0.9))
    test_dist = (0.50)
    activations = [nn.ReLU()]
    n_epochs = 3000
    test_every = 30
    width =512
    sample_width = 1000
    depth= 4
    variational = True
    lr = 0.0001
    test_repeats = 6

    train_env = TE.Train_env(data, load_model=False)  # specify network
    train_env.train_network(width, sample_width, depth, variational, train_dist, test_dist, n_epochs=n_epochs, test_every=test_every, test_repeats = test_repeats) # specify training and test


elif mode == "Boost":

    train_dist = ((0.1, 0.9))
    test_dist = (0.50)
    activations = [nn.ReLU()]
    n_epochs = 1000
    test_every = 50
    width = 1024
    sample_width = 5000
    depth = 5
    variational = True
    lr = 0.0001
    repeats = 10

    train_env = BN.Train_env(data, load_model=False)  # specify network
    train_env.train_network(width, sample_width, depth, variational, train_dist, test_dist, lr=lr, n_epochs=n_epochs,
                            test_every=test_every, repeats=repeats)  # specify training and test


elif mode == 'RecursiveNet': # similar to Variational

    train_dist = ((0.1, 0.9))
    test_dist = (0.50)
    activations = [nn.ReLU()]
    n_epochs = 1
    test_every = 200
    width = 5
    sample_width = 5
    depth = 2
    variational = True
    lr = 0.001
    repeats = 4

    train_env = RN.Train_env(data, load_model=False, device=device)  # specify network
    train_env.train_network(width, sample_width, depth, variational, train_dist, test_dist, lr=lr, n_epochs=n_epochs,
                            test_every=test_every, repeats=repeats)  # specify training and test

elif mode == 'DataShapley':

    # make training environment
    train_env = RN.Train_env(data, load_model=True, device=device)  # use either BoostNet or RecursiveNet
    # specify and train network
    repeats = 4

    shapley = DS.Shapley(data, protein_names, train_env.net)
    shapley.calc_shapleyAll(steps=100, device=device)
    #np.save('results/shapley_values', shapley.shapleyvalues.cpu().detach().numpy())



elif mode == 'DataShapley_allq':

    # make training environment
    train_env = RN.Train_env(data, load_model=True, device=device)  # use either BoostNet or RecursiveNet
    # specify and train network
    repeats = 4

    shapley = DSq.Shapley(data, protein_names, train_env.net)
    shapley.calc_shapleyAll(steps=4000, device=device, probabilities = np.arange(0.1,1,0.1))


elif mode == 'DataShapley_allq_specific':

    # make training environment
    train_env = RN.Train_env(data, load_model=True, device=device)  # use either BoostNet or RecursiveNet
    # specify and train network
    repeats = 4

    shapley = DSs.Shapley(data, protein_names, train_env.net)
    shapley.calc_shapleyAll(steps=4000, device=device, probabilities = np.arange(0.1,1,0.1))


elif mode == 'DataShapley_ProteinConcentration':

    # make training environment
    train_env = RN.Train_env(data, load_model=True, device=device)  # use either BoostNet or RecursiveNet
    # specify and train network
    repeats = 4

    shapley = DSpc.Shapley(data, protein_names, train_env.net)
    shapley.calc_shapleyAll(steps=4, device=device, probabilities = np.arange(0.1,1,0.1))