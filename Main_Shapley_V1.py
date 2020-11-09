# todo cross validation

import os
import sys
import torch as tc
import pandas as pd
import model_Shapley_V1 as model
import shapley_Shapley_V1 as shapley

#specify hyperparameters
DATAPATH = '../data2/TCPA_data_sel.csv'
RESULTPATH = 'results/results'
device = tc.device('cuda:0')

train_network = True
Shapley = True

#import data from PATH
data_with_names = pd.read_csv(DATAPATH)
ID, data = data_with_names.iloc[:,:2], tc.tensor(data_with_names.iloc[:,2:].values)
protein_names = data_with_names.columns[2:]

if train_network:
    #specify neural network
    vae = model.VAE(input_dim=data.size(1), width=512, sample_width=256, depth=3)

    #init gibb sampler with neural network
    gibbs_sampler = model.GibbsSampler(neuralnet=vae, warm_up=4, convergence=0.01, result_path=RESULTPATH, device = device)

    #train and test model in n fold crossvalidation
    model.cross_validate(model=gibbs_sampler, data=data, path = RESULTPATH, ncrossval=1, )

if Shapley:




