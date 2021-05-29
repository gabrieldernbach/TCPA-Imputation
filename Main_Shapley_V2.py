import os
import sys
# implement plot
# implement one hot tumor into vae and cross.validate

import torch as tc
import pandas as pd
import model_Shapley_V2 as model
import minqshapley_Shapley_V1 as sh
import Data_Shapley_V2 as data_sh
import counterfactual_Shapley_V2 as cf
import singlesampleShapley_Shapley as sssh
import conditional_loss_Shapley_V1 as cond
import qtriangleshapley_Shapley_V1 as tri
import qshapley_Shapley_V1 as qsh
import hsic

datatype = sys.argv[1]
#specify hyperparameters
for folder in ('results','results/figures', 'results/log', 'results/trained_model', 'results/adjacency','results/data', 'results/shapley'+datatype, 'results/hsic'+ datatype, 'results/triangle', 'results/counter',
               'results/conditional_loss'):
    if not os.path.exists(folder):
        os.makedirs(folder)
device = tc.device('cuda:1')

train_network = True
calc_shapley = True
calc_hsic = True

load_epoch, load_variational, load_k, load_lin = 4000, True, 1, 'nonlinear' #define model that shall be loaded for shapley
##################

train_set, test_set, protein_names = data_sh.get_data(datatype)

print(train_set.shape)

print(protein_names)

if train_network:
    for variational in [True]:
        for nonlinear in [True]:
            for k in [1]:
                #specify neural network
                vae = model.VAE(input_dim=train_set.size(1), width=train_set.size(1)*32, sample_width=train_set.size(1)*256, depth=12, variational = variational, nonlinear = True, k = k)
                #vae = model.MlpMix(reps=10, n_samp=256, hidden=128, ins= train_set.size(1))
                #init gibb sampler with neural network
                gibbs_sampler = model.GibbsSampler(neuralnet=vae, warm_up=6, convergence=0.0, result_path='results', device = device)
                #train and test model in n fold crossvalidation
                model.cross_validate(model=gibbs_sampler, train_data=train_set, test_data = test_set, path = 'results', train_epochs = 4001, lr = 0.00001, train_repeats =10, batch_factor=1, ncrossval=1)
    print('training finished')





if calc_hsic:
    gibbs_sampler = tc.load('results/trained_model/Gibbs_sampler_trainepochs={}_var={}_k={}_{}.pt'.format(load_epoch, load_variational, load_k, load_lin)) # save and load always gibbs_sampler or model within?
    gibbs_sampler.device = device
    shapley = hsic.Shapley(gibbs_sampler, data = test_set, protein_names= protein_names, device=device, datatype = datatype)
    shapley.calc_all(device=device, steps=200)

if calc_shapley:
    gibbs_sampler = tc.load('results/trained_model/Gibbs_sampler_trainepochs={}_var={}_k={}_{}.pt'.format(load_epoch, load_variational, load_k, load_lin)) # save and load always gibbs_sampler or model within?
    gibbs_sampler.device = device
    shapley = sh.Shapley(gibbs_sampler, data = test_set, protein_names= protein_names, device=device, datatype = datatype)
    shapley.calc_all(device=device, steps=2001)