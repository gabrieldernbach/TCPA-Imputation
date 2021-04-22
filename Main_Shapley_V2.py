import os
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

#specify hyperparameters
for folder in ('results','results/figures', 'results/log', 'results/trained_model', 'results/adjacency','results/data', 'results/shapley', 'results/triangle', 'results/counter',
               'results/conditional_loss'):
    if not os.path.exists(folder):
        os.makedirs(folder)
device = tc.device('cuda:0')

train_network = False
calc_shapley = True
calc_single_shapley = False
counterfactual = False
conditional = False
triangle=False
load_epoch, load_variational, load_k, load_lin = 500, False, 1, 'nonlinear' #define model that shall be loaded for shapley
##################

train_set, test_set, protein_names = data_sh.get_data('beeline')

print(train_set.shape)

print(protein_names)

if train_network:
    for variational in [False]:
        for nonlinear in [True]:
            for k in [1]:
                #specify neural network
                vae = model.VAE(input_dim=train_set.size(1), width=train_set.size(1)*16, sample_width=train_set.size(1)*16, depth=8, variational = variational, nonlinear = True, k = k)
                #init gibb sampler with neural network
                gibbs_sampler = model.GibbsSampler(neuralnet=vae, warm_up=6, convergence=0.0, result_path='results', device = device)
                #train and test model in n fold crossvalidation
                model.cross_validate(model=gibbs_sampler, train_data=train_set, test_data = test_set, path = 'results', train_epochs = 5, lr = 0.0001, train_repeats = 15, batch_factor=1, ncrossval=1)
    print('training finished')

if conditional:
    gibbs_sampler = tc.load('results/trained_model/Gibbs_sampler_trainepochs={}_var={}_k={}_{}.pt'.format(load_epoch, load_variational, load_k, load_lin)) # save and load always gibbs_sampler or model within?
    gibbs_sampler.device = device
    shapley = cond.Shapley(gibbs_sampler, data = test_set, protein_names= protein_names, device=device)
    shapley.calc_all(device=device)

if calc_shapley:
    gibbs_sampler = tc.load('results/trained_model/Gibbs_sampler_trainepochs={}_var={}_k={}_{}.pt'.format(load_epoch, load_variational, load_k, load_lin)) # save and load always gibbs_sampler or model within?
    gibbs_sampler.device = device
    shapley = sh.Shapley(gibbs_sampler, data = test_set, protein_names= protein_names, device=device)
    shapley.calc_all(device=device, steps=1)

if triangle:
    gibbs_sampler = tc.load('results/trained_model/Gibbs_sampler_trainepochs={}_var={}_k={}_{}.pt'.format(load_epoch, load_variational, load_k, load_lin)) # save and load always gibbs_sampler or model within?
    gibbs_sampler.device = device
    shapley = tri.Shapley(gibbs_sampler, data = test_set, protein_names= protein_names, device=device)
    shapley.calc_all(device=device, steps=5001)

if calc_single_shapley:
    gibbs_sampler = tc.load('results/trained_model/Gibbs_sampler_trainepochs={}_var={}_k={}_{}.pt'.format(load_epoch, load_variational, load_k, load_lin)) # save and load always gibbs_sampler or model within?
    gibbs_sampler.device = device
    shapley = sssh.Shapley(gibbs_sampler, data = test_set, protein_names= protein_names, device=device)
    #shapley.calc_shapleypqs(p=2,sample=0, steps = 100, device = device, probability = 0.5)
    shapley.calc_all(device=device, steps=300)

if counterfactual:
    gibbs_sampler = tc.load('results/trained_model/Gibbs_sampler_trainepochs={}_var={}_k={}_{}.pt'.format(load_epoch, load_variational, load_k, load_lin)) # save and load always gibbs_sampler or model within?
    gibbs_sampler.device = device
    counter = cf.Counter(gibbs_sampler, data = test_set, protein_names= protein_names, device=device)
    #counter.calc_counterfactualpq(5,0.1,3, device=device)
    counter.calc_all(device=device)


