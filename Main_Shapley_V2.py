import os

# todo cross validation
# implement plot
# implement one hot tumor into vae and cross.validate

import torch as tc
import pandas as pd
import model_Shapley_V1 as model
import shapley_Shapley_V1 as sh
import plots_Shapley_V1 as plots
import Data_Shapley_V1 as data_sh

#specify hyperparameters
DATAPATH = '../data2/TCPA_data_sel.csv'
RESULTPATH = 'results/results'
for folder in ('figures', 'log', 'trained_model'):
    if not os.path.exists(RESULTPATH + '/' + folder):
        os.makedirs(RESULTPATH + '/' + folder)
device = tc.device('cuda:0')

train_network = True
calc_shapley = False
plot = False

randomized_data = data_sh.get_data('breast_cancer')

if train_network:
    for variational in [True]:
        for nonlinear in [True]:
            for k in [1]:
                #specify neural network
                vae = model.VAE(input_dim=randomized_data.size(1), width=randomized_data.size(1)*4, sample_width=randomized_data.size(1)*4, depth=3, variational = variational, nonlinear = True, k = 10)
                #init gibb sampler with neural network
                gibbs_sampler = model.GibbsSampler(neuralnet=vae, warm_up=4, convergence=0.0, result_path=RESULTPATH, device = device)
                #train and test model in n fold crossvalidation
                model.cross_validate(model=gibbs_sampler, data=randomized_data, path = RESULTPATH, train_epochs = 10001, lr = 0.00001, train_repeats = 3, ncrossval=1)



if calc_shapley:
    gibbs_sampler = tc.load(RESULTPATH + '/trained_model/Gibbs_sampler.pt') # save and load always gibbs_sampler or model within?
    gibbs_sampler.device = device
    shapley = sh.Shapley(gibbs_sampler, data, protein_names, device=device)
    shapley.calc_all(device=device, steps=10000)


if plot:
    plots.plot(RESULTPATH)






