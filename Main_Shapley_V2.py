import os
# implement plot
# implement one hot tumor into vae and cross.validate

import torch as tc
import pandas as pd
import model_Shapley_V2 as model
import shapley_Shapley_V1 as sh
import plots_Shapley_V1 as plots
import Data_Shapley_V2 as data_sh

#specify hyperparameters
DATAPATH = '../data2/TCPA_data_sel.csv'
RESULTPATH = 'results/results'
for folder in ('figures', 'log', 'trained_model'):
    if not os.path.exists(RESULTPATH + '/' + folder):
        os.makedirs(RESULTPATH + '/' + folder)
device = tc.device('cuda:0')

train_network = False
##################
calc_shapley = True
load_epoch, load_variational, load_k, load_lin = 1000, False, 1, 'nonlinear' #define model that shall be loaded for shapley
##################
plot = False

randomized_data = data_sh.get_data('four_groups')
print(randomized_data.shape)

protein_names = None

if train_network:
    for variational in [False]:
        for nonlinear in [True]:
            for k in [1]:
                #specify neural network
                vae = model.VAE(input_dim=randomized_data.size(1), width=randomized_data.size(1)*4, sample_width=randomized_data.size(1)*4, depth=3, variational = variational, nonlinear = True, k = 1)
                #init gibb sampler with neural network
                gibbs_sampler = model.GibbsSampler(neuralnet=vae, warm_up=4, convergence=0.0, result_path=RESULTPATH, device = device)
                #train and test model in n fold crossvalidation
                model.cross_validate(model=gibbs_sampler, data=randomized_data, path = RESULTPATH, train_epochs = 1001, lr = 0.0001, train_repeats = 3, ncrossval=1)



if calc_shapley:
    gibbs_sampler = tc.load(RESULTPATH + '/trained_model/Gibbs_sampler_trainepochs={}_var={}_k={}_{}.pt'.format(load_epoch, load_variational, load_k, load_lin)) # save and load always gibbs_sampler or model within?
    gibbs_sampler.device = device
    shapley = sh.Shapley(gibbs_sampler, data = randomized_data, protein_names= protein_names, device=device)
    shapley.calc_all(device=device, steps=100)


if plot:
    plots.plot(RESULTPATH)






