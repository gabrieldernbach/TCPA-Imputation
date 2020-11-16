import os

# todo cross validation

#todo save model and test result
# implement linear model in vae
# implement importance weighted vae into vae
# implement non variational into vae
# implement the last 3 things into cross.validate so that it is saved correctly
# implement plot
# implement one hot tumor into vae and cross.validate


import torch as tc
import pandas as pd
import model_Shapley_V1 as model
import shapley_Shapley_V1 as sh

#specify hyperparameters
DATAPATH = '../data2/TCPA_data_sel.csv'
RESULTPATH = 'results/results' # in RESULTPATH there must be 3 folders: figures, log, trained_model
for folder in ('figures', 'log', 'trained_model'):
    if not os.path.exists(RESULTPATH + '/' + folder):
        os.makedirs(RESULTPATH + '/' + folder)
device = tc.device('cuda:0')

train_network = True
calc_shapley = False

#import data from PATH
data_with_names = pd.read_csv(DATAPATH)
ID, data = data_with_names.iloc[:,:2], tc.tensor(data_with_names.iloc[:,2:].values)
protein_names = data_with_names.columns[2:]
tc.manual_seed(0)
randomized_data = data[tc.randperm(data.shape[0]), :].float()

if train_network:
    #specify neural network
    vae = model.VAE(input_dim=data.size(1), width=512, sample_width=256, depth=3, variational = False, nonlinear = True, k = 1)

    #init gibb sampler with neural network
    gibbs_sampler = model.GibbsSampler(neuralnet=vae, warm_up=4, convergence=0.0, result_path=RESULTPATH, device = device)

    #train and test model in n fold crossvalidation
    model.cross_validate(model=gibbs_sampler, data=randomized_data, path = RESULTPATH, train_epochs = 1, lr = 0.001, train_repeats = 1, ncrossval=1)

if calc_shapley:
    gibbs_sampler = tc.load(RESULTPATH + '/trained_model/Gibbs_sampler.pt') # save and load always gibbs_sampler or model within?
    gibbs_sampler.device = device
    shapley = sh.Shapley(gibbs_sampler, data, protein_names, device=device)
    shapley.calc_all(device=device, steps=10000)






