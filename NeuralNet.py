import os
import torch
import torch as tc
import torch.nn as nn
import torch.autograd

import matplotlib.pyplot as plt
import pandas as pd

from functions import *
from classes import *
from analysis import *

data_with_names = pd.read_csv('../data2/TCPA_data_sel.csv')
ID, data = data_with_names.iloc[:,:2], tc.tensor(data_with_names.iloc[:,2:].values)

[print(key, list(ID['Tumor']).count(key)) for key in ID['Tumor'].unique()]    #info


Bernoulli = [(0.1,0.9), (0.5,0.9)]
activations = [nn.ReLU()]
super_epochs = 10
widths =[(50,30,10)]
depths= [5]

analysis = Analysis(learn_Bernoulli=Bernoulli, super_epochs=super_epochs, activations=activations, widths=widths, depths=depths, repeats=10)
analysis.train_networks(test_bool=False, end_test_bool=True, epochs=1500, test_every=1)

#importance_measure = Importance_measure(analysis.saved_net, data, 'names')
#importance_measure.compute_importance(p=0.1,n_epochs = 10, repeats = 10)
#plt.plot(importance_measure.unimportance)


np.save('results/grid_search10.npy', analysis.results)
np.save('results/grid_search10.npy', {'results':analysis.results, 'Bernoulli':Bernoulli, 'widths': widths, 'depths': depths})







