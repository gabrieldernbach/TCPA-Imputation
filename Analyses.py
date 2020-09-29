import sys
import torch as tc
import torch.nn as nn
import torch.autograd
from Train_Env import *

import pandas as pd

print(os.getcwd())
data_with_names = pd.read_csv('../data2/TCPA_data_sel.csv')
ID, data = data_with_names.iloc[:,:2], tc.tensor(data_with_names.iloc[:,2:].values)
#[print(key, list(ID['Tumor']).count(key)) for key in ID['Tumor'].unique()]    #info

# todo: variational

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
sample_width = 30
depth= 4
variational = True
lr = 0.0001
test_repeats = 6

train_env = Train_env(data, load_model=False)  # specify network
train_env.train_network(width, sample_width, depth, variational, train_dist, test_dist, n_epochs=n_epochs, test_every=test_every, test_repeats = test_repeats) # specify training and test
train_env.data_shapley()







#train_env.grid_search(**grid_search_dict)
#train_env.importance_measure(p, n_epochs)





#np.save('results/grid_search11.npy', analysis.results)
#np.save('results/grid_search11.npy', {'results':analysis.results, 'Bernoulli':Bernoulli, 'widths': widths, 'depths': depths})
