import pandas as pd
import sklearn
import torch as tc
from sklearn import datasets
from sklearn.utils import shuffle
import numpy as np
import torch.nn as nn

dim = 8
drop_out = 0.7
nsamples = 10000
tc.manual_seed(0)
mask = np.random.uniform(size=(dim, dim)) > drop_out
adjacency = mask * np.tril(np.ones_like(mask), -1) * 1.0

functions = [lambda x: tc.zeros_like(x),
             lambda x: tc.sin(x),
              lambda x:tc.cos(x),
              lambda x: x,
             lambda x: x**2]

class Node():
    #node gets a value and returns a vector with values for other nodes
    def __init__(self, dim, function_ids, id):
        # function_id = 0 means there is no edge between nodes
        self.id = id
        self.dim = dim
        self.function_ids = function_ids
        self.weights = tc.tensor(np.random.choice(np.concatenate((np.arange(-2,-0.5, 0.1), np.arange(0.5,2.0, 0.1))), size = self.dim))
    def propagate(self,x):
        results = [functions[self.function_ids[i]](self.weights[i]*x[:,self.id]) for i in range(self.dim)]
        tensor = tc.stack(results, dim=1).float()
        mean_ = tensor.mean(dim=1, keepdim=True)
        std_ = tensor.std(dim=1, keepdim=True) + 0.1
        norm_tensor = (tensor-mean_)/std_
        return norm_tensor

class graph_generator(nn.Module):
    def __init__(self, dim, adjacency, seed = 0):
        super(graph_generator, self).__init__()
        self.dim=dim
        self.adjacency = tc.tensor(adjacency, dtype = tc.int)
        tc.manual_seed(0)
        self.function_pos = tc.randint(1,len(functions), size= self.adjacency.shape, dtype = tc.int) * self.adjacency

        self.nodes = [Node(dim, self.function_pos[:,i], i) for i in range(self.dim)]

    def forward(self, noise):
        assert noise.size(1) == self.dim, 'noise wrong size'
        nodevalues = noise # implicitly this is a self loop with f(x) = x
        for i in range(self.dim):
            # loop through nodes
            nodevalues += 2 * self.nodes[i].propagate(nodevalues)
        return nodevalues




gg = graph_generator(dim, adjacency)
use_data = gg(tc.randn(nsamples,dim).float())
adjacency = gg.adjacency
function = gg.function_pos