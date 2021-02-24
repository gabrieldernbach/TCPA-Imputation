import numpy
import numpy as np
import scipy.stats
from scipy.linalg import block_diag
from sklearn.datasets import make_spd_matrix
import torch as tc
import torch.nn as nn
import scipy.linalg as linalg
from sklearn.datasets import make_spd_matrix
# call by python propagate_one_network.py 1
# ------------------------------------------------------
# set connectivity
# ------------------------------------------------------
# construct random graph
mode = 'full'
depth = 3

nfeatures = 16
nsamples = 1000  # this is now the number of samples for train and test_set, so in sum 2*nsamples
connectivity = 1.0
block_size = (4, 4)
correlated = False
asymmetric = False
path = False

matrix_size = (nfeatures, nfeatures)

def block_generator(connectivity, block_size):
    x = np.random.mtrand.RandomState(0).binomial(size=block_size[0] * block_size[1], n=1, p=connectivity).reshape(
        block_size)
    return x * x.transpose()  # symmetrize --> leads to sparsity**2


def specific_network(subgraph):
    blocks = [block_generator(connectivity, block_size) if i in subgraph else block_generator(0, block_size) for i in
              range(matrix_size[0] // block_size[0])]
    block_matrix = np.array(block_diag(*blocks))
    np.fill_diagonal(block_matrix, 1)
    return block_matrix


############################
sparse_graph = specific_network([0, 1, 2, 3])
full_graph = sparse_graph * 0 + 1

#################################################################
# ARMA model
class AR():
    def __init__(self, vector_size, graph):
        self.vector_size = vector_size
        tc.manual_seed(0)
        self.init_vector = tc.rand(vector_size)
        self.diagonal = tc.eye(vector_size) * tc.rand(vector_size)
        self.eigenvectors = tc.tensor(linalg.orth(tc.rand(vector_size, vector_size) + 1 * tc.eye(vector_size)))
        self.function = tc.mm(tc.mm(self.eigenvectors * tc.tensor(graph).float(), self.diagonal),
                              tc.inverse(self.eigenvectors)) * tc.tensor(graph).float()
        # print(linalg.eigvals(self.function))
        self.time_series = [(self.init_vector)]
        self.values = None
        self.activation = nn.Tanh()
        # print(self.function)

        self.covariance, self.meanv = make_spd_matrix(vector_size, random_state=None), tc.zeros(vector_size)
        self.dist = tc.distributions.multivariate_normal.MultivariateNormal(self.meanv, tc.tensor(
            self.covariance).float())  # noise distribution

    def step(self, n):
        for i in range(n):
            eps = self.dist.sample() if correlated else 1 * tc.randn(self.vector_size)
            next_value = tc.matmul(self.function, (self.time_series[-1] + eps))
            next_value = self.activation(next_value)
            self.time_series.append(next_value)
        self.values = numpy.array(tc.stack(self.time_series, dim=0))

    def get_set(self, size, warmup=50):
        self.time_series = [(self.init_vector)]
        self.step(warmup + size - 1)
        return (self.values[warmup:, :], numpy.array(self.function))

    def get_sampled_set(self, size, warmup=50):
        set = []
        for i in range(size):
            self.time_series = [tc.rand(self.vector_size)]
            self.step(warmup)
            set.append(self.time_series[-1])
        return numpy.array(tc.stack(set, dim=0)), numpy.array(self.function)


def get_data(n):
    ar_model = AR(nfeatures, specific_network([0, 1, 2, 3]))
    art_train = ar_model.get_sampled_set(nsamples, 100)[0]
    #art_test = ar_model.get_sampled_set(nsamples, 100)[0]
    return art_train#, art_test

art_train = get_data(4)
