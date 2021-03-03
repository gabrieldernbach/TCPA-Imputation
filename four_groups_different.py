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
nfeatures = 16
nsamples = 4000  # this is now the number of samples for train and test_set, so in sum 2*nsamples
connectivity = 1.0
block_size = (8,8)
correlated = True
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
sparse_graphs = [specific_network([i]) for i in range(4)]
full_graph = sparse_graphs[0] * 0 + 1

#################################################################
# ARMA model
import torch as tc
import torch.nn as nn
import scipy.linalg as linalg
from sklearn.datasets import make_spd_matrix


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
            next_value = tc.matmul(self.function,
                                   (self.time_series[-1] + self.dist.sample()))  # 1*tc.randn(self.vector_size)))
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
        #numpy.savetxt('/mnt/scratch2/mlprot/LRP_experiment/plots_statistics/use_data/time_series_tanh.csv', self.values,
        #              delimiter='\t')
        return numpy.array(tc.stack(set, dim=0)), numpy.array(self.function)


def get_data(n):
    ar_models = [AR(nfeatures, specific_network([i])) for i in range(n)]
    art_train = [ar.get_sampled_set(nsamples // n, 100)[0] for ar in ar_models]
    #art_test = [ar.get_sampled_set(nsamples // n, 100)[0] for ar in ar_models]
    return np.concatenate(art_train)


art_train = get_data(4)
labels = np.repeat(np.array([0,1,2,3]),nsamples/4) # cave: n in get_data is set to 4
