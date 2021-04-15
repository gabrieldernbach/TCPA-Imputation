import torch as tc
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from entropy_estimators import continuous
import itertools
from numpy.random import randn, rand

class ShapleySet(Dataset):
    # ShapleySet generates the masked data from ground truth data. Two masks are returned, with and without p masked
    def __init__(self, data, p, q, probability= 0.5):
        self.probability = probability # with 0.5, all combinations of masked and unmasked proteins are equally likely
        #tc.manual_seed(random.randint(1,100000)) # set seed for ... delete?
        self.nsamples, self.nfeatures = data.shape[0], data.shape[1]
        self.data = data
        self.R = self.init_randomsample()
        self.p = p
        self.q = q
        self.Mask, self.Maskp = None, None
        self.getMasks()

    def init_randomsample(self):
        # permute samples for each feature in order to later sample from it
        tensor_list = [self.data[tc.randperm(self.nsamples), i].float() for i in range(self.nfeatures)]
        return tc.stack(tensor_list).t()

    def getMasks(self):
        # generate Mask for ShapleySet
        self.Mask = tc.distributions.bernoulli.Bernoulli(tc.tensor([self.probability] * self.nfeatures)).sample()
        self.Mask[self.q] = 0
        self.Maskp = self.Mask.clone()
        self.Mask[self.p], self.Maskp[self.p] = 0, 1


    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx):
        assert self.Mask is not None, 'did not sample'
        target = self.data[idx, :].float()
        random_values = self.R[idx, :].float()

        masked_data = tc.where(self.Mask == 1, target, random_values)
        masked_dataP = tc.where(self.Maskp == 1, target, random_values)

        return target, masked_data, self.Mask, masked_dataP, self.Maskp


class RandomFeature:
    def __init__(self, dim, sigma=None):
        self.dim = dim
        self.sigma = sigma
        self.omega = None
        self.tau = None

    def fit(self, X):
        n, d = X.shape
        if self.sigma is None:
            self.sigma = 1 / X.var()
        self.omega = randn(d, self.dim)
        self.tau = 2 * np.pi * rand(1, self.dim)

    def transform(self, X):
        feat = np.cos(X @ self.omega * self.sigma + self.tau)
        return np.sqrt(2 / self.dim) * feat

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def hsic_plus(a, b):
    n, _ = a.shape
    ff = RandomFeature(dim=1024)
    za = ff.fit_transform(a)
    zb = ff.fit_transform(b)
    H = np.eye(n) - 1.0 / n * np.ones((n, n))
    gencov = 1 / n * za.T @ H @ zb
    return np.sum(gencov ** 2)


def distance(a, b):
    # euclidean distance (a - b)**2 = a**2 - 2ab + b**2
    asq = np.sum(a ** 2, -1, keepdims=True)
    bsq = np.sum(b ** 2, -1, keepdims=True)
    return asq - 2 * a @ b.T + bsq.T


def hsic(a, b, sig_a=0.1, sig_b=0.1):
    n, _ = a.shape
    K = np.exp(-distance(a, a) / sig_a)
    L = np.exp(-distance(b, b) / sig_b)
    H = np.eye(n) - 1.0 / n * np.ones((n, n))
    traced = np.trace(L @ H @ K @ H)
    Z = ((n - 1) ** 2)
    return traced / Z


class Shapley:
    def __init__(self, model, data, protein_names, device):
        self.data = data
        self.nsamples, self.nfeatures = data.shape
        self.model = model
        self.model.neuralnet.eval()
        self.protein_names = protein_names if protein_names else range(self.nfeatures)
        self.device = device

    def calc_shapleypq(self, p, q, steps, device, probability):
        print(p,q)
        if p==q:
            return p,q
        #calculate shapley values for one predicting protein p

        # shapleyset is initialized for every protein p
        self.shapleyset = ShapleySet(self.data, p, q,probability) # not necessary to make it a instance variable?
        self.shapleyloader = DataLoader(self.shapleyset, batch_size=self.nsamples) # take the whole dataset as sample
        self.model.to(device)
        hsic_final = tc.tensor(1).to(self.device) # initialize the mean difference between sets with and without p
        criterion = F.mse_loss
        convergencechecker = [a for a in range(10)] # random numbers
        counter = tc.ones(self.nfeatures).to(device)

        # add losses until convergence
        for t in range(1, 1+steps):
            if (t % 500) ==0:
                print(t)
            self.shapleyset.getMasks()
            for target, masked_data, Mask, masked_dataP, MaskP in self.shapleyloader:
                #print(Mask) # check if Masks are different every time!
                target, masked_data, Mask, masked_dataP, MaskP = target.to(device), masked_data.to(device), Mask.to(
                    device), masked_dataP.to(device), MaskP.to(device)
                #calculate prediction and loss
                with tc.no_grad():
                    pred, predP = self.model(masked_data, Mask), self.model(masked_dataP, MaskP)


                residualsQ = np.array(pred[:, q].cpu()- target[:, q].cpu())
                residualsP = np.array(pred[:, p].cpu()- target[:, p].cpu())

                hisc_value = tc.tensor(hsic_plus(residualsP[:,None], residualsQ[:,None])).float()

                hsic_final = tc.max(hisc_value, hsic_final)
                #meandiff = (t - 1) / t * meandiff + 1 / t * tc.tensor(hisc_final)

                # counter remembers the frequency of q being masked
                convergencechecker.append(hsic_final)

            if tc.all(tc.abs(tc.tensor(convergencechecker[-9:-1]) - tc.tensor(convergencechecker[-8:]))<0.00001):
                #break if consequent meanvalues are not different
                print(p, 'converged at', len(convergencechecker))
                break

        pandasframe = pd.DataFrame(data = {'target':  self.protein_names[q], 'source': self.protein_names[p], 'shapley': [hsic_final.cpu().detach().numpy()]})
        pandasframe.to_csv('results/shapley/batched_shapley_values_{}_{}_{:.2f}_{}_specific.csv'.format(self.protein_names[p], self.protein_names[q], probability, len(convergencechecker)-1), index=False)

    def calc_all(self, device, steps, probabilities=[0.5]):
        for probability in probabilities:
                Parallel(n_jobs=4)(delayed(self.calc_shapleypq)(p, q, steps, device, probability) for p, q in itertools.product(range(self.nfeatures), range(self.nfeatures)) if p>q)






