import torch as tc
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random
from joblib import Parallel, delayed
import pandas as pd

class ShapleySet(Dataset):
    # ShapleySet generates the masked data from ground truth data. Two masks are returned, with and without p masked
    def __init__(self, data, p, pvalue, q):
        self.nsamples, self.nfeatures = data.shape[0]//2, data.shape[1] #ugly solution to do shapley only on test set (first half ist trianing set)
        self.data = data[self.nsamples:,:]
        self.p = p
        self.pvalue = pvalue
        self.q = q
        self.Mask = None
        self.getMask()

    def init_randomsample(self):
        # permute samples for each feature in order to later sample from it
        tensor_list = [self.data[tc.randperm(self.nsamples), i].float() for i in range(self.nfeatures)]
        return tc.stack(tensor_list).t()

    def getMask(self):
        # generate Mask for ShapleySet
        self.Mask = tc.ones(self.nfeatures)
        self.Mask[self.q] = 0 # q is the only position that should change by itself, p is set from outside

    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx):
        assert self.Mask is not None, 'did not sample'
        sample = self.data[idx, :].float()
        sample[self.p] = self.pvalue
        return sample, self.Mask

class Counter:
    def __init__(self, model, data, protein_names, device):
        self.data = data
        self.nsamples, self.nfeatures = data.shape # nsamples is double the length as in ShapleySet
        self.model = model
        self.model.neuralnet.eval()
        self.protein_names = protein_names if protein_names else range(self.nfeatures)
        self.device = device

    def calc_counterfactualpq(self, p, pvalue, q, device):
        #calculate shapley values for one predicting protein p

        # shapleyset is initialized for every pvalue
        self.shapleyset = ShapleySet(self.data, p, pvalue, q) # not necessary to make it a instance variable?
        self.shapleyloader = DataLoader(self.shapleyset, batch_size=self.nsamples) # take the whole dataset as sample
        self.model.to(device)

        for sample, Mask in self.shapleyloader:
            sample, Mask = sample.to(device), Mask.to(device)
            #calculate prediction and loss
            with tc.no_grad():
                pred = self.model(sample, Mask)
                result = pred.mean(axis=0)[q]

        pandasframe = pd.DataFrame(data = {'p':p,'q':q, 'pvalue':pvalue.cpu().numpy(), 'qvalue': result.cpu().numpy()}, index = [0])
        pandasframe.to_csv('results/counter/data/batched_countervalues_{}_{}_{:.2f}_.csv'.format(self.protein_names[p], self.protein_names[q], pvalue), index=False)

    def calc_all(self, device):
        for p in range(self.nfeatures):
            for q in range(self.nfeatures):
                print(p,q)
                Parallel(n_jobs=1)(delayed(self.calc_counterfactualpq)(p, pvalue, q, device) for pvalue in tc.arange(start=-5, end=5, step=0.2))

