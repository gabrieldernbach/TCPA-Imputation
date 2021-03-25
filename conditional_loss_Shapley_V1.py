import torch as tc
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random
from joblib import Parallel, delayed
import pandas as pd

class ShapleySet(Dataset):
    # ShapleySet generates the masked data from ground truth data. Two masks are returned, with and without p masked
    def __init__(self, data, p, q):
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
        self.Mask = tc.ones(self.nfeatures)
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

class Shapley:
    def __init__(self, model, data, protein_names, device):
        self.data = data
        self.nsamples, self.nfeatures = data.shape
        self.model = model
        self.model.neuralnet.eval()
        self.protein_names = protein_names if protein_names else range(self.nfeatures)
        self.device = device

    def calc_shapleypq(self, p, q, device):
        #calculate shapley values for one predicting protein p

        # shapleyset is initialized for every protein p
        self.shapleyset = ShapleySet(self.data, p, q) # not necessary to make it a instance variable?
        self.shapleyloader = DataLoader(self.shapleyset, batch_size=self.nsamples) # take the whole dataset as sample
        self.model.to(device)
        meandiff = tc.zeros(1).to(self.device) # initialize the mean difference between sets with and without p
        criterion = F.mse_loss

        self.shapleyset.getMasks()
        for target, masked_data, Mask, masked_dataP, MaskP in self.shapleyloader:

            target, masked_data, Mask, masked_dataP, MaskP = target.to(device), masked_data.to(device), Mask.to(
                device), masked_dataP.to(device), MaskP.to(device)
            #calculate prediction and loss
            with tc.no_grad():
                pred, predP = self.model(masked_data, Mask), self.model(masked_dataP, MaskP)

            loss, lossP = criterion(pred, target), criterion(predP, target)
            meandiff = loss-lossP

        pandasframe = pd.DataFrame(data = {'value': meandiff.cpu().detach().numpy()}, index= [0])
        pandasframe.to_csv('results/conditional_loss/batched_shapley_values_{}_{}_specific.csv'.format(self.protein_names[p], self.protein_names[q]))

    def calc_all(self, device):
        for q in range(self.nfeatures):
            Parallel(n_jobs=1)(delayed(self.calc_shapleypq)(p, q, device) for p in range(self.nfeatures))






