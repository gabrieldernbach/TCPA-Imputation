import torch as tc
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random
from joblib import Parallel, delayed
import pandas as pd

class ShapleySet(Dataset):
    # ShapleySet generates the masked data from ground truth data. Two masks are returned, with and without p masked
    #this shapleyset is specific for a p and a sample
    def __init__(self, data, p, sample, probability= 0.5):
        self.probability = probability # with 0.5, all combinations of masked and unmasked proteins are equally likely
        #tc.manual_seed(random.randint(1,100000)) # set seed for ... delete?
        self.nsamples, self.nfeatures = data.shape[0], data.shape[1]
        self.raw_data = data
        self.data = data[sample,:]
        self.R = self.init_randomsample()
        self.p = p

    def init_randomsample(self):
        # permute samples for each feature in order to later sample from it
        tensor_list = [self.raw_data[tc.randperm(self.nsamples), i].float() for i in range(self.nfeatures)]
        return tc.stack(tensor_list).t()

    def getMasks(self):
        # generate Mask for ShapleySet
        Mask = tc.distributions.bernoulli.Bernoulli(tc.tensor([self.probability] * self.nfeatures)).sample()
        Maskp = Mask.clone()
        Mask[self.p], Maskp[self.p] = 0, 1
        return Mask, Maskp

    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx):
        target = self.data.float()
        random_values = self.R[idx, :].float()
        Mask, MaskP = self.getMasks()

        masked_data = tc.where(Mask == 1, target, random_values)
        masked_dataP = tc.where(MaskP == 1, target, random_values)

        return target, masked_data, Mask, masked_dataP, MaskP

class Shapley:
    def __init__(self, model, data, protein_names, device):
        self.data = data
        self.nsamples, self.nfeatures = data.shape
        self.model = model
        self.model.neuralnet.eval()
        self.protein_names = protein_names if protein_names else range(self.nfeatures)
        self.device = device

    def calc_shapleypqs(self, p, sample, steps, device, probability):
        #calculate shapley values for one predicting protein p and one sample

        # shapleyset is initialized for every protein p and every sample
        self.shapleyset = ShapleySet(self.data, p, sample, probability) # not necessary to make it a instance variable?
        self.shapleyloader = DataLoader(self.shapleyset, batch_size=self.nsamples) # take the whole dataset as sample
        self.model.to(device)
        meandiff = tc.zeros(1).to(self.device) # initialize the mean difference between sets with and without p
        criterion = F.mse_loss
        convergencechecker = [0,1] # random numbers
        counter = 1

        # add losses until convergence
        for t in range(1, 1+steps):
            for target, masked_data, Mask, masked_dataP, MaskP in self.shapleyloader:

                target, masked_data, Mask, masked_dataP, MaskP = target.to(device), masked_data.to(device), Mask.to(
                    device), masked_dataP.to(device), MaskP.to(device)
                #calculate prediction and loss
                with tc.no_grad():
                    pred, predP = self.model(masked_data, Mask), self.model(masked_dataP, MaskP)

                loss, lossP = criterion(pred, target, reduction = 'none'), criterion(predP, target, reduction = 'none')

                #add loss only for proteins q that were masked (=not visible)
                #specific = (Mask == 0).float()
                meanloss, meanlossP = loss.mean(dim=0), lossP.mean(dim=0)
                meandiff = (counter - 1) / counter * meandiff + 1 / counter * (meanloss - meanlossP)
                counter += 1
                convergencechecker.append(meandiff)

            if all(abs(convergencechecker[-2] - convergencechecker[-1])<0.00001) and t>100:
                #break if consequent meanvalues are not different
                print(p, 'converged at', len(convergencechecker))
                break

        pandasframe = pd.DataFrame(data = {'masked_protein': self.protein_names, 'shapley': meandiff.cpu().detach()})
        pandasframe.to_csv('results/singlesample_shapley/data/single_shapley_values_{}_{}_{:.2f}_{}_specific.csv'.format(self.protein_names[p], sample, probability, len(convergencechecker)-1), index=False)

    def calc_all(self, device, steps, probabilities=[0.5]):
        for probability in probabilities:
            for sample in range(self.nsamples):
                Parallel(n_jobs=1)(delayed(self.calc_shapleypqs)(p, sample, steps, device, probability) for p in range(self.nfeatures))

