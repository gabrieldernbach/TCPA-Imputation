import torch as tc
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random
from joblib import Parallel, delayed
import pandas as pd
import os
import networkx as nx
import itertools
import numpy as np

class ShapleySet(Dataset):
    # ShapleySet generates the masked data from ground truth data. Two masks are returned, with and without p masked
    def __init__(self, data, p, q, conditional, probability= 0.5):
        self.probability = probability # with 0.5, all combinations of masked and unmasked proteins are equally likely
        #tc.manual_seed(random.randint(1,100000)) # set seed for ... delete?
        self.nsamples, self.nfeatures = data.shape[0], data.shape[1]
        self.data = data
        self.R = self.init_randomsample()
        self.p = p
        self.q = q
        self.conditional = conditional
        self.Mask, self.Maskp = None, None
        self.getMasks()

    def init_randomsample(self):
        # permute samples for each feature in order to later sample from it
        tensor_list = [self.data[tc.randperm(self.nsamples), i].float() for i in range(self.nfeatures)]
        return tc.stack(tensor_list).t()

    def getMasks(self):
        # generate Mask for ShapleySet
        self.Mask = tc.distributions.bernoulli.Bernoulli(tc.tensor([self.probability] * self.nfeatures)).sample()
        self.Mask[self.conditional] = 1
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

    def calc_shapleypq(self, p, q, conditional, steps, device, probability):
        #calculate shapley values for one predicting protein p

        # shapleyset is initialized for every protein p
        self.shapleyset = ShapleySet(self.data, p, q, conditional, probability) # not necessary to make it a instance variable?
        self.shapleyloader = DataLoader(self.shapleyset, batch_size=self.nsamples) # take the whole dataset as sample
        self.model.to(device)
        meandiff = tc.zeros(1).to(self.device) # initialize the mean difference between sets with and without p
        criterion = F.mse_loss
        convergencechecker = [a for a in range(10)] # random numbers

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

                loss, lossP = criterion(pred[:, q], target[:, q]), criterion(predP[:, q], target[:, q])

                meandiff = (t - 1) / t * meandiff + 1 / t * (loss - lossP)

                convergencechecker.append(meandiff)

            if tc.all(tc.abs(tc.tensor(convergencechecker[-10:-1]) - tc.tensor(convergencechecker[-9:]))<0.0001) and t >2000:
                #break if consequent meanvalues are not different
                print(p, q, 'converged at', len(convergencechecker))
                break

        pandasframe = pd.DataFrame(data = {'p':self.protein_names[p], 'q':self.protein_names[q], 'conditional': self.protein_names[conditional],'shapley': meandiff.cpu().detach()})
        pandasframe.to_csv('results/triangle/batched_shapley_values_{}_{}_{}_{:.2f}_{}_specific.csv'.format(self.protein_names[p], self.protein_names[q], self.protein_names[conditional], probability, len(convergencechecker)-1), index=False)

    def calc_all(self, device, steps, probabilities=[0.5]):
        edges = get_edges()
        for probability in probabilities:
            Parallel(n_jobs=4)(delayed(self.calc_shapleypq)(p, q, conditional, steps, device, probability) for p, q, conditional in edges)





def get_edges():
    def load_file(filename):
        file_data = pd.read_csv(os.getcwd() + '/results/shapley/' + filename)
        target = filename.split('_')[3]
        file_data['target'] = target
        return file_data

    filenames = os.listdir(os.getcwd() + '/results/shapley/')
    data = pd.concat([load_file(filename) for filename in filenames])
    data['target'] = data['target'].astype(int)
    threshold = np.median(data['shapley'])
    data2 = data[data['shapley'] > threshold]
    edge_list = (list(zip(list(data2['target']), list(data2['masked_protein']))))
    graph = nx.from_edgelist(edge_list)

    all_cliques = nx.enumerate_all_cliques(graph)
    triad_cliques = [x for x in all_cliques if len(x) == 3]


    def get_triads(triad_clique):
        duo_list = list(itertools.permutations(triad_clique))

        return duo_list

    triangle_edges = [get_triads(a) for a in triad_cliques]
    flattened_edges = [item for sublist in triangle_edges for item in sublist]
    flattened_edges = list(set(flattened_edges))
    print(flattened_edges)
    print('edges:', len(flattened_edges))
    return flattened_edges