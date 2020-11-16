import os
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import random
import pandas as pd

def bn_linear(input_dim: int, output_dim: int):
    return nn.Sequential(
        nn.BatchNorm1d(input_dim),
        nn.Linear(input_dim, output_dim, bias=False)
    )

#class ResBlock as smallest Unit
class ResBlock(nn.Module):
    def __init__(self, input_dim, width,  act_bool, activation=nn.LeakyReLU()):
        super(ResBlock, self).__init__()

        self.alpha = nn.Parameter(tc.tensor(1.0, requires_grad=True)) # delete?
        self.input_dim, self.act_bool = input_dim, act_bool
        self.layers = nn.Sequential(
            bn_linear(input_dim, width),
            activation,
            nn.Dropout(p=0.1),
            bn_linear(width, input_dim),
            activation if act_bool else nn.Identity()
        )

    def forward(self,x):
        assert tc.is_floating_point(x), 'input is not float'
        residual = x.clone()
        return residual*self.alpha + self.layers(x)

class VAE(nn.Module):
    def __init__(self, input_dim, width, sample_width, depth, variational=True, nonlinear = True, k=1):
        super(VAE,self).__init__()
        #specify if Autoencoder is variational, linear or nonlinear and if IWAE is applied with k iterations
        self.variational = variational
        self.lin = 'nonlinear' if nonlinear else 'linear'
        self.activation = nn.LeakyReLU() if nonlinear else nn.Identity()
        self.k = k

        self.encoder = nn.Sequential(
            *[ResBlock(input_dim, width, act_bool=False, activation = self.activation) for _ in range(depth)]
        )

        # todo: sample width smaller/larger/equal to width?
        self.mean_layer, self.logvar_layer = nn.Linear(input_dim, sample_width), nn.Linear(input_dim, sample_width)

        self.decoder = nn.Sequential(
            nn.Linear(sample_width, input_dim),
            *[ResBlock(input_dim, width, act_bool=False) for _ in range(depth)]
        )

    def sample(self, mean, log_var, k=1):
        #std = tc.exp(0.5 * log_var).repeat(k,1,1)
        #eps = tc.randn_like(std)
        #samples = mean + eps*std
        #results = samples.mean(dim=0)

        std = tc.exp(0.5 * log_var)
        eps = tc.randn_like(std)
        results = mean + eps*std
        return results

    def forward(self, x):
        encoding = self.encoder(x)

        #compute mean and log variance
        mean_l, log_var_l = self.mean_layer(encoding), self.logvar_layer(encoding)

        if self.variational:
            latent = self.sample(mean_l, log_var_l, k = self.k)
        else:
            latent = mean_l

        reconstruction = self.decoder(latent)
        # return mean und log variance for kl loss in VAE framework
        return reconstruction, mean_l, log_var_l

# wrapper for VAE: repeatedly maps protein data with VAE on correct protein data. between repeats known data is initialized again as ground truth
class GibbsSampler(nn.Module):
    def __init__(self, neuralnet, warm_up, convergence, result_path, max_repeats = 30, device = 'cpu'):
        super(GibbsSampler,self).__init__()
        self.neuralnet = neuralnet # VAE or other module
        self.warm_up = warm_up # first repeats are discarded
        self.max_repeats = max_repeats # maximum repeats if prediction does not converge
        self.device = device
        self.convergence = convergence # threshold for convergence decision
        self.result_path = result_path

    def forward(self,x, Mask):
        self.results = [0, self.convergence+1] # used to check convergence of Gibbs sampler
        self.single_results=[] # interpret behaviour of gibb sampler
        t=0
        self.neuralnet.to(self.device)
        x = x.to(self.device)
        mov_avg = tc.zeros_like(x)
        initial_value = x.clone()

        Mask = Mask.to(self.device)

        for i in range(self.max_repeats):
            x, mean_l, log_var_l = self.neuralnet(x)

            #replace known true values in reconstruction of x with initial_value
            x = tc.where(Mask==1, initial_value, x)
            self.single_results.append(x.detach())

            if i > self.warm_up:
                assert self.warm_up < self.max_repeats, 'max_repeats not higher than warm up, loop does not end'
                t+=1
                print(t)
                mov_avg = ((t-1)/t) * mov_avg + (1/t) * x.detach()
                print('sum', x.sum(),mov_avg.sum())
                self.results.append(mov_avg)
                if tc.all(tc.abs(self.results[-2]-self.results[-1]) < self.convergence):
                    print('converged at', len(self.results)-2)
                    break

        return mov_avg


    def train(self, batch_masked, batch_target, Mask, lr, train_repeats = 1):
        self.neuralnet.to(self.device).train()
        optimizer = tc.optim.Adam(self.neuralnet.parameters(), lr = lr)
        optimizer.zero_grad()
        criterion = F.mse_loss

        batch_masked, batch_target, Mask = batch_masked.to(self.device), batch_target.to(self.device), Mask.to(self.device)

        x = batch_masked.clone()
        loss = 0
        for i in range(train_repeats):
            prediction, mean_, log_var_ = self.neuralnet(x)
            x = tc.where(Mask==1, batch_masked, prediction)
            x.detach_()
            #compute KL-divergence and MSE Loss
            kl_loss = -0.5 * tc.mean(1 + log_var_ - mean_.pow(2) - log_var_.exp())
            loss += criterion(prediction[Mask == 0], batch_target[Mask == 0]) + kl_loss

        loss.backward()
        optimizer.step()

    def test(self, batch_masked, batch_target, Mask):
        self.neuralnet.eval().to(self.device)
        criterion = F.mse_loss

        mse_losses=[]
        batch_masked, batch_target, Mask = batch_masked.to(self.device), batch_target.to(self.device), Mask.to(self.device)

        endresult = self.forward(batch_masked, Mask)
        print(criterion(endresult, batch_target))
        intermediate_singe_results = [criterion(prediction, batch_target) for prediction in self.single_results if tc.is_tensor(prediction)]
        intermediate_averaged_results =  [criterion(prediction, batch_target) for prediction in self.results if tc.is_tensor(prediction)]
        return intermediate_averaged_results, intermediate_singe_results

def cross_validate(model, data, path, train_epochs, lr,train_repeats, ncrossval=1, proportion = [0.7,0.1, 0.2]):
    #todo crossvalidation
    nsamples, nfeatures = data.shape
    tc.manual_seed(0)
    trainset = ProteinSet(data[:3000,:])
    testset = ProteinSet(data[:3000,:])
    trainloader = DataLoader(trainset, batch_size = nsamples, shuffle = True)
    testloader = DataLoader(testset, batch_size = nsamples, shuffle = True)

    for epoch in range (train_epochs):
        print('train', epoch)
        for masked_data,target, Mask in trainloader:
            model.train(masked_data, target, Mask, lr = lr, train_repeats = train_repeats)
    tc.save(model, path + '/trained_model/Gibbs_sampler_trainepochs={}_var={}_k={}_{}.pt'.format(train_epochs, model.neuralnet.variational, model.neuralnet.k, model.neuralnet.lin))

    for masked_data, target, Mask in testloader:
        with tc.no_grad():
            averaged_results, single_results = model.test(masked_data,target,Mask)
            print(len(averaged_results), len(single_results))
            pandasframe_a = pd.DataFrame({'averages_results': [x.cpu().numpy() for x in averaged_results]})
            pandasframe_s = pd.DataFrame({'single_results': [x.cpu().numpy() for x in single_results]})

            pandasframe_a.to_csv(path + '/log/average_trainepochs={}_var={}_k={}_{}.pt'.format(train_epochs, model.neuralnet.variational, model.neuralnet.k, model.neuralnet.lin))
            pandasframe_s.to_csv(path + '/log/single_trainepochs={}_var={}_k={}_{}.pt'.format(train_epochs, model.neuralnet.variational, model.neuralnet.k, model.neuralnet.lin))

class ProteinSet(Dataset):
    def __init__(self, data):
        self.data = data.float()
        self.nsamples, self.nfeatures = data.shape
        self.R = self.init_randomsample()

    def init_randomsample(self):
        tensor_list = [self.data[tc.randperm(self.nsamples),i] for i in range(self.nfeatures)]
        return tc.stack(tensor_list).t()

    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx):
        Mask = tc.distributions.bernoulli.Bernoulli(tc.tensor([0.5] * self.nfeatures)).sample().float()
        target = self.data[idx, :]
        random_values = self.R[idx, :]
        masked_data = tc.where(Mask == 1.0, target, random_values)

        return masked_data, target, Mask




