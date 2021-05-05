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
        nn.Linear(input_dim, output_dim, bias=True)
    )
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

#class ResBlock as smallest Unit
class ResBlock(nn.Module):
    def __init__(self, input_dim, width,  act_bool, activation=nn.LeakyReLU(0.1)):
        super(ResBlock, self).__init__()

        self.input_dim, self.act_bool = input_dim, act_bool
        self.layers = nn.Sequential(
            #bn_linear(input_dim, width),
            nn.Linear(input_dim, width),
            activation,
            nn.Dropout(p=0.02),
            bn_linear(width, width),
            activation,
            nn.Dropout(p=0.02),
            bn_linear(width, input_dim),
            activation if act_bool else nn.Identity()
        )
        self.layers.apply(init_weights)
    def forward(self,x):
        assert tc.is_floating_point(x), 'input is not float'
        residual = x.clone()
        return residual + self.layers(x)

class VAE(nn.Module):
    def __init__(self, input_dim, width, sample_width, depth, variational=True, nonlinear = True, k=1):
        super(VAE,self).__init__()
        #specify if Autoencoder is variational, linear or nonlinear and if IWAE is applied with k iterations
        self.variational = variational
        self.lin = 'nonlinear' if nonlinear else 'linear'
        self.activation = nn.LeakyReLU() if nonlinear else nn.Identity()
        self.k = k

        self.encoder = nn.Sequential(nn.Linear(input_dim, width),
            *[ResBlock(width, 2*width, act_bool=True, activation = self.activation) for _ in range(depth)]
        )

        self.mean_layer, self.logvar_layer = nn.Linear(width, sample_width), nn.Linear(width, sample_width)

        self.decoder = nn.Sequential(
            nn.Linear(sample_width, width),
            *[ResBlock(width, 2*width, act_bool=True) for _ in range(depth)],
            nn.Linear(width, input_dim)
        )

    def sample(self, mean, log_var, k=1):
        std = tc.exp(0.5 * log_var).repeat(k,1)
        eps = tc.randn_like(std)
        results = mean.repeat(k,1) + eps*std
        results = tc.clamp(results, min=-10, max=+10)
        #print(tc.any(tc.abs(results)>10))

        #std = tc.exp(0.5 * log_var)
        #eps = tc.randn_like(std)
        #results = mean + eps*std
        return results

    def forward(self, x):
        encoding = self.encoder(x)
        #compute mean and log variance
        mean_l, log_var_l = self.mean_layer(encoding), self.logvar_layer(encoding)
        if self.variational:
            latent = self.sample(mean_l, log_var_l, k = self.k)
            reconstruction = self.decoder(latent).reshape(self.k, *(x.shape)).mean(dim=0)
        else:
            latent = mean_l
            reconstruction = self.decoder(latent)

        # return mean und log variance for kl loss in VAE framework
        return reconstruction, mean_l, log_var_l

# wrapper for VAE: repeatedly maps protein data with VAE on correct protein data. between repeats known data is initialized again as ground truth
class GibbsSampler(nn.Module):
    def __init__(self, neuralnet, warm_up, convergence, result_path, max_repeats = 15, device = 'cpu'):
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
                #print(t)
                mov_avg = ((t-1)/t) * mov_avg + (1/t) * x.detach()
                self.results.append(mov_avg)
                if tc.all(tc.abs(self.results[-2]-self.results[-1]) < self.convergence):
                    print('converged at', len(self.results)-2)
                    break

        return mov_avg


    def train(self, batch_masked, batch_target, Mask, lr, train_repeats, epoch):
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
            if self.neuralnet.variational:
                kl_loss = -0.5 * tc.mean(1 + log_var_ - mean_.pow(2) - log_var_.exp())

            else:
                kl_loss= 0

            loss = criterion(prediction[Mask == 0], batch_target[Mask == 0]) + kl_loss

            if i>1 or epoch < 400:
                loss.backward()
                optimizer.step()

    def test(self, batch_masked, batch_target, Mask):
        self.neuralnet.eval().to(self.device)
        criterion = F.mse_loss

        mse_losses=[]
        batch_masked, batch_target, Mask = batch_masked.to(self.device), batch_target.to(self.device), Mask.to(self.device)

        endresult = self.forward(batch_masked, Mask)
        #print(criterion(endresult, batch_target))
        intermediate_singe_results = [criterion(prediction[Mask==0], batch_target[Mask==0]) for prediction in self.single_results if tc.is_tensor(prediction)]
        intermediate_averaged_results =  [criterion(prediction[Mask==0], batch_target[Mask==0]) for prediction in self.results if tc.is_tensor(prediction)]
        return intermediate_averaged_results, intermediate_singe_results

def cross_validate(model, train_data, test_data, path, train_epochs, lr,train_repeats, batch_factor, ncrossval=1, proportion = [0.7,0.1, 0.2]):
    #todo crossvalidation
    nsamples, nfeatures = train_data.shape
    tc.manual_seed(0)
    trainset = ProteinSet(train_data, batch_factor)
    testset = ProteinSet(test_data, batch_factor)
    #trainloader = DataLoader(trainset, batch_size = 1000, shuffle = True)
    testloader = DataLoader(testset, batch_size = nsamples*batch_factor, shuffle = False)

    for epoch in range (train_epochs):
        trainset.R = trainset.init_randomsample()
        trainloader = DataLoader(trainset, batch_size=nsamples * batch_factor, shuffle=True)

        if epoch < train_epochs*1/4:
            lr_true = lr
        elif epoch < train_epochs*2/4:
            lr_true = lr / 10
        elif epoch < train_epochs*3/4:
            lr_true = lr / 100
        else:
            lr_true = lr/1000

        for masked_data,target, Mask in trainloader:
            model.train(masked_data, target, Mask, lr = lr_true, train_repeats = train_repeats, epoch=epoch)
        if epoch in [1, 10, 50, 100, 200, 300, 500, 700, 800, 1000, 1300, 1500, 1800, 2000, 2300, 2500, 2800, 3000, 3500, 4000]:
            print('trained', epoch)
            tc.save(model, path + '/trained_model/Gibbs_sampler_trainepochs={}_var={}_k={}_{}.pt'.format(epoch, model.neuralnet.variational, model.neuralnet.k, model.neuralnet.lin))

            for masked_data, target, Mask in testloader:
                with tc.no_grad():
                    averaged_results, single_results = model.test(masked_data,target,Mask)
                    print('test_loss', averaged_results[-1], single_results[-1])
                    pandasframe_a = pd.DataFrame({'averages_results': [x.cpu().numpy() for x in averaged_results]})
                    pandasframe_s = pd.DataFrame({'single_results': [x.cpu().numpy() for x in single_results]})

                    pandasframe_a.to_csv(path + '/log/average_trainepochs={}_var={}_k={}_{}.csv'.format(epoch, model.neuralnet.variational, model.neuralnet.k, model.neuralnet.lin))
                    pandasframe_s.to_csv(path + '/log/single_trainepochs={}_var={}_k={}_{}.csv'.format(epoch, model.neuralnet.variational, model.neuralnet.k, model.neuralnet.lin))

            for masked_data, target, Mask in trainloader:
                with tc.no_grad():
                    averaged_results_train, single_results_train = model.test(masked_data,target,Mask)
                    print('train_loss', averaged_results_train[-1], single_results_train[-1])
                    break


class ProteinSet(Dataset):
    def __init__(self, data, factor):
        self.factor=factor
        self.data = data.float()
        self.nsamples, self.nfeatures = data.shape
        self.R = self.init_randomsample()

    def init_randomsample(self):
        tensor_list = [self.data[tc.randperm(self.nsamples),i] for i in range(self.nfeatures)]
        return tc.stack(tensor_list).t()

    def __len__(self):
        return self.nsamples*self.factor

    def __getitem__(self, idx):
        Mask = tc.distributions.bernoulli.Bernoulli(tc.tensor([0.5] * self.nfeatures)).sample().float()
        target = self.data[idx%self.nsamples, :]
        random_values = tc.zeros_like(self.R[idx%self.nsamples, :])
        masked_data = tc.where(Mask == 1.0, target, random_values)

        return masked_data, target, Mask


'''
class RandomFeature:
    def __init__(self, dim, sigma=None):
        self.dim = dim
        self.sigma = sigma
        self.omega = None
        self.tau = None
        self.pi = tc.tensor(np.pi)

    def fit(self, X):
        n, d = X.shape
        if self.sigma is None:
            self.sigma = 1 / X.var()
        self.omega = tc.randn(d, self.dim)
        self.tau = 2 * self.pi * tc.rand(1, self.dim)

    def transform(self, X):
        product = tc.einsum('sf, zd -> sdz', X, self.omega)
        feat = tc.cos(product* self.sigma + self.tau.unsqueeze(2))
        return (2 / self.dim)**0.5 * feat

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
        
        
def hsic_loss(target, prediction):
    ns, nf = target.shape
    residuals = target#-prediction
    ff = RandomFeature(dim=1024)
    z_vector_v = ff.fit_transform(residuals)
    H = tc.eye(ns) - 1.0/ nf *tc.ones(ns, ns)
    gencov = 1 / nf  + tc.einsum('sdz, ss, rdy -> zy', z_vector_v, H, z_vector_v)
    return gencov**2 #returns z*z matrix (which is the same as nfeatures*nfeatures)


'''