
import os
import torch
import torch as tc
import torch.nn as nn
import torch.autograd
from sklearn.manifold import TSNE
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import random

from torch.distributions.uniform import Uniform


data_with_names = pd.read_csv('../data2/TCPA_data_sel.csv')
ID, data = data_with_names.iloc[:,:2], tc.tensor(data_with_names.iloc[:,2:].values)



#tsne = TSNE().fit_transform(data)
#dataforplot = np.vstack((tsne.T, ID['Tumor'])).T
#tsne_df = pd.DataFrame(data=dataforplot, columns = ('d1','d2','label'))
#
#sn.FacetGrid(tsne_df, hue='label', size=6).map(plt.scatter, 'd1', 'd2')



from torch.utils.data import Dataset, DataLoader
data.shape
class DataSet(Dataset):
    def __init__(self,data):
        self.data = data
        self.len = data.size(0)
        
    def __getitem__(self,idx):
        false_idx = random.choice(list(range(idx)) + list(range(idx,self.len)))
        
        return self.data[idx,:].float(), self.data[false_idx,:].float()  
    
    def __len__(self):
        return self.len
    

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(hidden_dim, output_dim))
        
    def forward(self,x):
        return(self.layers(x))
    
    
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Discriminator, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim))
        
    def forward(self,x):
        return(F.sigmoid(self.layers(x)))
    
class JS_Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(JS_Discriminator,self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim))
        
    def forward(self,x):
        return(F.sigmoid(self.layers(x)))
    


device = torch.device('cuda:0')
representation_size = 20

dataset = DataSet(data)
Dataloader = DataLoader(dataset, batch_size = 1, shuffle=True)


test_example,false_example = next(iter(Dataloader))
encoder = Encoder(data.size(1), 512, representation_size)
discriminator = Discriminator(data.size(1) + representation_size, 60,1)
jsdiscriminator = JS_Discriminator(representation_size, 60, 1)



encoder.to(device), discriminator.to(device), jsdiscriminator.to(device)


optim_discriminator = tc.optim.Adam(discriminator.parameters(), lr= 0.0001)
optim_encoder = tc.optim.Adam(encoder.parameters(), lr=0.0001)
optim_jsdiscriminator = tc.optim.Adam(jsdiscriminator.parameters(), lr=0.0001)



def train(n_epochs):
    for j in range(n_epochs):
        for i,(correct_data, false_data) in enumerate(Dataloader):
            correct_data = correct_data.cuda()
            false_data =false_data.cuda()
            optim_discriminator.zero_grad(), optim_encoder.zero_grad()

            representation = encoder(correct_data)

            pos_samples = tc.cat((correct_data, representation), dim=1)
            neg_samples = tc.cat((false_data, representation), dim=1)

            pos_discrimination = discriminator(pos_samples)
            neg_discrimination = discriminator(neg_samples)
            
            loss1 = -(pos_discrimination.mean(dim=0) - tc.log(tc.exp(neg_discrimination).mean(dim=0)))
            loss2 = tc.log(jsdiscriminator(tc.rand(representation.shape))).mean(dim=0) + tc.log(jsdiscriminator(representation)).mean(dim=0)
            
            if i%4==0:
                loss = loss1 + loss2
                
                loss.backward()

                optim_discriminator.step() 

                optim_jsdiscriminator.step()
            
            else: 
                loss = loss1-loss2
                loss.backward()

                optim_encoder.step() 
                
        if i%1 == 0: print(loss)




torch.cuda.set_device(0)
train(100)

with tc.no_grad():
    encoded_data = encoder(data.float().clone().detach()).cpu()



encoded_data



tsne = TSNE().fit_transform(np.array(encoded_data))
dataforplot = np.vstack((tsne.T, ID['Tumor'])).T
tsne_df = pd.DataFrame(data=dataforplot, columns = ('d1','d2','label'))
sn.FacetGrid(tsne_df, hue='label', size=6).map(plt.scatter, 'd1', 'd2')
plt.show()





