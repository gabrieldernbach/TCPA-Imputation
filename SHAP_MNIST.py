import torch.optim as optim
import torch.nn.functional as F
import random
import torch
import torch as tc
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import random
import torchvision
from torchvision import transforms
import torchvision.datasets as datasets
from Classes import *
from torch.distributions.beta import Beta
from torch.distributions.bernoulli import Bernoulli
import matplotlib.pyplot as plt
import numpy as np


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])

transform_train = transforms.Compose([
    #transforms.RandomAffine(30, translate=(0.1, 0.1), scale=(0.8, 1.3)),
    #transforms.RandomResizedCrop(32, scale=(0.9, 1)),
    #transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize(32),
    transforms.ToTensor()])

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
#print(mnist_trainset)
#trainloader = DataLoader(mnist_trainset, batch_size = 1)

class Block(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Block,self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size =3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(output_dim),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=2),
            nn.ReLU()
        )

    def forward(self,x):
        return(self.layers(x))


class Net(nn.Module):
    def __init__(self, input_dim, dim1, dim2, in_features, n_classes):
        super(Net, self).__init__()

        self.layers = nn.Sequential(
            Block(input_dim, dim1),
            Block(dim1,dim2))

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.FCN = nn.Linear(in_features, n_classes)

    def forward(self,x):
        x = self.layers(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = F.log_softmax(self.FCN(x))
        return x



def train_with_eval(epoch):
    net.train()
    correct = 0
    for batch_id, ((masked_data, target, mask), _ ) in enumerate(masked_loader):
        data = masked_data.cuda()
        target = target.cuda()
        optimizer.zero_grad()
        out = net(data)
        prediction = out.data.max(1, keepdim=True)[1]
        correct += prediction.eq(target.data.view_as(prediction)).cpu().sum()

        criterion = F.nll_loss
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
    print('Epoche:', epoch)

    precision = correct.item() /len(masked_loader.dataset)
    print('training_precision:', precision)


class Masked_data(Dataset):
    def __init__(self, alpha, beta):
        self.data = mnist_trainset
        self.alpha, self.beta = alpha,beta
        self.image_shape = self.data[0][0].shape
        self.image_length = self.image_shape[1]**2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        percent_masked = random.uniform(self.alpha, self.beta)
        mask1 = self.get_mask(percent_masked)
        mask2 = self.get_mask(percent_masked)

        masked_data_idx1, target_idx = self.data[idx]
        masked_data_idx2 = masked_data_idx1.clone()

        masked_data_idx1[mask1==1] = -1
        masked_data_idx2[mask2==1]=-1

        return (masked_data_idx1.float(),target_idx, mask1), (masked_data_idx2.float(),target_idx, mask2)

    def get_mask(self, percent_masked):
        n_masked = int(self.image_length*percent_masked)
        index= tc.randint(0, self.image_length, (n_masked,))
        mask=tc.zeros(self.image_length)
        mask[index] = tc.tensor([1])
        mask=mask.reshape(self.image_shape)
        return mask

class One_mask(Masked_data):
    def __init__(self):
        super(One_mask,self).__init__(alpha=0,beta=0)
    def __len__(self):
        return self.image_length

    def __getitem__(self,idx):
        one_value_image = tc.zeros(self.image_length)
        one_value_image[idx] = 1
        return one_value_image.reshape(self.image_shape).float()

class Sorter(nn.Module):
    def __init__(self):
        super(Sorter,self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1,10, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(10,10, kernel_size=3,stride=2),
            nn.ReLU())


        self.target_worker = nn.Linear(10,10)
        self.discriminator = nn.Sequential(nn.Linear(500,30),
                                           nn.ReLU(),
                                           nn.BatchNorm1d(30),
                                           nn.Linear(30, 1))

        self.avg = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self,x,target):
            target = self.target_worker(F.one_hot(target, num_classes=10).float())

            x = self.layers(x)
            y = x.view(x.size(0), -1)
            print(y.shape, target.shape)
            y_target = tc.cat((y, target), dim=1)
            return self.discriminator(y_target)

def test_with_eval(net, data, target):
    net.eval()
    correct = 0
    data = data.cuda()
    target = target.cuda()
    optimizer.zero_grad()
    out = net(data)
    prediction = out.data.max(1, keepdim=True)[1]

    criterion = F.nll_loss
    return criterion(out, target, reduction= 'none')



def test_relevance(sorter):
    i = 0
    for (masked_data1, target1,mask1),(masked_data2,target2,mask2) in masked_loader:
        sort_optimizer.zero_grad()
        mask1, mask2, target = mask1.cuda(), mask2.cuda(), target1.cuda()
        with torch.no_grad():
            loss1 = test_with_eval(net,masked_data1, target1)
            loss2 = test_with_eval(net,masked_data2, target2)

        correct_sort = (loss2 > loss1).int().long().detach()
        print(correct_sort)

        sort_prediction1 = sorter(mask1, target)
        sort_prediction2 = sorter(mask2, target)

        softmax_values = F.log_softmax(tc.cat((sort_prediction1, sort_prediction2), dim=1), dim=1)
        #exp_pred2 = tc.exp(sort_prediction2)
        #softmax_pred2 = exp_pred2/(exp_pred2 + tc.exp(sort_prediction1))
        #print(softmax_values, correct_sort)
        criterion = nn.CrossEntropyLoss()  #-(correct_sort * tc.log(softmax_pred2) + (1-correct_sort)*tc.log(1-softmax_pred2)).mean()  # binary cross entropy ...todo improve loss function(KL-divergence?)
        loss = criterion(softmax_values, correct_sort)
        loss.backward()
        sort_optimizer.step()
        print(i)
        i += 1
    print(loss)


def return_relevance_map(sorter):
    target_size=10
    one_mask_dataset = One_mask()
    one_mask_dataloader = DataLoader(one_mask_dataset, batch_size = 32**2)
    results = tc.zeros(target_size,*one_mask_dataset[0].squeeze().shape).cuda()
    per_target_relevances = []
    for i in range(target_size): #target_size
        for data in one_mask_dataloader:
            unsorted_relevances = sorter(data.cuda(), tc.tensor([i]*32**2).cuda())
            per_target_relevances.append(np.array(unsorted_relevances.reshape((32,32)).detach().cpu()))

    return np.array(per_target_relevances)


net=Net(1,32,64,64,10).cuda()
optimizer = optim.Adam(net.parameters(), lr=0.001)


masked_data = Masked_data(0.0, 0.1)
masked_loader = DataLoader(masked_data, batch_size=3000, shuffle=True)

if os.path.isfile('MNIST_net.pt'):
    print('net found')
    net = torch.load('MNIST_net.pt')
else:
    print('net not found')
    for epoch in range(1):
        train_with_eval(epoch)
        torch.save(net, 'MNIST_net.pt')

sorter = Sorter().cuda()
sort_optimizer = optim.Adam(sorter.parameters(), lr=0.001, weight_decay = 4e-3)

for epoch in range(1):
    test_relevance(sorter)


relevances = return_relevance_map(sorter)
print(relevances.shape)
np.save('results/Relevance_Map.npy', relevances)
