import torch as tc
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from classes import *


def get_dataloaders(data, p):
    dataset = Mydataset(data, p = p)
    tc.manual_seed(0)
    traindata, testdata = random_split(dataset, [4000, 730])
    trainloader = DataLoader(traindata, batch_size = 2000, shuffle = True)
    testloader = DataLoader(testdata, batch_size = 1, shuffle = True)
    return trainloader, testloader

def train_it(neuralnet, trainloader, lr=0.001):
    device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
    neuralnet.to(device).train()
    optimizer = optim.Adam(neuralnet.parameters(), lr=lr, weight_decay = 4e-3)
    criterion = F.mse_loss

    for i, (data, target) in enumerate(trainloader):
        optimizer.zero_grad()

        data = data.to(device)
        target = target.to(device)

        prediction,_, _ = neuralnet(data)
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()

def test_it(neuralnet, testloader, repeats = 5):
    device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
    neuralnet.to(device).eval()

    criterion = F.mse_loss
    mse_losses = []
    for i, (data, target) in enumerate(testloader):
        data = data.to(device)
        target = target.to(device)

        prediction, pred_history, nans = neuralnet(data, repeats = repeats)

        loss = criterion(prediction, target, reduction = 'none').mean(dim=1)
        mse_losses.append(tc.tensor([criterion(pred[nans], target[nans]) for pred in pred_history]).unsqueeze(0))

    all_mse_losses = tc.cat(mse_losses, dim=0).mean(0)

    return all_mse_losses



def plot_results(results):
    #plt.close()
    plt.ion()
    plt.show()
    for j, result in enumerate(results):
        plt.figure(0)
        plt.plot(tc.arange(result.shape[0]), result, color=(1 - j / 30, j / 30, j /60))
        plt.draw()
        plt.pause(0.001)


def importance_test(neuralnet, testloader, repeats):
    device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
    neuralnet.to(device).eval()

    criterion = F.mse_loss
    mse_losses = []
    for i, (data, target) in enumerate(testloader):
        data = data.to(device)
        target = target.to(device)

        prediction, pred_history, nans = neuralnet(data, repeats = repeats)

        loss = criterion(prediction, target, reduction = 'none').mean(dim=1)

    return loss, nans