import Time
import os
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import random
import pandas as pd
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import numpy as np


dim=4
#ar = Time.AR(4)
#ar.step(1000)
#plt.plot(ar.values)
#plt.show()

timedataset = Time.Timedataset(dim,5000)
timedataloader = DataLoader(timedataset, batch_size=500)
vae1 = Time.VAE(dim, width = dim*5, sample_width = dim*5, depth = 2)
vae2 = Time.VAE(dim, width = dim*5, sample_width = dim*5, depth = 2)
gan = Time.GAN(dim)

for epoch in range(5000):
    pass
    Time.trainVAE(gan, vae1, vae2, timedataloader, lr = 0.000001)
    Time.trainGAN(gan, vae1, vae2, timedataloader, lr = 0.00001)

    if epoch %30 ==0:
        Time.test(vae1,vae2,timedataloader)