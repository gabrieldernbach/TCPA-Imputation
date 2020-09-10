import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import torch as tc

from functions import *
from classes import *
from analysis import *

data_with_names = pd.read_csv('../data2/TCPA_data_sel.csv')
ID, data = data_with_names.iloc[:,:2], tc.tensor(data_with_names.iloc[:,2:].values)
print(data.shape)
data_train, data_test, _,_ = train_test_split(np.array(data),np.array(data), test_size= 0.2, random_state=0)

