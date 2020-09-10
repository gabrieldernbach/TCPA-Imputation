import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ana_data = pd.read_csv('../ana/full-2.csv')
dict = {word:word.strip() for word in ana_data.columns}
ana_data = ana_data.rename(columns = dict)
#print(ana_data)

log_data = pd.read_csv('../log/train.py-full-2.txt', sep = " ", engine = 'python')

log_data = pd.read_csv('../log/train.py-full-2.txt', sep = "\|| ", engine = 'python').iloc[:, [0,3,4,7,8,9,10]]
log_data.columns = ['it', 'var', 'err', '1','2','3','4']

err = log_data['err']
its = log_data['it']
plt.plot(its[:9], err[:9])
plt.ylabel('MSE')
plt.xlabel('epochs')

plt.show()

