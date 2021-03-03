import pandas as pd
import sklearn
import torch as tc
from sklearn import datasets
from sklearn.utils import shuffle
import Neural_net_generator as Ng
import numpy as np

def get_data(dataname):
    if dataname == "protein":
        DATAPATH = '../data2/TCPA_data_sel.csv'
        #import data from PATH
        data_with_names = pd.read_csv(DATAPATH)
        ID, data = data_with_names.iloc[:,:2], tc.tensor(data_with_names.iloc[:,2:].values)
        protein_names = data_with_names.columns[2:]
        tc.manual_seed(0)
        randomized_data = data[tc.randperm(data.shape[0]), :].float()

    elif dataname == 'iris':
        iris,_ = datasets.load_iris(return_X_y=True)
        randomized_data = tc.tensor(sklearn.preprocessing.scale(iris))

    elif dataname == 'boston':
        boston,_ = datasets.load_boston(return_X_y=True)
        randomized_data = tc.tensor(sklearn.preprocessing.scale(boston))

    elif dataname == 'breast_cancer':
        breast_cancer,_ = datasets.load_breast_cancer(return_X_y=True)
        randomized_data = tc.tensor(sklearn.preprocessing.scale(breast_cancer))

    elif dataname == 'wine':
        wine,_ = datasets.load_wine(return_X_y=True)
        randomized_data = tc.tensor(sklearn.preprocessing.scale(wine))

    elif dataname == 'four_groups':
        import four_groups
        data = four_groups.art_train
        meanv = data.mean(axis=0)[None,:]
        stdv = data.std(axis=0)[None,:]
        print(meanv.shape, stdv.shape)
        randomized_data = tc.tensor((data - meanv)/stdv)


    elif dataname == 'four_groups_different':
        import four_groups_different
        data, labels = four_groups_different.art_train, four_groups_different.labels
        meanv = data.mean(axis=0)[None,:]
        stdv = data.std(axis=0)[None,:]
        print(meanv.shape, stdv.shape)
        randomized_data, labels = shuffle(((data - meanv)/stdv), labels)
        randomized_data = tc.tensor(randomized_data)

        pandasframe = pd.DataFrame({'labels': labels[randomized_data.size(0)//2:]})
        pandasframe.to_csv('results/singlesample_shapley/labels/labels.csv')


    elif dataname == 'nn_data':
        adjacency, data = Ng.get_data()
        meanv, sdv = data.mean(axis=0, keepdim=True), data.std(axis=0, keepdim=True)
        ordered_data = (data - meanv) / sdv
        randomized_data = ordered_data[tc.randperm(ordered_data.shape[0]),:]


    train_set, test_set = randomized_data[:randomized_data.size(0)//2,:], randomized_data[randomized_data.size(0)//2:,:]

    return train_set, test_set




if __name__ == "__main__":
    data = get_data("dag_nn")
    breakpoint()
