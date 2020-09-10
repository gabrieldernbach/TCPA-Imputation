import torch
from torch_geometric.data import Data


graph = np.genfromtxt('../data/adj_tcpa_full_degmin4.csv', delimiter = "\t", dtype = 'str')
data_for_graph = np.genfromtxt('../data2/TCPA_data_sel.csv', delimiter = ',', dtype = 'str')
names, dataGCN, graphGCN = get_data_and_graph(data_for_graph, graph)

edge_list = [[i.shape[0], i.shape[1]] for i in graphGCN if i!=0]
print(len(edge_list))


class GraphNet():

    pass