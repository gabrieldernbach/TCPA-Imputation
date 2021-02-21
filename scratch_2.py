from networkx.generators.random_graphs import gnp_random_graph
from networkx.algorithms.dag import topological_sort

G = gnp_random_graph(n=16, p=0.1, directed=True)
gi = topological_sort(G)
breakpoint()
