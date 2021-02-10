import numpy as np


class LinearGaussian:
    def __init__(
            self,
            dim: int = 5,
            sigma_graph: float = 0.5,
            sigma_noise: float = 0.1,
            drop_out: float = 0.5,
            directed_acyclic: bool = True):

        if sigma_graph < 0:
            raise ValueError(f"sigma_edges must be bigger 0, got {sigma_graph}")
        if sigma_noise < 0:
            raise ValueError(f"sigma_nodes must be bigger 0, got {sigma_noise}")
        if drop_out < 0 or drop_out > 1:
            raise ValueError(f"drop_out must be in [0, 1], got {drop_out}")

        self.d_dims = dim
        self.sigma_graph = sigma_graph
        self.sigma_noise = sigma_noise
        self.drop_out = drop_out
        self.directed_acyclic = directed_acyclic

        # initialize fully connected graph
        node_variance = np.random.randn(dim, dim) * sigma_graph
        # initialize mask of elementwise bernoulli(p=drop_out)
        mask = np.random.uniform(size=(dim, dim)) > drop_out

        if directed_acyclic:  # ensure dag
            # constrain to directed graph by setting all upper triangular to zero
            mask = mask * np.tril(np.ones_like(mask))
        else:  # ensure cyclic graph
            node_variance = self.symmetrize(node_variance)
            mask = self.symmetrize(mask)

        # get adjacency matrix
        self.adjacency = mask * node_variance

    def symmetrize(self, x):
        return np.tril(x) + np.tril(x).T + np.diag(np.diag(x))

    def sample(self, n_samples=500):
        # initialize independent samples (noise terms)
        sample_variance = np.random.randn(self.d_dims, n_samples) * self.sigma_noise
        observation = self.adjacency @ sample_variance
        return observation



if __name__ == "__main__":
    lg = LinearGaussian(dim=5, directed_acyclic=False)
    x = lg.sample(500)

    print(lg.adjacency)
    print(x)
