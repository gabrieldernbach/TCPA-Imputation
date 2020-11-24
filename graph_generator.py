import numpy as np


def LinearGaussian(n_samples: int = 500,
                   d_dims: int = 5,
                   sigma_graph: float = 0.5,
                   sigma_noise: float = 0.1,
                   drop_out: float = 0.5,
                   directed: bool = True,
                   ):
    """
    Generate linear gaussian model.
    """
    if sigma_graph < 0:
        raise ValueError(f"sigma_graph must be bigger 0, got {sigma_graph}")
    if sigma_noise < 0:
        raise ValueError(f"sigma_noise must be bigger 0, got {sigma_noise}")
    if drop_out < 0 or drop_out > 1:
        raise ValueError(f"drop_out mus be in [0, 1], got {drop_out}")

    # initialize fully connected graph
    G = np.random.randn(d_dims, d_dims) * sigma_graph
    # generate independent samples (noise terms)
    S = np.random.randn(d_dims, n_samples) * sigma_noise

    # mask with elementwise bernoulli
    M = np.random.uniform(size=(d_dims, d_dims)) > drop_out
    # remove upper triangle if a directed graph is requested
    M = np.tril(M) if directed else M

    # get adjacency matrix
    A = M * G
    # transform independent samples by the graph dependency
    X = A @ S

    return A, X


print(LinearGaussian())
