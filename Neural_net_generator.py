import torch
import torch.nn as nn
from tqdm import tqdm
import pdb
from linear_gaussian_generator import LinearGaussian


class LinSkip(nn.Module):
    def __init__(self, ins, outs, act="linear"):
        super(LinSkip, self).__init__()
        self.skip = ins == outs
        self.act = nn.ModuleDict({
            "linear": nn.Identity(),
            "lrelu": nn.LeakyReLU(),
        })[act]

        self.layer = nn.Sequential(
            nn.Linear(ins, 4 * outs),
            self.act,
            nn.Linear(4 * outs, outs),
        )

    def forward(self, x):
        z = self.layer(x)
        x = x + z if self.skip else z
        return x


class Node(nn.Module):
    def __init__(self, ins: int, adjacency: torch.tensor, noise: float):
        super(Node, self).__init__()
        dim = 256
        self.layer = nn.Sequential(
            LinSkip(ins, dim, "lrelu"),
            LinSkip(dim, dim, "lrelu"),
            LinSkip(dim, dim, "lrelu"),
            LinSkip(dim, 1, "linear"),
        )
        self.listen = adjacency  # boolean mask indicating who to listen to
        self.noise = noise  # scaling uniform output noise

    def forward(self, x):
        x_ = self.listen.float() * x
        o = self.layer(x_)
        o += (torch.rand(o.shape) * 2 - 1) * self.noise
        return o


def sample_graph(nodes, n_samples):
    observation = torch.zeros(n_samples, len(nodes))
    for i, node in enumerate(tqdm(nodes)):
        observation[:, i] = node(observation).squeeze(-1)
    return observation


def get_data(seed=0):
    torch.manual_seed(seed)
    # todo:
    #   limit the graph to a certain degree or use generators from networkx
    #   https://networkx.org/documentation/stable/reference/generators.html
    #   for an overview of practically relevant graphs
    #   https://epubs.siam.org/doi/pdf/10.1137/S003614450342480?xid=PS_smithsonian&
    #   for now, use the linear gaussian generator
    dim = 32
    n_samples = 8000
    lg = LinearGaussian(dim=dim, directed_acyclic=True)

    # create topologically ordered network
    adjacency = torch.tensor(lg.adjacency != 0)
    noise = torch.randn(dim) ** 2
    with torch.no_grad():
        nodes = [Node(dim, a, n) for a, n in zip(adjacency, noise)]
        observations = sample_graph(nodes, n_samples)

    #torch.save((adjacency, observations), f"dag_nn_{i}.pt")

    return adjacency, observations