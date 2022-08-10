# %% Imports
import random
import numpy as np
from torch_geometric.datasets import Planetoid
import torch
dataset = Planetoid(root='data/Cora', name='Cora')

# %% Initial data set-up
data = dataset[0]


# %% Forest Fire Algorithm
def forest_fire(graph, sample_size):
    """Function to run the forest fire algorithm on a graph to determine list of nodes to create a subgraph from.

    Args:
        graph (Data): A torch_geometric Data object,
        sample_size (int): Number of nodes to sample

    Returns:

        LongTensor: A LongTensor of nodes to include in the subgraph
    """
    pass

subset = forest_fire(data, 1)

# %% Run subset

sg = data.subgraph(subset)
