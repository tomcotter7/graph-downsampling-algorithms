# %% Import Statements
import random
import numpy as np
from torch_geometric.datasets import Planetoid
import torch
dataset = Planetoid(root='~/somewhere/Cora', name='Cora')

# %% Initial data setup
data = dataset[0]


# %% Random walk function
def random_walk(graph, sample_size):
    """Function to perform a random walk on the graph and select 'sample_size' nodes
    to subgraph

    Args:
        graph (Data): A torch_geometric Data object,
        sample_size (int): Number of nodes to sample
    """
    x = graph['x']
    # Get the starting node
    starting_node = random.randint(0, len(x))
    current_node = starting_node
    c = 0.15
    sample = {current_node}
    steps = 0
    while len(sample) < sample_size:
        current_node = random.choice(np.where(x[current_node])[0])
        sample.add(current_node)
        steps += 1
        if random.uniform(0, 1) < c:
            current_node = starting_node
        if steps == sample_size * 10:
            starting_node = random.randint(0, len(x))
            current_node = starting_node
    sample = list(sample)
    sample = torch.tensor(sample, dtype=torch.long)
    return sample


subset = random_walk(data, 100)

# %% Run subset
sg = data.subgraph(subset)
