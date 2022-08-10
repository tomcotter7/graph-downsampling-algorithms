# %% Import Statements
import random
import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric import utils
import torch
import networkx as nx
dataset = Planetoid(root='data/Cora', name='Cora')

# %% Initial data setup
data = dataset[0]

g = utils.to_networkx(data, to_undirected=True)
nx.draw(g)


# %% Random walk function
def random_walk(graph, sample_size):
    """Function to perform a random walk on the graph and select 'sample_size' nodes
    to subgraph

    Args:
        graph (Data): A torch_geometric Data object,
        sample_size (int): Number of nodes to sample

    Returns:
        LongTensor: A LongTensor of nodes to include in the subraph
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

# %% Draw subset
g = utils.to_networkx(sg, to_undirected=True)
nx.draw(g)
