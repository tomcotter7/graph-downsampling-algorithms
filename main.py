# %% Import Statements
import random
import numpy as np
from torch_geometric.datasets import Planetoid
import torch
dataset = Planetoid(root='~/somewhere/Cora', name='Cora')

# %% Initial data setup
data = dataset[0]
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Contains isolated nodes: {data.has_isolated_nodes()}')
print(f'Contains self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')


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
