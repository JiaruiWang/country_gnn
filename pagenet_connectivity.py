# %% Import packages
import pandas as pd
import networkx as nx
import os.path as osp

import torch
from torch_geometric.data import Data, Dataset

# %% 

def get_node(file_path: str = None) -> torch.Tensor:
    num_nodes = 60924683
    x = [i for i in range(num_nodes)]
    
    return x

def get_edge(file_path: str) -> torch.LongTensor:

    df = pd.read_csv(file_path, sep='\t', header=None)
    edge_index = df.to_records(index=False)
    edge = list(edge_index)
    return edge

def get_y_label(file_path: str) -> torch.LongTensor:
    '''Node-level ground-truth labels as 243 country classes.

    Args:
        file_path: A string of file path for the c_label.csv contains country labels.
    Return:
        y: A torch.Tensor with shape (num_nodes).
    '''

    df = pd.read_csv(file_path, sep='\t', header=None)
    y = torch.LongTensor(df.T.values[0])
    return y

def get_masks(file_path: str) -> torch.Tensor:
    '''Get masks for train, validation or test.

    Args:
        file_path: A string of file path for the mask files contains nodes mask.
    Return:
        mask: A torch.Tensor with shape (num_nodes) as True or False for each node. 
    '''

    df = pd.read_csv(file_path, sep='\t', header=None)
    mask = torch.tensor(df.T.values[0], dtype=torch.bool)
    return mask

# %%
idx = 0
# for raw_path in self.raw_paths:
#     # Read data from `raw_path`.
#     # Every sub dir in './data/raw_dir/' for each dataset.

x = get_node()
edge_index = get_edge('./data/raw_dir/edge_index.csv')
# %%
# Create networkx graph
G = nx.Graph()

# y = get_y_label('./data/raw_dir/c_label.csv')

# data = Data(x=x, edge_index=edge_index, y=y)
# data.train_mask = get_masks('./data/raw_dir/c_tr.csv')
# data.val_mask = get_masks('./data/raw_dir/c_va.csv')
# data.test_mask = get_masks('./data/raw_dir/c_te.csv')
# data.num_classes = 243

# data.num_classes = 243

# if self.pre_filter is not None and not self.pre_filter(data):
#     continue

# if self.pre_transform is not None:
#     data = self.pre_transform(data)

# torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
# idx += 1



    


# %% test the dataset 
# root = "data/"
# dataset = PagenetDataset(root)
# data = dataset.get()

# %% examine the graph
# print(f'Dataset: {data}:')
# print('======================')
# print(f'Number of graphs: {len(data)}')
# print(data)
# print('===========================================================================================================')

# Gather some statistics about the graph.
# print(f'Number of nodes: {data.num_nodes}')
# print(f'Number of features: {data.num_features}')
# print(f'Number of classes: {data.num_classes}')
# print(f'Number of edges: {data.num_edges}')
# print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
# print(f'Number of training nodes: {data.train_mask.sum()}')
# print(f'Number of validation nodes: {data.val_mask.sum()}')
# print(f'Number of testing nodes: {data.test_mask.sum()}')
# print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
# print(f'Has isolated nodes: {data.has_isolated_nodes()}')
# print(f'Has self-loops: {data.has_self_loops()}')
# print(f'Is undirected: {data.is_undirected()}')
# %%
