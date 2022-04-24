# %% Import packages
import pandas as pd
import os.path as osp

import torch
from torch_geometric.data import Data, Dataset

# %% Define PagenetDataset
class PagenetDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        '''Initialization.
        Args:
            Root is the root data dir './data/'.
        '''

        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        '''Get raw data file names in the './data/raw_dir/'.
        '''

        return ['edge_index.csv', 'c_label.csv', 'c_tr.csv', 'c_va.csv', 'c_te.csv']

    @property
    def processed_file_names(self):
        '''Generated dataset file names saved in the './data/processed_dir/'.
        '''

        return ['page_net_total_disk_node_feat_1.pt']

    # @property
    # def num_nodes(self):
    # data.num_nodes = x.size(dim=0)

    def download(self):
        # Download to `self.raw_dir`.
        pass
        # path = download_url(url, self.raw_dir)
        # ...

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx=None):
        data = torch.load(osp.join(self.processed_dir, f'page_net_total_disk_node_feat_1.pt'))
        return data

    def process(self):
        idx = 0
        # for raw_path in self.raw_paths:
        #     # Read data from `raw_path`.
        #     # Every sub dir in './data/raw_dir/' for each dataset.

        x = self.get_node_feature()
        edge_index = self.get_edge_index('./data/raw_dir/edge_index.csv')
        y = self.get_y_label('./data/raw_dir/c_label.csv')

        self.data = Data(x=x, edge_index=edge_index, y=y)
        self.data.train_mask = self.get_masks('./data/raw_dir/c_tr.csv')
        self.data.val_mask = self.get_masks('./data/raw_dir/c_va.csv')
        self.data.test_mask = self.get_masks('./data/raw_dir/c_te.csv')
        self.data.num_classes = 243
        
        # data.num_classes = 243

        # if self.pre_filter is not None and not self.pre_filter(data):
        #     continue

        # if self.pre_transform is not None:
        #     data = self.pre_transform(data)

        # torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
        # idx += 1

        torch.save(self.data, osp.join(self.processed_dir, f'page_net_total_disk_node_feat_1.pt'))

        
    def get_node_feature(self, file_path: str = None) -> torch.Tensor:
        '''Get node feature matrix with shape [num_nodes, num_node_features].

        Args: 
            file_path: A string of File path for the x_features.csv file.
        Return:
            x: A torch.Tensor with shape (num_nodes, num_node_features).
        '''
            
        num_nodes = 60924683

        # 1. Just add constant value 1 for each node as feature augmentation.
        x = [[1] for i in range(num_nodes)]
        x = torch.Tensor(x)

        # 2. Add position anchor sets, node features are the distances to the sets.
        
        return x
    
    def get_edge_index(self, file_path: str) -> torch.LongTensor:
        '''Graph connectivity in COO format with shape [2, num_edges].

        Args:
            file_path: A string of File path for the edge_index.csv file.
        Return:
            edge_index: A torch.LongTensor with shape (2, num_edges).
        '''

        df = pd.read_csv(file_path, sep='\t', header=None)
        dft = df.T
        edge_index = torch.LongTensor(dft.values)
        return edge_index

    def get_y_label(self, file_path: str) -> torch.Tensor:
        '''Node-level ground-truth labels as 243 country classes.

        Args:
            file_path: A string of file path for the c_label.csv contains country labels.
        Return:
            y: A torch.Tensor with shape (num_nodes).
        '''

        df = pd.read_csv(file_path, sep='\t', header=None)
        y = torch.Tensor(df.T.values[0])
        return y

    def get_masks(self, file_path: str) -> torch.Tensor:
        '''Get masks for train, validation or test.

        Args:
            file_path: A string of file path for the mask files contains nodes mask.
        Return:
            mask: A torch.Tensor with shape (num_nodes) as True or False for each node. 
        '''

        df = pd.read_csv(file_path, sep='\t', header=None)
        mask = torch.tensor(df.T.values[0], dtype=torch.bool)
        return mask


# %% test the dataset 
root = "data/"
dataset = PagenetDataset(root)
data = dataset.get()

# %% examine the graph
print(f'Dataset: {data}:')
print('======================')
print(f'Number of graphs: {len(data)}')
print(data)
print('===========================================================================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of features: {data.num_features}')
print(f'Number of classes: {data.num_classes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Number of validation nodes: {data.val_mask.sum()}')
print(f'Number of testing nodes: {data.test_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
# print(f'Has isolated nodes: {data.has_isolated_nodes()}')
# print(f'Has self-loops: {data.has_self_loops()}')
# print(f'Is undirected: {data.is_undirected()}')
# %%
