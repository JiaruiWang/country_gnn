# %% Import packages
import pandas as pd
import os.path as osp
import numpy as np

import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset, Data

# %% Define PagenetDataset
class USLGC2DummyDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        '''Initialization.
        Args:
            Root is the root data dir '../../data/'.
        '''

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data = torch.load(osp.join(self.processed_dir, f'us_lgc_mnl_label_2dummy_in_mem_dataset.pt'))
    @property
    def raw_file_names(self):
        '''Get raw data file names in the '../../data/raw_dir/'.
        '''

        return ['us_edges_lgc_relabeled.csv',]

    @property
    def processed_file_names(self):
        '''Generated dataset file names saved in the '../../data/processed_dir/'.
        '''

        return ['us_lgc_mnl_label_2dummy_in_mem_dataset.pt']

    # @property
    # def num_nodes(self):
    # data.num_nodes = x.size(dim=0)

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        idx = 0

        
        us_edge_index = self.get_edge_index('../../data/raw_dir/us_edges_lgc_relabeled.csv')
        us_x = self.get_node_feature('../../data/raw_dir/us_lgc_page_id_name_category_city_likespage_fan_outward_inward.csv')
        us_y = self.get_y_label('../../data/raw_dir/us_pages_lgc_true_label_51_label.csv')
        us_id = self.get_id('../../data/raw_dir/us_pages_lgc_idx_id_mask_label_state.csv')
        self.data = Data(x=us_x, edge_index=us_edge_index, y=us_y)
        self.data.num_classes = 51
        self.data.num_features = 2
        self.data.id = us_id

        torch.save(self.data, osp.join(self.processed_dir, f'us_lgc_mnl_label_2dummy_in_mem_dataset.pt'))

        
    def get_node_feature(self, file_path: str = None) -> torch.Tensor:
        '''Get node feature matrix with shape [num_nodes, num_node_features].

        Args: 
            file_path: A string of File path for the x_features.csv file.
        Return:
            x: A torch.Tensor with shape (num_nodes, num_node_features).
        '''
            
        # num_nodes = 60924683


        # 2. Add position anchor sets, node features are the distances to the sets.
        df = pd.read_csv(file_path, sep='\t', header=None)
        out_degree = torch.Tensor(df.iloc[:, 6:7].values)
        in_degree = torch.Tensor(df.iloc[:, 7:8].values)
        out_min, out_max, in_min, in_max = out_degree.min(), out_degree.max(), in_degree.min(), in_degree.max()
        out_degree_norm = (out_degree - out_min) / (out_max - out_min)
        in_degree_norm = (in_degree - in_min) / (in_max - in_min)
        degrees = torch.concat((out_degree_norm, in_degree_norm), 1)


        x = degrees.new_ones(degrees.shape)
        print(x.shape)
        print(x[0:10])
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

    def get_y_label(self, file_path: str) -> torch.LongTensor:
        '''Node-level ground-truth labels as 243 country classes.

        Args:
            file_path: A string of file path for the c_label.csv contains country labels.
        Return:
            y: A torch.Tensor with shape (num_nodes).
        '''

        df = pd.read_csv(file_path, sep='\t', header=None)
        df = df.iloc[:, 1:2]
        # df.to_csv('./data/raw_dir/us_pages_lgc_with_new_label.csv', sep='\t', index=True,
        #           header=False)
        y = torch.LongTensor(df.T.values[0])
        print(y.shape)
        print(y[0:10])
    
        return y

    def get_id(self, file_path: str) -> torch.LongTensor:
        '''Get id for each page.

        Args:
            file_path: A string of file path for the us_pages_lgc_idx_id_mask_label_state.csv
            contains country labels.
        Return:
            id: A torch.Tensor with shape (num_nodes).
        '''

        df = pd.read_csv(file_path, sep='\t', header=None)
        df = df.iloc[:, 1:2]
        # df.to_csv('./data/raw_dir/us_pages_lgc_with_new_label.csv', sep='\t', index=True,
        #           header=False)
        id = torch.LongTensor(df.T.values[0])
        print(id.shape)
        print(id[0:10])
    
        return id

    # def get_masks(self, file_path: str) -> torch.Tensor:
    #     '''Get masks for train, validation or test.

    #     Args:
    #         file_path: A string of file path for the mask files contains nodes mask.
    #     Return:
    #         mask: A torch.Tensor with shape (num_nodes) as True or False for each node. 
    #     '''

    #     df = pd.read_csv(file_path, sep='\t', header=None)
    #     mask = torch.tensor(df.T.values[0], dtype=torch.bool)
    #     return mask


# %% test the dataset 
root = "../../data"
dataset = USLGC2DummyDataset(root)


# %%
