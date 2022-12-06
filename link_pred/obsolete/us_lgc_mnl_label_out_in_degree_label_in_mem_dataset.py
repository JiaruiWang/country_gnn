# %% Import packages
import pandas as pd
import os.path as osp
import numpy as np

import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset, Data

# %% Define PagenetDataset
class USLGCDegreeLabelDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        '''Initialization.
        Args:
            Root is the root data dir '../data/'.
        '''

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data = torch.load(osp.join(self.processed_dir, f'us_lgc_mnl_label_out_in_degree_label_in_mem_dataset.pt'))
    @property
    def raw_file_names(self):
        '''Get raw data file names in the '../data/raw_dir/'.
        '''

        return ['us_edges_lgc_relabeled.csv',]

    @property
    def processed_file_names(self):
        '''Generated dataset file names saved in the '../data/processed_dir/'.
        '''

        return ['us_lgc_mnl_label_out_in_degree_label_in_mem_dataset.pt']

    # @property
    # def num_nodes(self):
    # data.num_nodes = x.size(dim=0)

    def download(self):
        # Download to `self.raw_dir`.
        pass
        # path = download_url(url, self.raw_dir)
        # ...

    # def len(self):
    #     return len(self.processed_file_names)

    # def get(self, idx=None):
    #     data = torch.load(osp.join(self.processed_dir, f'page_net_s_100k_memory_node_feat_1.pt'))
    #     return data

    def process(self):
        idx = 0
        # for raw_path in self.raw_paths:
        #     # Read data from `raw_path`.
        #     # Every sub dir in './data/raw_dir/' for each dataset.

        # world_edge_index = self.get_edge_index('./data/raw_dir/edge_index.csv')
        # us_lgc_mask_in_world = self.get_masks('./data/raw_dir/us_pages_lgc_mask_in_world.csv')
        # print(us_lgc_mask_in_world.shape)
        # us_edge_index, attr = torch_geometric.utils.subgraph(subset=us_lgc_mask_in_world, 
        #                                                      edge_index=world_edge_index,
        #                                                      relabel_nodes=True)
        # edge = us_edge_index.numpy().T
        # print(edge.shape, type(edge))
        # print(edge)
        # np.savetxt("./data/raw_dir/us_edges_lgc_relabeled.csv", edge, delimiter='\t',fmt='%i')
        
        us_edge_index = self.get_edge_index('../data/raw_dir/us_edges_lgc_relabeled.csv')
        us_x = self.get_node_feature('../data/raw_dir/us_lgc_page_id_name_category_city_likespage_fan_outward_inward.csv')
        us_y = self.get_y_label('../data/raw_dir/us_pages_lgc_true_label_51_label.csv')
        # us_id = self.get_id('../data/raw_dir/us_pages_lgc_idx_id_mask_label_state.csv')
        self.data = Data(x=us_x, edge_index=us_edge_index, y=us_y)
        self.data.num_classes = 51
        self.data.num_features = 53
        # self.data.id = us_id
        # if self.pre_filter is not None and not self.pre_filter(data):
        #     continue

        # if self.pre_transform is not None:
        #     data = self.pre_transform(data)

        # torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
        # idx += 1

        # data, slices = self.collate(data_list)
        # print("In process data and slices:")
        # print(data)
        # print(slices)
        torch.save(self.data, osp.join(self.processed_dir, f'us_lgc_mnl_label_out_in_degree_label_in_mem_dataset.pt'))

        
    def get_node_feature(self, file_path: str = None) -> torch.Tensor:
        '''Get node feature matrix with shape [num_nodes, num_node_features].

        Args: 
            file_path: A string of File path for the x_features.csv file.
        Return:
            x: A torch.Tensor with shape (num_nodes, num_node_features).
        '''
            
        # num_nodes = 60924683

        # # 1. Just add constant value 1 for each node as feature augmentation.
        # x = [[1] for i in range(num_nodes)]
        # x = torch.Tensor(x)

        # 2. Add position anchor sets, node features are the distances to the sets.
        df = pd.read_csv(file_path, sep='\t', header=None)
        out_degree = torch.Tensor(df.iloc[:, 6:7].values)
        in_degree = torch.Tensor(df.iloc[:, 7:8].values)
        out_min, out_max, in_min, in_max = out_degree.min(), out_degree.max(), in_degree.min(), in_degree.max()
        out_degree_norm = (out_degree - out_min) / (out_max - out_min)
        in_degree_norm = (in_degree - in_min) / (in_max - in_min)
        degrees = torch.concat((out_degree_norm, in_degree_norm), 1)
        print(degrees.shape, degrees[0:10])
        us_y = self.get_y_label('../data/raw_dir/us_pages_lgc_true_label_51_label.csv')
        one_hot_y = torch.nn.functional.one_hot(us_y)
        print(one_hot_y.shape, one_hot_y[0:10])

        x = torch.concat((degrees, one_hot_y), 1)
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

    # def get_id(self, file_path: str) -> torch.LongTensor:
    #     '''Get id for each page.

    #     Args:
    #         file_path: A string of file path for the us_pages_lgc_idx_id_mask_label_state.csv
    #         contains country labels.
    #     Return:
    #         id: A torch.Tensor with shape (num_nodes).
    #     '''

    #     df = pd.read_csv(file_path, sep='\t', header=None)
    #     df = df.iloc[:, 1:2]
    #     # df.to_csv('./data/raw_dir/us_pages_lgc_with_new_label.csv', sep='\t', index=True,
    #     #           header=False)
    #     id = torch.LongTensor(df.T.values[0])
    #     print(id.shape)
    #     print(id[0:10])
    
    #     return id

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
root = "../data/"
dataset = USLGCDegreeLabelDataset(root)


# # # %% examine the graph
# print(f'Dataset: {dataset}:')
# print('======================')
# print(f'Number of graphs: {len(dataset)}')
# print(dataset)
# print('===========================================================================================================')
# data = dataset.data
# print(data)

# # Gather some statistics about the graph.
# print(f'Number of nodes: {data.num_nodes}')
# print(f'Number of features: {data.num_features}')
# print(f'Number of classes: {data.num_classes}')
# print(f'Number of edges: {data.num_edges}')
# print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')

# print(f'Has isolated nodes: {data.has_isolated_nodes()}') #false
# print(f'Has self-loops: {data.has_self_loops()}') #true
# print(f'Is undirected: {data.is_undirected()}') #false
# # print(data.x[0:5])
# # print(data.y[0:5])
# # %%
# transform = torch_geometric.transforms.RandomNodeSplit(split='random',
#                                            num_train_per_class=100000,
#                                            num_val=0.05,
#                                            num_test=0.05)
# data = transform(data)
# print(data)
# print(f'Number of training nodes: {data.train_mask.sum()}')
# print(f'Number of validation nodes: {data.val_mask.sum()}')
# print(f'Number of testing nodes: {data.test_mask.sum()}')
# print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
# loader = torch_geometric.loader.NeighborLoader(
#     data,
#     # Sample 30 neighbors for each node for 2 iterations
#     num_neighbors=[10] * 2,
#     # Use a batch size of 128 for sampling training nodes
#     batch_size=128,
#     input_nodes=data.train_mask,
# )

# count = 0
# for data in loader:
#     print(count)
#     print(data) #  Output data by batch 
#     count += 1
#     if count == 2: break

# %%
