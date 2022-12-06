# %% Import packages
import pandas as pd
import os.path as osp
import numpy as np
import csv

import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset, Data

# %% Define PagenetDataset
class USLGCNewcityTrueLableNoCentersDistributionFeaturesWithIdDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        '''Initialization.
        Args:
            Root is the root data dir '../../data/'.
        '''

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data = torch.load(osp.join(self.processed_dir, f'us_lgc_newcity_truelabel_no_centers_48by6_neighbor_distribution_with_id_in_mem_dataset.pt'))
    @property
    def raw_file_names(self):
        '''Get raw data file names in the '../../data/raw_dir/'.
        '''

        return ['./data/raw_dir/us_edges_lgc_relabeled.csv', 
                './new_city_data/new_cities_2hop_bfs_2_round_id_idx_city_dupStates_trueLabel_mostPopulationLabel_2ndRoundDupstate0_1stRoundDupstate0.csv', 
                './new_city_data/population_label_51_by_6_neighbor_distribution.csv']

    @property
    def processed_file_names(self):
        '''Generated dataset file names saved in the '../../data/processed_dir/'.
        '''

        return ['us_lgc_newcity_truelabel_no_centers_48by6_neighbor_distribution_with_id_in_mem_dataset.pt']

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
        
        us_edge_index = self.get_edge_index('../../data/raw_dir/us_edges_lgc_relabeled.csv')
        us_x = self.get_node_feature('../../new_city_data/new_cities_true_label_no_centers_48_by_6_neighbor_distribution.csv')
        y_has_centers_51, labeled_mask, y_no_centers_48, labeled_mask_no_centers = self.get_y_label('../../new_city_data/new_cities_2hop_bfs_2_round_pl_id_idx_city_dupStates_trueLabel_mostPopulationLabel_2ndRoundDupstate0_1stRoundDupstate0.csv')
        us_id = self.get_id('../../new_city_data/new_cities_2hop_bfs_2_round_pl_id_idx_city_dupStates_trueLabel_mostPopulationLabel_2ndRoundDupstate0_1stRoundDupstate0.csv')
        self.data = Data(x=us_x, edge_index=us_edge_index, y=y_no_centers_48)
        self.data.num_classes = 48
        self.data.id = us_id
        self.data.y_has_centers_51 = y_has_centers_51
        self.data.labeled_mask_has_centers_51 = labeled_mask
        self.data.labeled_mask_no_centers_48 = labeled_mask_no_centers
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
        torch.save(self.data, osp.join(self.processed_dir, f'us_lgc_newcity_truelabel_no_centers_48by6_neighbor_distribution_with_id_in_mem_dataset.pt'))

        
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
        df = pd.read_csv(file_path, sep=',', header=None)
        x = torch.Tensor(df.values)
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

    def get_y_label(self, file_path: str):
        '''Node-level ground-truth labels as 243 country classes.

        Args:
            file_path: A string of file path for the c_label.csv contains country labels.
        Return:
            y: A torch.Tensor with shape (num_nodes).
        '''
        '''
        us_y = self.get_y_label('./data/raw_dir/us_pages_lgc_true_label_51_label.csv')
        4	4
        4	4
        32	31
        47	45
        32	31
        -1	4
        -1	4
        -1	31
        -1	4
        4	4
        us_y = self.get_y_label('./new_city_data/new_cities_2hop_bfs_2_round_id_idx_city_dupStates_trueLabel_mostPopulationLabel_2ndRoundDupstate0_1stRoundDupstate0.csv')
        4846711747,0,san francisco,1,4,4,4,4
        5246634919,20,redwood city,1,4,4,4,4
        5281959998,23,new york,1,31,31,31,31
        5340039981,32,virginia beach,1,45,45,45,45
        5352734382,34,new york,1,31,31,31,31
        5381334667,42,oakland,16,-1,4,4,4
        5417674986,53,washington,18,-1,50,50,50
        5459524943,77,syracuse,7,-1,31,31,31
        5461604986,78,west valley city,1,43,43,43,43
        5466504237,81,inglewood,2,-1,4,4,4
        '''
        states_no_centers_name_idx = {}
        no_centers_states_df = pd.read_csv('../../new_city_data/states_no_centers.csv', sep=',', header=None)
        no_centers_states = no_centers_states_df.values
        for i in range(len(no_centers_states)):
            states_no_centers_name_idx[no_centers_states[i][0]] = i
        print(states_no_centers_name_idx)

        states_idx_name = {}
        states_df = pd.read_csv('../../new_city_data/states.csv', sep=',', header=None)
        states = states_df.values     
        for i in range(len(states)):
            states_idx_name[i] = states[i][0]
        print(states_idx_name)   

        df = pd.read_csv(file_path, sep=',', header=None)
        trueLabel_df = df.iloc[:, 4:5]
        city_df = df.iloc[:, 2:3]
        count_washington = 0
        count_true_label = 0
        
        city = city_df.values
        trueLabel = trueLabel_df.values
        print('trueLabel[0:10]', trueLabel[0:10])
        print('city[0:10]', city[0:10])
        for i in range(len(city)):
            # print(city_df.values[i][0], trueLabel_df.values[i][0])
            if city[i][0] == 'washington':
                trueLabel[i][0] = 50
                count_washington += 1
            if trueLabel[i][0] != -1:
                count_true_label += 1
        print('count_washington',count_washington, 'count_true_label',count_true_label)
        # df.to_csv('./data/raw_dir/us_pages_lgc_with_new_label.csv', sep='\t', index=True,
        #           header=False)
        y = torch.LongTensor(trueLabel.T[0])

        labeled_mask = (y != -1)
        print('y.shape',y.shape)
        print('y[0:10]',y[0:10])
        print('labeled_mask.shape',labeled_mask.shape)
        print('labeled_mask[0:10]',labeled_mask[0:10])
        print('labeled_mask.sum()', labeled_mask.sum())

        count_centers = 0
        count_non_centers = 0
        for i in range(len(trueLabel)):
            # print(city_df.values[i][0], trueLabel_df.values[i][0])
            if trueLabel[i][0] in [4, 31, 42]:
                trueLabel[i][0] = -1
                count_centers += 1
            if trueLabel[i][0] != -1:
                trueLabel[i][0] = states_no_centers_name_idx[states_idx_name[trueLabel[i][0]]]
                count_non_centers += 1
        print('count_centers',count_centers, 'count_non_centers',count_non_centers)
        y_no_centers = torch.LongTensor(trueLabel.T[0])
        labeled_mask_no_centers = (y_no_centers != -1)
        print('y_no_centers.shape',y_no_centers.shape)
        print('y_no_centers[0:10]',y_no_centers[0:10])
        print('labeled_mask_no_centers.shape',labeled_mask_no_centers.shape)
        print('labeled_mask_no_centers[0:10]',labeled_mask_no_centers[0:10])
        print('labeled_mask_no_centers.sum()', labeled_mask_no_centers.sum())
    
        return (y, labeled_mask, y_no_centers, labeled_mask_no_centers)

    def get_id(self, file_path: str) -> torch.LongTensor:
        '''Get id for each page.

        Args:
            file_path: A string of file path for the us_pages_lgc_idx_id_mask_label_state.csv
            contains country labels.
        Return:
            id: A torch.Tensor with shape (num_nodes).
        '''
        '''
        us_id = self.get_id('./data/raw_dir/us_pages_lgc_idx_id_mask_label_state.csv')
        0	4846711747	1	4	California
        1	5246634919	1	4	California
        2	5281959998	1	31	New York
        3	5340039981	1	45	Virginia
        4	5352734382	1	31	New York
        5	5381334667	0	4	California
        6	5417674986	0	4	California
        7	5459524943	0	31	New York
        8	5461604986	0	4	California
        9	5466504237	1	4	California
        us_id = self.get_id('./new_city_data/new_cities_2hop_bfs_2_round_id_idx_city_dupStates_trueLabel_mostPopulationLabel_2ndRoundDupstate0_1stRoundDupstate0.csv')
        4846711747,0,san francisco,1,4,4,4,4
        5246634919,20,redwood city,1,4,4,4,4
        5281959998,23,new york,1,31,31,31,31
        5340039981,32,virginia beach,1,45,45,45,45
        5352734382,34,new york,1,31,31,31,31
        5381334667,42,oakland,16,-1,4,4,4
        5417674986,53,washington,18,-1,50,50,50
        5459524943,77,syracuse,7,-1,31,31,31
        5461604986,78,west valley city,1,43,43,43,43
        5466504237,81,inglewood,2,-1,4,4,4
        '''

        df = pd.read_csv(file_path, sep=',', header=None)
        df = df.iloc[:, 0:1]
        # df.to_csv('./data/raw_dir/us_pages_lgc_with_new_label.csv', sep='\t', index=True,
        #           header=False)
        id = torch.LongTensor(df.T.values[0])
        print(id.shape)
        print(id[0:10])
    
        return id

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
root = "../../data/"
dataset = USLGCNewcityTrueLableNoCentersDistributionFeaturesWithIdDataset(root)
print(dataset[0])

# %%
