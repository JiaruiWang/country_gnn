#%%
# import data
import os.path as osp
from copy import deepcopy

import torch, tqdm
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

from torch_sparse import SparseTensor
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import Linear, SAGEConv, GraphSAGE, MLP, GraphConv
from torch_geometric.utils import negative_sampling, degree
from torch_geometric.loader import LinkNeighborLoader, GraphSAINTRandomWalkSampler

from us_lgc_mnl_label_degree_in_mem_dataset import USLGCDegreeDataset
EPS = 1e-15
#%%
# define link split
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = T.Compose([
    # T.NormalizeFeatures(), # row normalization
    # T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.001, num_test=0.001, is_undirected=False,
                    #   disjoint_train_ratio=0.3,
                      add_negative_train_samples=False, neg_sampling_ratio=2),
])
# %% 
# Load dataset
dataset_all = USLGCDegreeDataset("../../data")
print(dataset_all)
print(type(dataset_all))

#%%
dataset = USLGCDegreeDataset("../../data", transform=transform)
# examine the graph
print(f'Dataset: {dataset}:')
print(type(dataset))
print('======================')
print(f'Number of Graph: {len(dataset)}')

print('===============================================================================')
# After applying the `RandomLinkSplit` transform, the data is transformed from
# a data object to a list of tuples (train_data, val_data, test_data), with
# each element representing the corresponding split.

# train_data, val_data, test_data = dataset[0]

data = dataset_all.data
row, col = data.edge_index
data.edge_weight = 1. / degree(col, data.num_nodes)[col]  # Norm by in-degree.
use_normalization = True
action='store_true'

train_data, val_data, test_data = transform(data)
print(type(train_data),train_data)
print(val_data)
print(test_data)
#%%
print(type(dataset))

print(type(val_data))

test_data_loader = LinkNeighborLoader(test_data,
                                      num_neighbors=[-1, -1],
                                      batch_size = 4096,
                                      edge_label_index=test_data.edge_label_index,
                                      edge_label=test_data.edge_label,
                                      neg_sampling_ratio=0)

print(type(test_data_loader))


#%%
    
class Net(torch.nn.Module):
    def __init__(self, in_channels,hidden_channels, out_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.lin =  Linear(2 * hidden_channels, hidden_channels)
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, out_channels)

    def set_aggr(self, aggr):
        self.conv1.aggr = aggr
        self.conv2.aggr = aggr


    def encode(self, x0, edge_index, edge_weight=None) -> torch.Tensor:
        x1 = F.relu(self.conv1(x0, edge_index, edge_weight))
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x2 = F.relu(self.conv2(x1, edge_index, edge_weight))
        x2 = F.dropout(x2, p=0.2, training=self.training)
        x = torch.cat([x1, x2], dim=-1)
        x = self.lin(x)
        return x

    def decode(self, z, edge_label_index) -> torch.Tensor:
        x = torch.cat((z[edge_label_index[0]], z[edge_label_index[1]]), dim=-1)
        x = self.lin1(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, p=0.5, training=self.training)
        out = self.lin3(x)
        return out

    # def decode_all(self, z):
    #     prob_adj = z @ z.t()
    #     return (prob_adj > 0).nonzero(as_tuple=False).t()       
net = Net(2, 64, 1).to(device)
# model = GraphSAGE(dataset_all.num_features, hidden_channels=64,
#                   num_layers=2).to(device)
# decoder = MLP([64, 128, 64, 16, 1], dropout=0.5)
optimizer = torch.optim.Adam(params=net.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()

#%%
# load model
net = Net(2, 64, 1).to(device)
PATH = './mnl_label_link_pred_saint_degree_epoch_500.pt'
net.load_state_dict(torch.load(PATH))
net.to(device)

# %%
total_pred = []
total_edge_label = []
total_edge_start_id = []
total_edge_end_id = []
total_edge_start_id_state_label = []
total_edge_end_id_state_label = []
total_edge_start_degree_weights = []
total_edge_end_degree_weights = []
@torch.no_grad()
def infer():
    net.eval()
    for batch in tqdm.tqdm(test_data_loader):
        # print("batch ",batch)
        total_edge_start_id.extend( batch.id[batch.edge_label_index[0]].tolist())
        total_edge_end_id.extend(batch.id[batch.edge_label_index[1]].tolist())
        total_edge_start_id_state_label.extend(batch.y[batch.edge_label_index[0]].tolist())
        total_edge_end_id_state_label.extend(batch.y[batch.edge_label_index[1]].tolist())
        total_edge_start_degree_weights.extend(batch.x[batch.edge_label_index[0]].tolist())
        total_edge_end_degree_weights.extend(batch.x[batch.edge_label_index[1]].tolist())
        total_edge_label.extend(batch.edge_label.tolist())

        batch = batch.to(device)
        z = net.encode(batch.x, batch.edge_index)
        # print("z ",z.shape)
        out = net.decode(z, batch.edge_label_index)       
        total_pred.extend(out.cpu().tolist())
    

# return roc_auc_score(total_edge_label, total_pred)
# %%
infer()
print(len(total_pred), len(total_edge_label), 
        len(total_edge_start_id), len(total_edge_end_id),
        len(total_edge_start_id_state_label), len(total_edge_end_id_state_label),
        len(total_edge_start_degree_weights), len(total_edge_start_degree_weights[0]),
        len(total_edge_end_degree_weights), len(total_edge_end_degree_weights[0])
        )
#%%