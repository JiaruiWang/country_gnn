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
from torch_geometric.nn import Linear, SAGEConv, GraphSAGE, MLP
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import LinkNeighborLoader

from us_lgc_mnl_label_out_in_degree_in_mem_dataset import USLGCDegreeDataset
EPS = 1e-15
#%%
# define link split
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = T.Compose([
    # T.NormalizeFeatures(), # row normalization
    # T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.001, num_test=0.001, is_undirected=False,
                      disjoint_train_ratio=0.3,
                      add_negative_train_samples=True, neg_sampling_ratio=1),
])
# %% 
# Load dataset
dataset_all = USLGCDegreeDataset("../data")
print(dataset_all)
print(type(dataset_all))
# print(f'Number of nodes: {dataset_all.num_nodes}')
# print(f'Number of features: {dataset_all.num_features}')
# print(f'Number of classes: {dataset_all.num_classes}')
# print(f'Number of edges: {dataset_all.num_edges}')
# print(f'Number of edges: {dataset_all[0].edge_label}')
#%%
dataset = USLGCDegreeDataset("../data/", transform=transform)
# examine the graph
print(f'Dataset: {dataset}:')
print(type(dataset))
print('======================')
print(f'Number of Graph: {len(dataset)}')

print('===============================================================================')
# After applying the `RandomLinkSplit` transform, the data is transformed from
# a data object to a list of tuples (train_data, val_data, test_data), with
# each element representing the corresponding split.
train_data, val_data, test_data = dataset[0]
print(train_data)
print(val_data)
print(test_data)
#%%
print(type(dataset))

print(type(val_data))
#%%
train_data_loader = LinkNeighborLoader( train_data,
                                        num_neighbors=[20, 10],
                                        batch_size = 1024000,
                                        # shuffle=True,
                                        edge_label_index=train_data.edge_label_index,
                                        edge_label=train_data.edge_label,
                                        neg_sampling_ratio=0)
val_data_loader = LinkNeighborLoader(val_data,
                                     num_neighbors=[-1, -1],
                                     batch_size = 4096,
                                     edge_label_index=val_data.edge_label_index,
                                     edge_label=val_data.edge_label,
                                     neg_sampling_ratio=0)
test_data_loader = LinkNeighborLoader(test_data,
                                      num_neighbors=[-1, -1],
                                      batch_size = 4096,
                                      edge_label_index=test_data.edge_label_index,
                                      edge_label=test_data.edge_label,
                                      neg_sampling_ratio=0)

print(type(val_data_loader))

#%%
total_num_nodes = 0
count = 0
for step, sub_data in enumerate(val_data_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of nodes in the current batch: {sub_data.num_nodes}')
    print(sub_data)
    print()
    total_num_nodes += sub_data.num_nodes
    count += 1
    if count == 1:
        break

#%%
    
class Net(torch.nn.Module):
    def __init__(self, in_channels,hidden_channels, out_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, out_channels)


    def encode(self, x, edge_index) -> torch.Tensor:
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

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
def train():
    net.train()

    total_loss = 0
    total_pred = 0
    for batch in tqdm.tqdm(train_data_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        z = net.encode(batch.x, batch.edge_index)
        out = net.decode(z, batch.edge_label_index).view(-1)
        loss = criterion(out, batch.edge_label)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * out.size(0)
        total_pred += out.size(0)

    return total_loss / total_pred


@torch.no_grad()
def test(data_loader):
    net.eval()
    total_pred = []
    total_edge_label = []
    for batch in tqdm.tqdm(data_loader):
        batch = batch.to(device)
        z = net.encode(batch.x, batch.edge_index)
        out = net.decode(z, batch.edge_label_index)
        total_edge_label.extend(batch.edge_label.cpu().tolist())
        total_pred.extend(out.cpu().tolist())
    return roc_auc_score(total_edge_label, total_pred)

#%%
best_val_auc = final_test_auc = 0
for epoch in range(1, 101):
    loss = train()
    val_auc = test(val_data_loader)
    # test_auc = test(test_data_loader)
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        # final_test_auc = test_auc
    print(f'Epoch: {epoch:03d}, Loss: {loss:.8f}, Val: {val_auc:.8f}, '
        #   f'Test: {test_auc:.8f}'
          )

print(f'Final Test: {final_test_auc:.8f}')
#%%

# %%
