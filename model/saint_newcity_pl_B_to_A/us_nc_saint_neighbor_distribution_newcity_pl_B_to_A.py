'''References
PgG examples:
graph sage unsupvervised
https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_sage_unsup.py
graph sage unsupvervised ppi
https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_sage_unsup_ppi.py

DGL examples:
node_classification.py
https://github.com/dmlc/dgl/blob/master/examples/pytorch/graphsage/node_classification.py#L88
'''

# %% 
# Import packages
from copy import deepcopy

import torch, tqdm, time
from torch_sparse import SparseTensor
import torch.nn.functional as F

import torch_geometric as pyg
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader, ImbalancedSampler, GraphSAINTRandomWalkSampler
from torch_geometric.nn import SAGEConv, GraphSAGE, GCNConv, GraphConv
from torch_geometric.utils import degree, subgraph
import torch_geometric.transforms as T
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from us_lgc_newcity_pl_more_washington_51by6_neighbor_distribution_with_id_in_mem_dataset import USLGCNewcityPLMoreWashingtonDistributionFeaturesWithIdDataset
EPS = 1e-15

# %% 
# Load dataset
dataset = USLGCNewcityPLMoreWashingtonDistributionFeaturesWithIdDataset("../../data/")
# examine the graph
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(dataset[0])
print('===============================================================================')
#%%
# data = dataset.data
# slices = dataset.slices
data = dataset.data
row, col = data.edge_index
data.edge_weight = 1. / degree(col, data.num_nodes)[col]  # Norm by in-degree.
use_normalization = True
action='store_true'

labeled_indices = data.labeled_mask.nonzero(as_tuple=False).view(-1)
n = data.labeled_mask.shape[0]
order = torch.randperm(labeled_indices.shape[0]) # random shuffled index
labeled_indices = labeled_indices[order] # shuffle the tensor
val_indices = labeled_indices[0:233000] # all from truelabel
test_indices = labeled_indices[233000:466000] # all from truelabel
train_indices = labeled_indices[466000:] # all from truelabel
data.train_mask = pyg.utils.index_to_mask(train_indices, n)
data.val_mask = pyg.utils.index_to_mask(val_indices, n)
data.test_mask = pyg.utils.index_to_mask(test_indices, n)
data.unlabeled_mask = (data.labeled_mask != 1)
# print(f'Slices in the {dataset} :\n {slices}')
print(dataset[0])
# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of features: {data.num_features}')
print(f'Number of classes: {data.num_classes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Number of validation nodes: {data.val_mask.sum()}')
print(f'Number of testing nodes: {data.test_mask.sum()}')
print(f'Number of labeled_mask: {data.labeled_mask.sum()}')
print(f'Number of unlabeled_mask: {data.unlabeled_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
# print(f'Has isolated nodes: {data.has_isolated_nodes()}') # True
# print(f'Has self-loops: {data.has_self_loops()}') # True
# print(f'Is undirected: {data.is_undirected()}') # False
#%%
print(data.test_mask[0:10])
test_mask_T = torch.unsqueeze(data.test_mask, dim=1)
print(test_mask_T.shape)
print(test_mask_T[0:10])
testMaskT_x = torch.cat((test_mask_T, data.x), dim=1)
print(testMaskT_x.shape)
#%%
# put train set and test set into one subgraph
data.test_B = (data.test_mask + data.unlabeled_mask)
subgraph_edge_index, subgraph_edge_weight = subgraph(data.test_B, data.edge_index, data.edge_weight, relabel_nodes=True)
# print(subgraph_edge_index.shape, subgraph_edge_weight.shape)

testMaskT_subgraph_x = testMaskT_x[data.test_B]
testMaskT_sub = testMaskT_subgraph_x[:, 0:1]
subgraph_x = testMaskT_subgraph_x[:, 1:307]
# print(testMaskT_subgraph_x.shape)
# print(testMaskT_sub.shape, subgraph_x.shape)
# print(testMaskT_sub[0:10])
# print(subgraph_x[0])
testMask_sub = torch.squeeze(testMaskT_sub)
print("testMaskT_sub.sum() ", testMaskT_sub.sum())
# print(testMask_sub.shape)
# print(testMask_sub[0:10])
#%%
subgraph_y = data.y[data.test_B]
subgraph_data = Data(x=subgraph_x, edge_index=subgraph_edge_index, y=subgraph_y)
subgraph_data.edge_weight = subgraph_edge_weight
subgraph_data.num_classes = 51
subgraph_data.train_mask = (testMask_sub == 0)
print("subgraph_data.train_mask.sum() ", subgraph_data.train_mask.sum())
# subgraph_data.train_mask = data.train_mask[data.labeled_mask]
# subgraph_data.val_mask = data.val_mask[data.labeled_mask]
# subgraph_data.test_mask = data.test_mask[data.labeled_mask]
print(subgraph_data)
print(subgraph_data.edge_index)
print(f'Number of nodes: {subgraph_data.num_nodes}')
print(f'Number of features: {subgraph_data.num_features}')
print(f'Number of classes: {subgraph_data.num_classes}')
print(f'Number of edges: {subgraph_data.num_edges}')
print(f'Average node degree: {subgraph_data.num_edges / subgraph_data.num_nodes:.2f}')
# print(f'Number of training nodes: {subgraph_data.train_mask.sum()}')
# print(f'Number of validation nodes: {subgraph_data.val_mask.sum()}')
# print(f'Number of testing nodes: {subgraph_data.test_mask.sum()}')
# print(f'Training node label rate: {int(subgraph_data.train_mask.sum()) / subgraph_data.num_nodes:.2f}')
# %%
'''
+----------------------------+------------------------+
| bucket                     | one_page_likespage_num |
+----------------------------+------------------------+
| 0 <= likespage <= 10       |               29154056 |
| 10 < likespage <= 100      |                8143647 |
| 100 < likespage <= 1000    |                1600350 |
| 1000 < likespage <= 10000  |                  49709 |
| 10000 < likespage <= 80000 |                    223 |
+----------------------------+------------------------+
'''
# %% 
# Sample batched data for train, val, and test

train_loader = GraphSAINTRandomWalkSampler(subgraph_data, batch_size=600000, walk_length=2,
                                           num_steps=5, sample_coverage=100,
                                           save_dir=dataset.processed_dir,
                                           num_workers=20)
val_loader = NeighborLoader(data, num_neighbors=[-1] * 2, batch_size=32768, 
                                input_nodes=data.val_mask)
test_loader = NeighborLoader(data, num_neighbors=[-1] * 2, batch_size=32768,
                                input_nodes=data.test_mask)
total_labeled_loader = NeighborLoader(data, num_neighbors=[-1] * 2, batch_size=1,
                                input_nodes=data.labeled_mask)


# %% 
# Define model

class Net(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        in_channels = data.num_features
        out_channels = data.num_classes
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        # self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(2 * hidden_channels, out_channels)

    def set_aggr(self, aggr):
        self.conv1.aggr = aggr
        self.conv2.aggr = aggr
        # self.conv3.aggr = aggr

    def forward(self, x0, edge_index, edge_weight=None):
        x1 = F.relu(self.conv1(x0, edge_index, edge_weight))
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x2 = F.relu(self.conv2(x1, edge_index, edge_weight))
        x2 = F.dropout(x2, p=0.2, training=self.training)
        # x3 = F.relu(self.conv3(x2, edge_index, edge_weight))
        # x3 = F.dropout(x3, p=0.2, training=self.training)
        x = torch.cat([x1, x2], dim=-1)
        x = self.lin(x)
        # return x.log_softmax(dim=-1)
        return x
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Now using device: ', device)
model = Net(hidden_channels=data.num_features)
model = model.to(device)
# weight = weight.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01  ,  weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss(reduction='mean')
# criterion = F.nll_loss()
# https://stackoverflow.com/questions/65192475/pytorch-logsoftmax-vs-softmax-for-crossentropyloss

# %%
# Define train()
def train(train_loader):
    model.train()
    model.set_aggr('add' if use_normalization else 'mean')
    torch.cuda.empty_cache()
    total_loss = total_examples = 0
    for data in train_loader:
    # for data in tqdm.tqdm(train_loader):
        data = data.to(device)
        # sampled_data = data.to(device)
        # print(sampled_data)
        adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                        sparse_sizes=(data.num_nodes, data.num_nodes))
        optimizer.zero_grad()
        if use_normalization:
            edge_weight = data.edge_norm * data.edge_weight
            out = model(data.x, adj.t(), edge_weight)
            # loss = F.nll_loss(out, data.y, reduction='none')
            loss = F.cross_entropy(out, data.y, reduction='none')
            # loss = F.cross_entropy(out, data.y, reduction='mean')
            # loss = criterion(out, data.y)
            loss = (loss * data.node_norm)[data.train_mask].sum()

        else:
            out = model(data.x, adj.t())
            # loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_nodes
        total_examples += data.num_nodes
    torch.cuda.empty_cache()
    return total_loss / total_examples

# %%
# Define test()
@torch.no_grad()
def test(loader, show_report:bool):
    model.eval()
    torch.cuda.empty_cache()
    accs = []

    loader = test_loader
    correct_sum, mask_sum = 0, 0
    y_total, pred_total = [], []
    for sampled_data in loader:
    # for sampled_data in tqdm.tqdm(loader):
        sampled_data = sampled_data.to(device)
        adj = SparseTensor(row=sampled_data.edge_index[0], col=sampled_data.edge_index[1],
                           sparse_sizes=(sampled_data.num_nodes, sampled_data.num_nodes))
        out = model(sampled_data.x, adj.t())
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct = pred[:sampled_data.batch_size] == sampled_data.y[:sampled_data.batch_size]  # Check against ground-truth labels.
        correct_sum += int(correct.sum()) 
        mask_sum += int(sampled_data.batch_size)
        if show_report:
            y_total.extend(sampled_data.y[:sampled_data.batch_size].cpu().tolist())
            pred_total.extend(pred[:sampled_data.batch_size].cpu().tolist())
    
    if show_report:                     
        print(classification_report(y_total, pred_total))
        print('correct_sum',correct_sum, 'mask_sum',mask_sum, 'correct_sum / mask_sum', correct_sum / mask_sum)
    
    accs= correct_sum / mask_sum # Derive ratio of correct predictions.
    torch.cuda.empty_cache()
    return accs

#%%
# load model
# PATH = f'./mnl_more_washington/1/'
# file = PATH + 'epoch_167_loss_18.5783_testauc_0.8622.pt'
# model.load_state_dict(torch.load(file))
# model.to(device)
# %%
# Start training

best_val_auc = final_test_auc = 0
best_loss = float('inf')
final_test_auc_epoch = 0
model_list = []
loss_list = []
test_auc_list = []
start = 0
end = 401
for epoch in range(start, end):
    start_time = int(time.time())
    loss = train(train_loader)
    # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    loss_list.append(loss)
    if best_loss > loss:
        best_loss = loss
    model_list.append( deepcopy(model.state_dict()))
    if epoch % 1 == 0:
        test_acc  = test(test_loader, show_report=False)
        end_time = int(time.time())
        minute = (end_time - start_time) // 60
        sec = (end_time - start_time) % 60
        print(f'Epoch: {epoch:03d}, Loss: {loss:.8f}, Test: {test_acc:.4f}, Time: {minute:03d} min {sec:02d} sec')
        test_auc_list.append(test_acc)

print(f'Final Test: {final_test_auc:.4f}')

# %%
# Save the model
PATH = f'./pl_more_washington/1/'
for i in range(len(loss_list)):
    file = PATH + f'epoch_{start + i:03d}_loss_{loss_list[i]:.4f}_testauc_{test_auc_list[i]:.4f}.pt'
    print(file)
    torch.save(model_list[i], file)
#%%
max_test_auc = max(test_auc_list)
max_index = test_auc_list.index(max_test_auc)
file = PATH + f'best_auc_epoch_{start + max_index:03d}_loss_{loss_list[max_index]:.8f}_testauc_{test_auc_list[max_index]:.4f}.pt'
print(file)
torch.save(model_list[max_index], file)
# %%
# define the loaded model

model_copy = Net(hidden_channels=data.num_features)
model_copy.load_state_dict(torch.load(PATH))
model_copy.to(device)
cpu = torch.device('cpu')
# %%
model.to(cpu)
# weight.to(cpu)

# %%
# define inference()
@torch.no_grad()
def inference():
    torch.cuda.empty_cache()
    model_copy.eval()

    test(val_loader, show_report=True)
    test(test_loader, show_report=True)
    test(total_labeled_loader, show_report=True)

# %%
inference()
# %%
test(total_labeled_loader, show_report=True)