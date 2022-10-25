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
from dis import pretty_flags
from unittest import TestLoader

import torch, tqdm
from torch_sparse import SparseTensor
import torch.nn.functional as F

from torch_geometric.loader import NeighborLoader, ImbalancedSampler, GraphSAINTRandomWalkSampler
from torch_geometric.nn import SAGEConv, GraphSAGE, GCNConv, GraphConv
from torch_geometric.utils import degree
import torch_geometric.transforms as T
from sklearn.metrics import classification_report

from us_lgc_all_label_population_label_51by6_neighbor_distribution_with_id_in_mem_dataset import USLGCPopulationLabelDistributionFeaturesWithIdDataset
EPS = 1e-15

# %% 
# Load dataset
dataset = USLGCPopulationLabelDistributionFeaturesWithIdDataset("./data/")
# examine the graph
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')

print('===============================================================================')
data = dataset.data
# row, col = data.edge_index
# data.edge_weight = 1. / degree(col, data.num_nodes)[col]  # Norm by in-degree.
# use_normalization = True
# action='store_true'

# transform = T.RandomNodeSplit(split='random',
#                               num_train_per_class=1000,
#                               num_val=590000,
#                               num_test=5232395)
# data = transform(data)
# temp = data.train_mask
# data.train_mask = data.test_mask
# data.test_mask = temp
# print(f'Data in the {dataset} :\n {data}')


# # Gather some statistics about the graph.
# print(f'Number of nodes: {data.num_nodes}')
# print(f'Number of features: {data.num_features}')
# print(f'Number of classes: {data.num_classes}')
# print(f'Number of edges: {data.num_edges}')
# print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
# print(f'Number of training nodes: {data.train_mask.sum()}')
# print(f'Number of validation nodes: {data.val_mask.sum()}')
# print(f'Number of testing nodes: {data.test_mask.sum()}')
# print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')

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

# %% 
# Sample batched data for train, val, and test
# train_loader = GraphSAINTRandomWalkSampler(data, batch_size=600000, walk_length=2,
#                                            num_steps=5, sample_coverage=1000,
#                                            save_dir=dataset.processed_dir,
#                                            num_workers=20)
# train_loader = GraphSAINTRandomWalkSampler(data, batch_size=600000, walk_length=2,
#                                            num_steps=5, sample_coverage=500,
#                                            save_dir=dataset.processed_dir,
#                                            num_workers=20)
# train_loader = GraphSAINTRandomWalkSampler(data, batch_size=600000, walk_length=2,
#                                            num_steps=5, sample_coverage=50,
#                                            save_dir=dataset.processed_dir,
#                                            num_workers=20)
# train_loader = GraphSAINTRandomWalkSampler(data, batch_size=600000, walk_length=2,
#                                            num_steps=5, sample_coverage=25,
#                                            save_dir=dataset.processed_dir,
#                                            num_workers=20)

# train_loader = GraphSAINTRandomWalkSampler(data, batch_size=600000, walk_length=2,
#                                            num_steps=5, sample_coverage=100,
#                                            save_dir=dataset.processed_dir,
#                                            num_workers=20)
# train_acc_loader = NeighborLoader(data, num_neighbors=[-1] * 2, batch_size=16384,
#                                 input_nodes=data.train_mask)
# val_loader = NeighborLoader(data, num_neighbors=[-1] * 2, batch_size=16384, 
                                # input_nodes=data.val_mask)
# test_loader = NeighborLoader(data, num_neighbors=[-1] * 2, 
#                                 # batch_size=16384,
#                                 batch_size=1,
#                                 input_nodes=data.test_mask)
total_labeled_loader = NeighborLoader(data, num_neighbors=[-1] * 2, batch_size=1)

# %%
# Show batched data in loader

# total_num_nodes = 0
# count = 0
# for step, sub_data in enumerate(train_loader):
#     print(f'Step {step + 1}:')
#     print('=======')
#     print(f'Number of nodes in the current batch: {sub_data.num_nodes}')
#     print(sub_data)
#     print(sub_data.train_mask.sum())
#     print(sub_data.val_mask.sum())
#     print(sub_data.test_mask.sum())
#     print()
#     total_num_nodes += sub_data.num_nodes
#     count += 1
#     if count == 1:
#         break
# print(f'Iterated over {total_num_nodes} of {data.num_nodes} nodes!')


# total_num_nodes = 0
# count = 0
# for step, sub_data in enumerate(test_loader):
#     print(f'Step {step + 1}:')
#     print('=======')
#     print(f'Number of nodes in the current batch: {sub_data.num_nodes}')
#     print(sub_data)
#     print(sub_data.train_mask.sum())
#     print(sub_data.val_mask.sum())
#     print(sub_data.test_mask.sum())
#     print()
#     total_num_nodes += sub_data.num_nodes
#     count += 1
#     if count == 1:
#         break
# print(f'Iterated over {total_num_nodes} of {data.num_nodes} nodes!')

# total_num_nodes = 0
# count = 0
# for step, sub_data in enumerate(total_labeled_loader):
#     print(f'Step {step + 1}:')
#     print('=======')
#     print(f'Number of nodes in the current batch: {sub_data.num_nodes}')
#     print(sub_data)
#     print(sub_data.train_mask.sum())
#     print(sub_data.val_mask.sum())
#     print(sub_data.test_mask.sum())
#     print(sub_data.y[:sub_data.batch_size])
#     print(sub_data.id[:sub_data.batch_size])
#     print()
#     total_num_nodes += sub_data.num_nodes
#     count += 1
#     if count == 1:
#         break
# print(f'Iterated over {total_num_nodes} of {data.num_nodes} nodes!')

# %% 
# Define GraphSAGE model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print('Now using device: ', device)
# model = GraphSAGE(
#     in_channels=data.num_features,
#     hidden_channels=64,
#     num_layers=2,
#     out_channels=243,
#     dropout=0.5,
# ).to(device)

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
# Save the model
PATH = './model/saint_population_label_all_label/saint_population_label_all_label_train_with_set_seed.pt'
# torch.save(best_model_state, PATH)

# %%
# define the loaded model

model_copy = Net(hidden_channels=data.num_features)
model_copy.load_state_dict(torch.load(PATH))
model_copy.to(device)
cpu = torch.device('cpu')

# %%
# define inference()

# @torch.no_grad()
# def inference():
#     torch.cuda.empty_cache()
#     model_copy.eval()

#     loader = val_loader
#     correct_sum, mask_sum = 0, 0
#     y_total, pred_total = [], []
#     print('Validation set:')
#     # for sampled_data in loader:
#     for sampled_data in tqdm.tqdm(loader):
#         sampled_data = sampled_data.to(device)
#         adj = SparseTensor(row=sampled_data.edge_index[0], col=sampled_data.edge_index[1],
#                             sparse_sizes=(sampled_data.num_nodes, sampled_data.num_nodes))
#         out = model_copy(sampled_data.x, adj.t())
#         pred = out.argmax(dim=1)  # Use the class with highest probability.
#         correct = pred[:sampled_data.batch_size] == sampled_data.y[:sampled_data.batch_size]  # Check against ground-truth labels.
#         correct_sum += int(correct.sum()) 
#         mask_sum += int(sampled_data.batch_size)
#         y_total.extend(sampled_data.y[:sampled_data.batch_size].cpu().tolist())
#         pred_total.extend(pred[:sampled_data.batch_size].cpu().tolist())                            
#     print(classification_report(y_total, pred_total))
#     print(correct_sum, mask_sum)
#     print(correct_sum / mask_sum)

#     torch.cuda.empty_cache()
#     loader = test_loader
#     correct_sum, mask_sum = 0, 0
#     y_total, pred_total = [], []
#     print('Testing set:')
#     # for sampled_data in loader:
#     for sampled_data in tqdm.tqdm(loader):
#         sampled_data = sampled_data.to(device)
#         adj = SparseTensor(row=sampled_data.edge_index[0], col=sampled_data.edge_index[1],
#                             sparse_sizes=(sampled_data.num_nodes, sampled_data.num_nodes))
#         out = model_copy(sampled_data.x, adj.t())
#         pred = out.argmax(dim=1)  # Use the class with highest probability.
#         correct = pred[:sampled_data.batch_size] == sampled_data.y[:sampled_data.batch_size]  # Check against ground-truth labels.
#         correct_sum += int(correct.sum()) 
#         mask_sum += int(sampled_data.batch_size)
#         y_total.extend(sampled_data.y[:sampled_data.batch_size].cpu().tolist())
#         pred_total.extend(pred[:sampled_data.batch_size].cpu().tolist())                            
#     print(classification_report(y_total, pred_total))
#     print(correct_sum, mask_sum)
#     print(correct_sum / mask_sum)

#     torch.cuda.empty_cache()
#     loader = total_labeled_loader 
#     correct_sum, mask_sum = 0, 0
#     y_total, pred_total = [], []
#     print('Total set:')
#     # for sampled_data in loader:
#     for sampled_data in tqdm.tqdm(loader):
#         sampled_data = sampled_data.to(device)
#         adj = SparseTensor(row=sampled_data.edge_index[0], col=sampled_data.edge_index[1],
#                            sparse_sizes=(sampled_data.num_nodes, sampled_data.num_nodes))
#         out = model_copy(sampled_data.x, adj.t())
#         pred = out.argmax(dim=1)  # Use the class with highest probability.
#         correct = pred[:sampled_data.batch_size] == sampled_data.y[:sampled_data.batch_size]  # Check against ground-truth labels.
#         correct_sum += int(correct.sum()) 
#         mask_sum += int(sampled_data.batch_size)
#         y_total.extend(sampled_data.y[:sampled_data.batch_size].cpu().tolist())
#         pred_total.extend(pred[:sampled_data.batch_size].cpu().tolist())                            
#     print(classification_report(y_total, pred_total))
#     print(correct_sum, mask_sum)
#     print(correct_sum / mask_sum) # Derive ratio of correct predictions.
#     torch.cuda.empty_cache()

# %%
# inference()

# %%

torch.cuda.empty_cache()
loader = total_labeled_loader
correct_sum, mask_sum = 0, 0
y_total, pred_total = [], []
id_total, out_total = [],[]
print('Total set:')
# for sampled_data in loader:
for sampled_data in tqdm.tqdm(loader):
    sampled_data = sampled_data.to(device)
    adj = SparseTensor(row=sampled_data.edge_index[0], col=sampled_data.edge_index[1],
                        sparse_sizes=(sampled_data.num_nodes, sampled_data.num_nodes))
    out = model_copy(sampled_data.x, adj.t())
    # print(out.shape)
    # print(out)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    # print(pred.shape)
    # print(pred)
    # print(pred[:sampled_data.batch_size].shape)
    # print(pred[:sampled_data.batch_size])
    correct = pred[:sampled_data.batch_size] == sampled_data.y[:sampled_data.batch_size]  # Check against ground-truth labels.
    correct_sum += int(correct.sum()) 
    mask_sum += int(sampled_data.batch_size)
    y_total.extend(sampled_data.y[:sampled_data.batch_size].cpu().tolist())
    id_total.extend(sampled_data.id[:sampled_data.batch_size].cpu().tolist())
    pred_total.extend(pred[:sampled_data.batch_size].cpu().tolist())
    out_total.append(out[:sampled_data.batch_size].cpu().tolist())  
torch.cuda.empty_cache()                      
print(classification_report(y_total, pred_total))
print(correct_sum, mask_sum)
print(correct_sum / mask_sum) # Derive ratio of correct predictions.

# %%
# %%
output_list = []
for i in range(5873395):
    row = []
    row.append(id_total[i])
    row.append(y_total[i])
    row.append(pred_total[i])
    row.extend(out_total[i][0])
    output_list.append(row)
print(len(output_list))
print(len(output_list[0]))
# %%
import pandas as pd
outdf = pd.DataFrame(output_list)
# %%
print(outdf[0:1])
# %%
outputfile = "./model/saint_population_label_all_label/saint_inference_output_id_y_pred_51probability.csv"
outdf.to_csv(outputfile, sep=',', header=False, index=False)
# %%
print(outdf.shape)
# %%
