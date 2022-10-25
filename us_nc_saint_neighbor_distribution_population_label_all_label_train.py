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

import torch, tqdm
from torch_sparse import SparseTensor
import torch.nn.functional as F

from torch_geometric.loader import NeighborLoader, ImbalancedSampler, GraphSAINTRandomWalkSampler
from torch_geometric.nn import SAGEConv, GraphSAGE, GCNConv, GraphConv
from torch_geometric.utils import degree
import torch_geometric.transforms as T
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

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
# data = dataset.data
# slices = dataset.slices
data = dataset.data
row, col = data.edge_index
data.edge_weight = 1. / degree(col, data.num_nodes)[col]  # Norm by in-degree.
use_normalization = True
action='store_true'

# print(data.x[0:1])
# data.x = data.x[:, [i*3 + 2 for i in range(52)]]
# print(data.x)
transform = T.RandomNodeSplit(split='random',
                              num_train_per_class=1000,
                              num_val=590000,
                              num_test=5232395)
data = transform(data)
temp = data.train_mask
data.train_mask = data.test_mask
data.test_mask = temp
print(f'Data in the {dataset} :\n {data}')
# print(f'Slices in the {dataset} :\n {slices}')

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
# print(f'Has isolated nodes: {data.has_isolated_nodes()}') # True
# print(f'Has self-loops: {data.has_self_loops()}') # True
# print(f'Is undirected: {data.is_undirected()}') # False
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
y_label = data.y.numpy()
print(y_label.shape)
print(np.unique(y_label))
weight = compute_class_weight(class_weight='balanced', classes=np.unique(y_label), y=y_label)
print(weight)
weight = torch.tensor(weight).float()
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
train_loader = GraphSAINTRandomWalkSampler(data, batch_size=600000, walk_length=2,
                                           num_steps=5, sample_coverage=100,
                                           save_dir=dataset.processed_dir,
                                           num_workers=20)
train_acc_loader = NeighborLoader(data, num_neighbors=[-1] * 2, batch_size=16384,
                                input_nodes=data.train_mask)
val_loader = NeighborLoader(data, num_neighbors=[-1] * 2, batch_size=16384, 
                                input_nodes=data.val_mask)
test_loader = NeighborLoader(data, num_neighbors=[-1] * 2, batch_size=16384,
                                input_nodes=data.test_mask)
total_labeled_loader = NeighborLoader(data, num_neighbors=[-1] * 2, batch_size=1)

# %%
# Show batched data in loader
total_num_nodes = 0
count = 0
for step, sub_data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of nodes in the current batch: {sub_data.num_nodes}')
    print(sub_data)
    print(sub_data.train_mask.sum())
    print(sub_data.val_mask.sum())
    print(sub_data.test_mask.sum())
    print()
    total_num_nodes += sub_data.num_nodes
    count += 1
    if count == 1:
        break

print(f'Iterated over {total_num_nodes} of {data.num_nodes} nodes!')


total_num_nodes = 0
count = 0
for step, sub_data in enumerate(test_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of nodes in the current batch: {sub_data.num_nodes}')
    print(sub_data)
    print(sub_data.train_mask.sum())
    print(sub_data.val_mask.sum())
    print(sub_data.test_mask.sum())
    print()
    total_num_nodes += sub_data.num_nodes
    count += 1
    if count == 1:
        break

print(f'Iterated over {total_num_nodes} of {data.num_nodes} nodes!')
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
model = Net(hidden_channels=data.num_features)
model = model.to(device)
weight = weight.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01  ,  weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss(reduction='mean', weight=weight)
# criterion = F.nll_loss()
# https://stackoverflow.com/questions/65192475/pytorch-logsoftmax-vs-softmax-for-crossentropyloss



# %%
# Define train()
def train():
    model.train()
    model.set_aggr('add' if use_normalization else 'mean')

    total_loss = total_examples = 0
    # for sampled_data in train_loader:
    for data in tqdm.tqdm(train_loader):
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

    return total_loss / total_examples

# %%
# Define test()
@torch.no_grad()
def test():
    model.eval()
    torch.cuda.empty_cache()
    accs = []

    loader = train_acc_loader 
    correct_sum, mask_sum = 0, 0
    # for sampled_data in loader:
    # for sampled_data in tqdm.tqdm(loader):
    #     sampled_data = sampled_data.to(device)
    #     adj = SparseTensor(row=sampled_data.edge_index[0], col=sampled_data.edge_index[1],
    #                        sparse_sizes=(sampled_data.num_nodes, sampled_data.num_nodes))
    #     out = model(sampled_data.x, adj.t())
    #     # out = model(sampled_data.x.to(device), sampled_data.edge_index.to(device))
    #     pred = out.argmax(dim=1)  # Use the class with highest probability.
    #     # y = sampled_data.y.to(device)
    #     correct = pred[:sampled_data.batch_size] == sampled_data.y[:sampled_data.batch_size]  # Check against ground-truth labels.
    #     correct_sum += int(correct.sum()) 
    #     mask_sum += int(sampled_data.batch_size)
    # print(correct_sum, mask_sum)
    # accs.append(correct_sum / mask_sum) # Derive ratio of correct predictions.
    accs.append(0)

    loader = val_loader
    correct_sum, mask_sum = 0, 0
    y_total, pred_total = [], []
    # for sampled_data in loader:
    for sampled_data in tqdm.tqdm(loader):
        sampled_data = sampled_data.to(device)
        adj = SparseTensor(row=sampled_data.edge_index[0], col=sampled_data.edge_index[1],
                           sparse_sizes=(sampled_data.num_nodes, sampled_data.num_nodes))
        out = model(sampled_data.x, adj.t())
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct = pred[:sampled_data.batch_size] == sampled_data.y[:sampled_data.batch_size]  # Check against ground-truth labels.
        correct_sum += int(correct.sum()) 
        mask_sum += int(sampled_data.batch_size)
        y_total.extend(sampled_data.y[:sampled_data.batch_size].cpu().tolist())
        pred_total.extend(pred[:sampled_data.batch_size].cpu().tolist())                            
    print(classification_report(y_total, pred_total))
    print(correct_sum, mask_sum)
    accs.append(correct_sum / mask_sum) # Derive ratio of correct predictions.

    loader = test_loader
    correct_sum, mask_sum = 0, 0
    y_total, pred_total = [], []
    # for sampled_data in loader:
    for sampled_data in tqdm.tqdm(loader):
        sampled_data = sampled_data.to(device)
        adj = SparseTensor(row=sampled_data.edge_index[0], col=sampled_data.edge_index[1],
                           sparse_sizes=(sampled_data.num_nodes, sampled_data.num_nodes))
        out = model(sampled_data.x, adj.t())
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct = pred[:sampled_data.batch_size] == sampled_data.y[:sampled_data.batch_size]  # Check against ground-truth labels.
        correct_sum += int(correct.sum()) 
        mask_sum += int(sampled_data.batch_size)
        y_total.extend(sampled_data.y[:sampled_data.batch_size].cpu().tolist())
        pred_total.extend(pred[:sampled_data.batch_size].cpu().tolist())                            
    print(classification_report(y_total, pred_total))
    print(correct_sum, mask_sum)
    accs.append(correct_sum / mask_sum) # Derive ratio of correct predictions.

    torch.cuda.empty_cache()
    return accs

# %%
# Start training
min_loss = float('inf')
best_model_state = None
for epoch in range(1, 300):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.10f}')
    # if epoch % 50 == 0:
    #     torch.cuda.empty_cache()
    #     print(f'Start testing Epoch: {epoch:03d}')
    #     train_acc, val_acc, test_acc  = test()
    #     print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    if loss < min_loss:
        min_loss = loss
        best_model_state = deepcopy(model.state_dict())
#%%
print('min_loss',  min_loss)

# %%
# Save the model
PATH = './model/saint_population_label_all_label/saint_population_label_all_label_train_with_set_seed.pt'
torch.save(best_model_state, PATH)

# %%
# define the loaded model

model_copy = Net(hidden_channels=data.num_features)
model_copy.load_state_dict(torch.load(PATH))
model_copy.to(device)
cpu = torch.device('cpu')
# %%
model.to(cpu)
weight.to(cpu)

# %%
# define inference()
@torch.no_grad()
def inference():
    torch.cuda.empty_cache()
    model_copy.eval()

    loader = val_loader
    correct_sum, mask_sum = 0, 0
    y_total, pred_total = [], []
    print('Validation set:')
    # for sampled_data in loader:
    for sampled_data in tqdm.tqdm(loader):
        sampled_data = sampled_data.to(device)
        adj = SparseTensor(row=sampled_data.edge_index[0], col=sampled_data.edge_index[1],
                            sparse_sizes=(sampled_data.num_nodes, sampled_data.num_nodes))
        out = model_copy(sampled_data.x, adj.t())
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct = pred[:sampled_data.batch_size] == sampled_data.y[:sampled_data.batch_size]  # Check against ground-truth labels.
        correct_sum += int(correct.sum()) 
        mask_sum += int(sampled_data.batch_size)
        y_total.extend(sampled_data.y[:sampled_data.batch_size].cpu().tolist())
        pred_total.extend(pred[:sampled_data.batch_size].cpu().tolist())                            
    print(classification_report(y_total, pred_total))
    print(correct_sum, mask_sum)
    print(correct_sum / mask_sum)

    torch.cuda.empty_cache()
    loader = test_loader
    correct_sum, mask_sum = 0, 0
    y_total, pred_total = [], []
    print('Testing set:')
    # for sampled_data in loader:
    for sampled_data in tqdm.tqdm(loader):
        sampled_data = sampled_data.to(device)
        adj = SparseTensor(row=sampled_data.edge_index[0], col=sampled_data.edge_index[1],
                            sparse_sizes=(sampled_data.num_nodes, sampled_data.num_nodes))
        out = model_copy(sampled_data.x, adj.t())
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct = pred[:sampled_data.batch_size] == sampled_data.y[:sampled_data.batch_size]  # Check against ground-truth labels.
        correct_sum += int(correct.sum()) 
        mask_sum += int(sampled_data.batch_size)
        y_total.extend(sampled_data.y[:sampled_data.batch_size].cpu().tolist())
        pred_total.extend(pred[:sampled_data.batch_size].cpu().tolist())                            
    print(classification_report(y_total, pred_total))
    print(correct_sum, mask_sum)
    print(correct_sum / mask_sum)

    torch.cuda.empty_cache()
    loader = total_labeled_loader 
    correct_sum, mask_sum = 0, 0
    y_total, pred_total = [], []
    print('Total set:')
    # for sampled_data in loader:
    for sampled_data in tqdm.tqdm(loader):
        sampled_data = sampled_data.to(device)
        adj = SparseTensor(row=sampled_data.edge_index[0], col=sampled_data.edge_index[1],
                           sparse_sizes=(sampled_data.num_nodes, sampled_data.num_nodes))
        out = model_copy(sampled_data.x, adj.t())
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct = pred[:sampled_data.batch_size] == sampled_data.y[:sampled_data.batch_size]  # Check against ground-truth labels.
        correct_sum += int(correct.sum()) 
        mask_sum += int(sampled_data.batch_size)
        y_total.extend(sampled_data.y[:sampled_data.batch_size].cpu().tolist())
        pred_total.extend(pred[:sampled_data.batch_size].cpu().tolist())                            
    print(classification_report(y_total, pred_total))
    print(correct_sum, mask_sum)
    print(correct_sum / mask_sum) # Derive ratio of correct predictions.
    torch.cuda.empty_cache()

# %%
inference()

# %%
print('min_loss',  min_loss)
# %%


torch.cuda.empty_cache()
loader = total_labeled_loader 
correct_sum, mask_sum = 0, 0
y_total, pred_total = [], []
print('Total set:')
# for sampled_data in loader:
for sampled_data in tqdm.tqdm(loader):
    sampled_data = sampled_data.to(device)
    adj = SparseTensor(row=sampled_data.edge_index[0], col=sampled_data.edge_index[1],
                        sparse_sizes=(sampled_data.num_nodes, sampled_data.num_nodes))
    out = model_copy(sampled_data.x, adj.t())
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    correct = pred[:sampled_data.batch_size] == sampled_data.y[:sampled_data.batch_size]  # Check against ground-truth labels.
    correct_sum += int(correct.sum()) 
    mask_sum += int(sampled_data.batch_size)
    y_total.extend(sampled_data.y[:sampled_data.batch_size].cpu().tolist())
    pred_total.extend(pred[:sampled_data.batch_size].cpu().tolist())                            
print(classification_report(y_total, pred_total))
print(correct_sum, mask_sum)
print(correct_sum / mask_sum) # Derive ratio of correct predictions.
torch.cuda.empty_cache()
#%%


