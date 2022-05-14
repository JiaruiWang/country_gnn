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

import torch, tqdm
from torch_sparse import SparseTensor
import torch.nn.functional as F

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv, GraphSAGE
import torch_geometric.transforms as T

from us_lgc_in_mem_dataset import USLGCdataset
EPS = 1e-15

# %% 
# Load dataset
dataset = USLGCdataset("./data/")
# examine the graph
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')

print('===============================================================================')
# data = dataset.data
# slices = dataset.slices
data = dataset.data
data.x = data.x[:, [i*3 + 2 for i in range(52)]]
print(data.x)
transform = T.RandomNodeSplit(split='random',
                              num_train_per_class=100000,
                              num_val=0.05,
                              num_test=0.05)
data = transform(data)
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

# %% 
# Sample batched data for train, val, and test

train_loader = NeighborLoader(data, num_neighbors=[10] * 2, batch_size=128,
                                input_nodes=data.train_mask)
val_loader = NeighborLoader(data, num_neighbors=[-1] * 2, batch_size=128, 
                                input_nodes=data.val_mask)
test_loader = NeighborLoader(data, num_neighbors=[-1] * 2, batch_size=128,
                                input_nodes=data.test_mask)
# %%
# Show batched data in loader
total_num_nodes = 0
count = 0
for step, sub_data in enumerate(val_loader):
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
    if count == 2:
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
    if count == 2:
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

class Sage(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = SAGEConv(in_channels=data.num_features, 
                              out_channels=hidden_channels,
                              aggr="add")
        self.conv2 = SAGEConv(in_channels=hidden_channels, 
                              out_channels=data.num_classes,
                              aggr="add")

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        
        return x

model = Sage(hidden_channels=64)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01 )# ,  weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()



# %%
# Define train()
def train():
    model.train()

    total_loss = total_examples = 0
    # for sampled_data in tqdm.tqdm(train_loader):
        # sampled_data = sampled_data.to(device)
    sampled_data = data.to(device)
    adj = SparseTensor(row=sampled_data.edge_index[0], 
                        col=sampled_data.edge_index[1],
                        sparse_sizes=(sampled_data.num_nodes,
                                        sampled_data.num_nodes))
    optimizer.zero_grad()
    out = model(sampled_data.x, adj.t())
    loss = criterion(out[sampled_data.train_mask], 
                        sampled_data.y[sampled_data.train_mask])
    loss.backward()
    optimizer.step()
    
    total_loss += loss
    total_examples += sampled_data.train_mask.sum()

    return total_loss / total_examples

# %%
# Define test()
@torch.no_grad()
def test():
    model.eval()
    accs = []
    # for loader in [train_loader, val_loader, test_loader]:
    loader = train_loader
    correct_sum, mask_sum = 0, 0
    for sampled_data in tqdm.tqdm(loader):
        out = model(sampled_data.x.to(device), sampled_data.edge_index.to(device))
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        y = sampled_data.y.to(device)
        correct = pred[sampled_data.train_mask] == y[sampled_data.train_mask]  # Check against ground-truth labels.
        correct_sum += int(correct.sum()) 
        mask_sum += int(sampled_data.train_mask.sum())
    accs.append(correct_sum / mask_sum) # Derive ratio of correct predictions.

    loader = val_loader
    correct_sum, mask_sum = 0, 0
    for sampled_data in tqdm.tqdm(loader):
        out = model(sampled_data.x.to(device), sampled_data.edge_index.to(device))
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        y = sampled_data.y.to(device)
        correct = pred[sampled_data.val_mask] == y[sampled_data.val_mask]  # Check against ground-truth labels.
        correct_sum += int(correct.sum()) 
        mask_sum += int(sampled_data.val_mask.sum())
    accs.append(correct_sum / mask_sum) # Derive ratio of correct predictions.

    loader = test_loader
    correct_sum, mask_sum = 0, 0
    for sampled_data in tqdm.tqdm(loader):
        out = model(sampled_data.x.to(device), sampled_data.edge_index.to(device))
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        y = sampled_data.y.to(device)
        correct = pred[sampled_data.test_mask] == y[sampled_data.test_mask]  # Check against ground-truth labels.
        correct_sum += int(correct.sum()) 
        mask_sum += int(sampled_data.test_mask.sum())
    accs.append(correct_sum / mask_sum) # Derive ratio of correct predictions.

    return accs

# %%
# Start training
for epoch in range(1, 200):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.10f}')
    if epoch % 100 == 0:
        train_acc, val_acc, test_acc = test()
        print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
# %%
