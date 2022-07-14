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

from torch_geometric.loader import NeighborLoader, ImbalancedSampler
from torch_geometric.nn import SAGEConv, GraphSAGE, GCNConv
import torch_geometric.transforms as T
from sklearn.metrics import classification_report

from us_lgc_all_label_51by6_neighbor_distribution_in_mem_dataset import USLGCDistributionFeaturesDataset
EPS = 1e-15

# %% 
# Load dataset
dataset = USLGCDistributionFeaturesDataset("./data/")
# examine the graph
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')

print('===============================================================================')
# data = dataset.data
# slices = dataset.slices
data = dataset.data
print(data.x[0:1])
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

# %% 
# Sample batched data for train, val, and test
sampler = ImbalancedSampler(data, input_nodes=data.train_mask)
train_loader = NeighborLoader(data, num_neighbors=[-1] * 2, batch_size=32768,
                                input_nodes=data.train_mask,
                                sampler=sampler)
val_loader = NeighborLoader(data, num_neighbors=[-1] * 2, batch_size=32768, 
                                input_nodes=data.val_mask)
test_loader = NeighborLoader(data, num_neighbors=[-1] * 2, batch_size=51000,
                                input_nodes=data.test_mask)
total_labeled_loader = NeighborLoader(data, num_neighbors=[-1] * 1, batch_size=2)
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

total_num_nodes = 0
count = 0
for step, sub_data in enumerate(total_labeled_loader):
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

class Sage(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = SAGEConv(in_channels=data.num_features, 
                              out_channels=hidden_channels,
                              aggr="add",
                              normalize=False)
        self.conv2 = SAGEConv(in_channels=hidden_channels, 
                              out_channels=data.num_classes,
                              aggr="add",
                              normalize=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        
        return x

# %%
model = Sage(hidden_channels=data.num_features)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001  ,  weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()



# %%
# Define train()
def train():
    model.train()

    total_loss = total_examples = 0
    # for sampled_data in train_loader:
    for sampled_data in tqdm.tqdm(train_loader):
        sampled_data = sampled_data.to(device)
        # sampled_data = data.to(device)
        # print(sampled_data)
        adj = SparseTensor(row=sampled_data.edge_index[0], col=sampled_data.edge_index[1],
                        sparse_sizes=(sampled_data.num_nodes, sampled_data.num_nodes))
        optimizer.zero_grad()
        out = model(sampled_data.x, adj.t())
        loss = criterion(out[:sampled_data.batch_size], 
                            sampled_data.y[:sampled_data.batch_size])
        # loss = criterion(out[:sampled_data.batch_size], 
        #                     sampled_data.y[:sampled_data.batch_size])
        loss.backward()
        optimizer.step()
        
        total_loss += loss * sampled_data.batch_size
        total_examples += sampled_data.batch_size

    return total_loss / total_examples

# %%
# Define test()
@torch.no_grad()
def test():
    model.eval()
    accs = []

    # loader = train_loader 
    # correct_sum, mask_sum = 0, 0
    # # for sampled_data in loader:
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
    print(correct_sum, mask_sum)
    accs.append(correct_sum / mask_sum) # Derive ratio of correct predictions.

    loader = test_loader
    correct_sum, mask_sum = 0, 0
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
    print(correct_sum, mask_sum)
    accs.append(correct_sum / mask_sum) # Derive ratio of correct predictions.

    torch.cuda.empty_cache()
    return accs


# %%
# Define prediction distribution check function()
@torch.no_grad()
def get_pred_classes_count():
    torch.cuda.empty_cache()
    model.eval()
    accs = []
    total_class_count = [0 for i in range(data.num_classes)]
    correct_class_count = [0 for i in range(data.num_classes)]
    
    loader = train_loader 
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

        for i in range(sampled_data.batch_size):
            total_class_count[pred[i].item()] += 1
            if correct[i] == True:
                correct_class_count[pred[i]] += 1
        y_total.extend(sampled_data.y[:sampled_data.batch_size].cpu().tolist())
        pred_total.extend(pred[:sampled_data.batch_size].cpu().tolist()) 
    print(classification_report(y_total, pred_total))
    print(correct_sum, mask_sum)
    accs.append(correct_sum / mask_sum) # Derive ratio of correct predictions.

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

        for i in range(sampled_data.batch_size):
            total_class_count[pred[i].item()] += 1
            if correct[i] == True:
                correct_class_count[pred[i]] += 1
        y_total.extend(sampled_data.y[:sampled_data.batch_size].cpu().tolist())
        pred_total.extend(pred[:sampled_data.batch_size].cpu().tolist())                            
    print(classification_report(y_total, pred_total))
    print(correct_sum, mask_sum)
    accs.append(correct_sum / mask_sum) # Derive ratio of correct predictions.


    loader = test_loader
    correct_sum, mask_sum = 0, 0
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

        for i in range(sampled_data.batch_size):
            total_class_count[pred[i].item()] += 1
            if correct[i] == True:
                correct_class_count[pred[i]] += 1
        print(classification_report(sampled_data.y[:sampled_data.batch_size].cpu(),
                                    pred[:sampled_data.batch_size].cpu()))
    print(correct_sum, mask_sum)
    accs.append(correct_sum / mask_sum) # Derive ratio of correct predictions.

    torch.cuda.empty_cache()
    return accs, total_class_count, correct_class_count

# %%
# Start training

min_loss = float('inf')
best_model_state = None
for epoch in range(1, 200):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.10f}')
    # train_acc, val_acc, test_acc  = test()
    # print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    # if epoch % 1 == 0:
    #     accs, _, _ = get_pred_classes_count()
    #     train_acc, val_acc, test_acc = accs
    #     print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    if loss < min_loss:
        min_loss = loss
        best_model_state = deepcopy(model.state_dict())

print('min_loss',  min_loss)
# %%
# Save the model
PATH = './model/sage_all_label_imbalancedSample/sage_all_label_imbalancedSample.pt'
# torch.save(best_model_state, PATH)

# %%
# define the loaded model

model_copy = Sage(hidden_channels=data.num_features)
model_copy.load_state_dict(torch.load(PATH))
model_copy.to(device)
cpu = torch.device('cpu')

# %%
model.to(cpu)
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
