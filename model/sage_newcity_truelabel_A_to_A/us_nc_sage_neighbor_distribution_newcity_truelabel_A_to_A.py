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

from torch_geometric.loader import NeighborLoader, ImbalancedSampler
from torch_geometric.nn import SAGEConv, GraphSAGE, GCNConv
import torch_geometric.transforms as T
import torch_geometric as pyg
from sklearn.metrics import classification_report

from us_lgc_newcity_truelabel_more_washington_51by6_neighbor_distribution_with_id_in_mem_dataset import USLGCNewcityTrueLableMoreWashingtonDistributionFeaturesWithIdDataset
EPS = 1e-15

# %% 
# Load dataset
dataset = USLGCNewcityTrueLableMoreWashingtonDistributionFeaturesWithIdDataset("../../data/")
# examine the graph
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')

print('===============================================================================')
# data = dataset.data
# slices = dataset.slices
data = dataset.data
# print(data.x[0:1])
# data.x = data.x[:, [i*3 + 2 for i in range(52)]]
# print(data.x)
# split labeled_mask to train, val, and test
labeled_indices = data.labeled_mask.nonzero(as_tuple=False).view(-1)
n = data.labeled_mask.shape[0]
order = torch.randperm(labeled_indices.shape[0]) # random shuffled index
labeled_indices = labeled_indices[order] # shuffle the tensor
val_indices = labeled_indices[0:233000]
test_indices = labeled_indices[233000:466000]
train_indices = labeled_indices[466000:]
data.train_mask = pyg.utils.index_to_mask(train_indices, n)
data.val_mask = pyg.utils.index_to_mask(val_indices, n)
data.test_mask = pyg.utils.index_to_mask(test_indices, n)

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
train_loader = NeighborLoader(data, num_neighbors=[-1] * 2, batch_size=4096,
                                input_nodes=data.train_mask)
val_loader = NeighborLoader(data, num_neighbors=[-1] * 2, batch_size=32768, 
                                input_nodes=data.val_mask)
test_loader = NeighborLoader(data, num_neighbors=[-1] * 2, batch_size=32768,
                                input_nodes=data.test_mask)
total_labeled_loader = NeighborLoader(data, num_neighbors=[-1] * 2, batch_size=1,
                                input_nodes=data.labeled_mask)
# %% 
# Define model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Now using device: ', device)

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

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# %%
# Define train()
def train(train_loader):
    model.train()

    total_loss = total_examples = 0
    for sampled_data in train_loader:
    # for sampled_data in tqdm.tqdm(train_loader):
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
def test(loader, show_report:bool):
    model.eval()
    torch.cuda.empty_cache()

    correct_sum, mask_sum = 0, 0
    y_total, pred_total = [], []
    for sampled_data in loader:
    # for sampled_data in tqdm.tqdm(loader):
        sampled_data = sampled_data.to(device)
        adj = SparseTensor(row=sampled_data.edge_index[0], col=sampled_data.edge_index[1],
                           sparse_sizes=(sampled_data.num_nodes, sampled_data.num_nodes))
        out = model(sampled_data.x, adj.t())
        # out = model(sampled_data.x.to(device), sampled_data.edge_index.to(device))
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        # y = sampled_data.y.to(device)
        correct = pred[:sampled_data.batch_size] == sampled_data.y[:sampled_data.batch_size]  # Check against ground-truth labels.
        correct_sum += int(correct.sum()) 
        mask_sum += int(sampled_data.batch_size)
        if show_report:
            y_total.extend(sampled_data.y[:sampled_data.batch_size].cpu().tolist())
            pred_total.extend(pred[:sampled_data.batch_size].cpu().tolist())
    if show_report:
        print(classification_report(y_total, pred_total))
        print('correct_sum',correct_sum, 'mask_sum',mask_sum, 'correct_sum / mask_sum', correct_sum / mask_sum)
    accs = correct_sum / mask_sum # Derive ratio of correct predictions.

    torch.cuda.empty_cache()
    return accs

# %%
# Start training
best_val_auc = final_test_auc = 0
best_loss = float('inf')
final_test_auc_epoch = 0
model_list = []
loss_list = []
test_auc_list = []
start = 0
end = 301
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
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test: {test_acc:.4f}, Time: {minute:03d} min {sec:02d} sec')
        test_auc_list.append(test_acc)

print(f'Final Test: {final_test_auc:.4f}')

# %%
# Save the model
PATH = f'./truelabel_more_washington/1/'
for i in range(len(loss_list)):
    file = PATH + f'epoch_{start + i:03d}_loss_{loss_list[i]:.4f}_testauc_{test_auc_list[i]:.4f}.pt'
    print(file)
    torch.save(model_list[i], file)

max_test_auc = max(test_auc_list)
max_index = test_auc_list.index(max_test_auc)
file = PATH + f'best_auc_epoch_{max_index:03d}_loss_{loss_list[max_index]:.4f}_testauc_{test_auc_list[max_index]:.4f}.pt'
print(file)
torch.save(model_list[max_index], file)

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

    test(val_loader, show_report=True)
    test(test_loader, show_report=True)
    test(total_labeled_loader, show_report=True)

# %%
inference()

# %%
