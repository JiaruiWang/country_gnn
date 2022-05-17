# %% Import packages
import pandas as pd
import os.path as osp

# import torch
# import torch_geometric
# from torch_geometric.data import InMemoryDataset, Data
# from torch_geometric.data import DataLoader
import mysql.connector

# %%
def get_node_feature(file_path: str = None):
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
    return df

df = get_node_feature('./data/raw_dir/dist_list_idx+52_in_out_all.csv')
x = list(df.values)
print(len(x), len(x[0]))
# %%
mydb = mysql.connector.connect(
  host="localhost",
  user="jerry",
  password="password",
  database="pagenet"
)
mycursor = mydb.cursor()
count = 0
for i in range(5873395):
    idx = x[i][0]
    sql = 'update pages set us_lgc=1 where idx={};'.format(idx)
    mycursor.execute(sql)
    count += 1
    if count % 100000 == 0:
        mydb.commit()
        print(count, "record updated.")
mydb.commit()
print(count, "record updated.")
# %%
