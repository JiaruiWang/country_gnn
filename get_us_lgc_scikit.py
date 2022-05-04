# %% 
# Import packages
# import pandas as pd
import numpy as np
import sknetwork as skn
print(skn.__path__)
# import networkx as nx
import os.path as osp
import mysql.connector


# %% 

def get_node(file_path: str = None) -> list:
    df = pd.read_csv(file_path, sep='\t', header=None)
    x = [i for i in range(num_nodes)]
    
    return x

def get_edge(file_path: str) -> list:

    df = pd.read_csv(file_path, sep='\t', header=None)
    edge_index = df.to_records(index=False)
    edge = list(edge_index)
    return edge


# %%
# Load U.S. graph
directed_G_US = skn.data.load_edge_list(file='./data/raw_dir/us_edges.csv',
                                     directed=True,
                                     delimiter='\t')

undirected_G_US = skn.data.load_edge_list(file='./data/raw_dir/us_edges.csv',
                                       directed=False,
                                       delimiter='\t')

# %% 
# Define histogram function for ndarray
def histogram(arr: np.ndarray) -> dict:
    unique, counts = np.unique(arr, return_counts=True)
    return dict(zip(unique, counts))

# %%
# Print U.S. graph
print(directed_G_US)
'''
{'adjacency': <5879042x5879042 sparse matrix of type '<class 'numpy.int64'>'
	with 84485587 stored elements in Compressed Sparse Row format>, 'names': 
        array([       0,       20,       23, ..., 60924663, 60924664, 60924672],
      dtype=int32)}'''
print(undirected_G_US)
directed_labels_US = skn.topology.get_connected_components(directed_G_US.adjacency)
undirected_labels_US = skn.topology.get_connected_components(undirected_G_US.adjacency)
print(histogram(directed_labels_US))
# 5873395:0 nodes 84480575 edges in the largest connected component, 2-2608 small component(1-4 nodes)
# 6163640 nodes in table us_pages, 5879042 nodes 84485587 edges total in us graph, 290245
'''TODO get edges one node is in the us_pages?
insert into one_node_us select * from edges where (page_id in (select id from us_pages) 
and likes_page not in (select id from us_pages)) or (page_id not in (select id from us_pages) 
and likes_page in (select id from us_pages));
'''

# %%
# Get US largest connected componnet
'''TODO
1. remove all the small components. 
2. add labels to world graph, run clustering or classification to get us sub graph?
'''
component_adjacency_US, component_index_US = skn.topology.get_largest_connected_component(directed_G_US.adjacency, return_labels=True)
print(component_adjacency_US.shape)
print(component_index_US.shape)
node_component_index_US = skn.topology.get_connected_components(directed_G_US.adjacency, connection='weak')
print(node_component_index_US.shape)
print(directed_G_US.names.shape)

# %%
# Print US largest connected component
print(component_adjacency_US)
print(component_index_US)
# %%
print(directed_G_US.names)
# %%
update_idx = directed_G_US.names[component_index_US]
print(update_idx.shape)
print(update_idx)
# %%
# Update largest component indice to mysql table us_edges column scikit_lgc.
mydb = mysql.connector.connect(
  host="localhost",
  user="jerry",
  password="password",
  database="pagenet"
)
mycursor = mydb.cursor()
sql_prefix = "update us_pages set scikit_lgc=True where idx="
count = 0
for i in update_idx:
    # if count > 60890100: print(i)
    sql = sql_prefix + str(i) + ";"
    mycursor.execute(sql)
    count += 1
    if count % 10000 == 0:
        mydb.commit()
        print(count, "record updated.")
mydb.commit()
print(count, "record updated.")
