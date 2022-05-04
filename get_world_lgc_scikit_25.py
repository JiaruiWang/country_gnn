# %% 
# Import packages
# import pandas as pd
import numpy as np
import sknetwork as skn
from sknetwork.data import load_edge_list
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
directed_G = skn.data.load_edge_list(file='./data/raw_dir/us_edges.csv',
                                     directed=True,
                                     delimiter='\t')

undirected_G = skn.data.load_edge_list(file='./data/raw_dir/us_edges.csv',
                                       directed=False,
                                       delimiter='\t')

# %% 
# Define histogram function for ndarray
def histogram(arr: np.ndarray) -> dict:
    unique, counts = np.unique(arr, return_counts=True)
    return dict(zip(unique, counts))

# %%
# Print U.S. graph
print(directed_G)
print(undirected_G)
directed_labels = skn.topology.get_connected_components(directed_G.adjacency)
undirected_labels = skn.topology.get_connected_components(undirected_G.adjacency)
print(histogram(directed_labels))
# 5873395:0 nodes in the largest connected component, 2-2608 small component(1-4 nodes)
# 6163640 nodes in total us,290245
'''TODO get edges one node is in the us_pages?
insert into one_node_us select * from edges where (page_id in (select id from us_pages) and likes_page not in (select id from us_pages)) or (page_id not in (select id from us_pages) and likes_page in (select id from us_pages));
'''
# %%



# %%
# Load world graph
# edge_list = np.genfromtxt('./data/raw_dir/edge_index.csv', delimiter='\t')
# print(edge_list.shape)
# %%
directed_G_world = skn.data.load_edge_list(file='./data/raw_dir/edge_index.csv',
                                     directed=True,
                                     delimiter='\t')

# %%
# Get world connected components
directed_labels_world = skn.topology.get_connected_components(directed_G_world.adjacency)
print(directed_labels_world.shape)

# %%
# Print world connected components
print(histogram(directed_labels_world))
# 60890130:0 nodes in the largest connected component, 2-15000 small component(1-28 nodes)
# 60924683 nodes in world, 34553

# %%
# Get world largest connected componnet
'''TODO
1. remove all the small components. 
2. add labels to world graph, run clustering or classification to get us sub graph?
'''
component_adjacency, component_index = skn.topology.get_largest_connected_component(directed_G_world.adjacency, return_labels=True)
print(component_adjacency.shape)
print(component_index.shape)

# %%
# Print world largest connected component
print(component_index)

# %%
# Update largest component indice to mysql table edges column largest_component.
mydb = mysql.connector.connect(
  host="localhost",
  user="jerry",
  password="password",
  database="pagenet"
)
mycursor = mydb.cursor()
sql_prefix = "update pages set largest_component=True where idx="
count = 0
for i in component_index:
    # if count > 60890100: print(i)
    sql = sql_prefix + str(i) + ";"
    mycursor.execute(sql)
    count += 1
    if count % 100000 == 0:
        mydb.commit()
        print(count, "record updated.")
mydb.commit()
print(count, "record updated.")
# %%
# Save connected component in world graph
component = skn.data.Bunch()
component.adjacency = component_adjacency
component.names = component_index
skn.data.save('largest_component_countries', component)
# %%
print(type(component))
print(type(component.adjacency))
print(type(component.names))
print(component)
print(component.adjacency)
# (0, 11319) should be (11319, 0) directed edge.
# %%
print(directed_G_world.adjacency)

# %%
edge_list = component_adjacency.toarray()
print(edge_list.shape)
print(edge_list)
# %%
