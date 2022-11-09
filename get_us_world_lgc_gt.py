# %% 
# Import packages
import pandas as pd
import numpy as np
import graph_tool.all as gt
# import networkx as nx
import os.path as osp
import mysql.connector


# %% 
# define get_edge()
# def get_node(file_path: str = None) -> list:
#     df = pd.read_csv(file_path, sep='\t', header=None)
#     x = [i for i in range(num_nodes)]
    
#     return x

def get_edge(file_path: str) -> pd.DataFrame:

    df = pd.read_csv(file_path, sep='\t', header=None)
    return df


# %%
# Load U.S. graph
# directed_G = skn.data.load_edge_list(file='./data/raw_dir/us_edges.csv',
#                                      directed=True,
#                                      delimiter='\t')

# undirected_G = skn.data.load_edge_list(file='./data/raw_dir/us_edges.csv',
#                                        directed=False,
#                                        delimiter='\t')


# %%
# Print U.S. graph
# print(directed_G)
# print(undirected_G)
# directed_labels = skn.topology.get_connected_components(directed_G.adjacency)
# undirected_labels = skn.topology.get_connected_components(undirected_G.adjacency)
# print(histogram(directed_labels))
# 5873395:0 nodes in the largest connected component, 2-2608 small component(1-4 nodes)
# 6163640 nodes in total US,290245
'''TODO get edges one node is in the us_pages?
insert into one_node_us select * from edges where (page_id in (select id from us_pages) and likes_page not in (select id from us_pages)) or (page_id not in (select id from us_pages) and likes_page in (select id from us_pages));
'''
# %%



# %%
# Load world graph
# edge_list = np.genfromtxt('./data/raw_dir/edge_index.csv', delimiter='\t')
# print(edge_list.shape)
edges = get_edge('./data/raw_dir/edge_index.csv')

# %%
# directed G in world graph
directed_G_world = gt.Graph(directed=True)
directed_G_world.add_edge_list(edge_list=edges.values, 
                               hashed=False,
                               hash_type='int')
print(directed_G_world)
# 60924683 nodes in world, 789494545 edges in world
# %%
# Check in and out edges.
print(len(directed_G_world.get_out_edges(0)))
# 192. 
# if hashed=True, 0 to 1-192. new index for all nodes. 
print(len(directed_G_world.get_in_edges(0)))
# 1476

# %%
# Largest component
largest_component = gt.extract_largest_component(directed_G_world)
print(largest_component)
'''
<GraphView object, directed, with 20448405 vertices and 594349701 
edges, edges filtered by (<EdgePropertyMap object with value type 
'bool', for Graph 0x7f50b6445210, at 0x7f4ff3e50f90>, False), 
vertices filtered by (<VertexPropertyMap object with value type 
'bool', for Graph 0x7f50b6445210, at 0x7f50b644cfd0>, False), at 
0x7f50b6445210>'''

# %%
# Undirected G in world graph
undirected_G_world = gt.Graph(directed=False)
undirected_G_world.add_edge_list(edge_list=edges.values, 
                               hashed=False,
                               hash_type='int')
print(undirected_G_world)
# 60924683 nodes in world, 789494545 edges in world

# %%
# Check in and out edges
print(undirected_G_world.get_out_edges(0))
# 1668 = 192 + 1476
print(len(undirected_G_world.get_in_edges(0)))
# 0

# %%
# extract largest conponent from undirected_G_world
largest_component = gt.extract_largest_component(undirected_G_world)
print(largest_component)
# 60890130:0 nodes in the largest connected component, 2-15000 small component(1-28 nodes)
# 60924683 nodes in world, 34553, 789494545 edges in world
'''
<GraphView object, undirected, with 60890130 vertices and 789473610 edges, edges filtered 
by (<EdgePropertyMap object with value type 'bool', for Graph 0x7f94dd595050, at 0x7f94dd598bd0>, 
False), vertices filtered by (<VertexPropertyMap object with value type 'bool', for Graph 
0x7f94dd595050, at 0x7f94dc84d4d0>, False), at 0x7f94dd595050>'''

# %%
# Label the largest component in the graph.
'''TODO
1. remove all the small components. 
2. add labels to world graph, run clustering or classification to get us sub graph?
'''
labels = gt.label_largest_component(undirected_G_world)
# %%
# Print Label the largest component
print(type(labels.a)) 
# graph_tool.PropertyArray
print(labels.get_array().shape)
# (60924683,). labels.get_array() ndarray
print(labels.get_array().sum())
# 60890130
# %%
# Update largest component indice to mysql table edges column largest_component.

# mydb = mysql.connector.connect(
#   host="localhost",
#   user="jerry",
#   password="password",
#   database="pagenet"
# )
# mycursor = mydb.cursor()

# count = 0

# for i in labels.a:
#     # world 60924683
#     sql = "update pages set gt_lg_cmpt={} where idx={}".format(i, count)
#     mycursor.execute(sql)
#     count += 1
#     if count % 100000 == 0:
#         mydb.commit()
#         print(count, "record updated.")
# mydb.commit()
# print(count, "record updated.")

# %%
# Load US graph
edges_us = get_edge('./data/raw_dir/us_edges.csv')

# %%
# Undirected G in US graph
undirected_G_US = gt.Graph(directed=False)
undirected_G_US.add_edge_list(edge_list=edges_us.values, 
                               hashed=False,
                               hash_type='int')
print(undirected_G_US)
'''
<Graph object, undirected, with 60924673 vertices and 
84485587 edges, at 0x7f4e623d70d0>'''

# %%
# extract largest conponent from undirected_G_US
largest_component_US = gt.extract_largest_component(undirected_G_US)
print(largest_component_US)
'''
<GraphView object, undirected, with 5873395 vertices and 84480575 
edges, edges filtered by (<EdgePropertyMap object with value type 
'bool', for Graph 0x7f4e6241f210, at 0x7f4d9fb82650>, False), 
vertices filtered by (<VertexPropertyMap object with value type 
'bool', for Graph 0x7f4e6241f210, at 0x7f4e62433d10>, False), at 
0x7f4e6241f210>'''

# %%
# Label the largest component in the US graph.

us_labels = gt.label_largest_component(undirected_G_US)

# %%
# %%
# Print Label the largest component in US
print(us_labels.a) 
# graph_tool.PropertyArray
print(us_labels.get_array().shape)
# (60924673,) in US graph
# (60924683,) in total US graph. labels.get_array() ndarray
print(us_labels.get_array().sum())
# 5873395

# %%
lgc_nodes_us = largest_component_US.get_vertices()
print(lgc_nodes_us.shape)
print(lgc_nodes_us)

# %%
# Update largest component indice to mysql table us_edges column gt_lgc.

# mydb = mysql.connector.connect(
#   host="localhost",
#   user="jerry",
#   password="password",
#   database="pagenet"
# )
# mycursor = mydb.cursor()
# sql_prefix = "update us_pages set gt_lgc=True where idx="
# count = 0
# for i in lgc_nodes_us:
#     # if count > 60890100: print(i)
#     sql = sql_prefix + str(i) + ";"
#     mycursor.execute(sql)
#     count += 1
#     if count % 100000 == 0:
#         mydb.commit()
#         print(count, "record updated.")
# mydb.commit()
# print(count, "record updated.")
# %%
