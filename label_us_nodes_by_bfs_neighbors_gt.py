# %% 
# Import packages
import csv
import pandas as pd
import numpy as np
import graph_tool.all as gt
import tqdm
# import networkx as nx
import os.path as osp
import mysql.connector
from pyparsing import counted_array, countedArray


# %% 
# Define get_edge function
def get_edge(file_path: str) -> pd.DataFrame:

    df = pd.read_csv(file_path, sep='\t', header=None)
    return df

# %%
# Load US largest connect component graph
edges_lgc_us = get_edge('./data/raw_dir/us_edges_lgc.csv')

# %%
# Undirected G in US lgc graph
undirected_lgc_us = gt.Graph(directed=False)
undirected_lgc_us.add_edge_list(edge_list=edges_lgc_us.values, 
                                hashed=False,
                                hash_type='int')
'''
<Graph object, undirected, with 60924673 vertices and 
84485587 edges, at 0x7f4e623d70d0>'''

# %%
# extract largest conponent from undirected_G_US
largest_component_us = gt.extract_largest_component(undirected_lgc_us)
print(largest_component_us)
'''
<GraphView object, undirected, with 5873395 vertices and 84480575 
edges, edges filtered by (<EdgePropertyMap object with value type 
'bool', for Graph 0x7f4e6241f210, at 0x7f4d9fb82650>, False), 
vertices filtered by (<VertexPropertyMap object with value type 
'bool', for Graph 0x7f4e6241f210, at 0x7f4e62433d10>, False), at 
0x7f4e6241f210>'''

# %%
# examine that the lgc nodes are idx
lgc_nodes_us = largest_component_us.get_vertices()
print(lgc_nodes_us.shape)
print(lgc_nodes_us)

# %%
# create a simple pointer lgc to largest_component_us
lgc = largest_component_us
print(lgc)

# %%
# Initialize mysql connector
mydb = mysql.connector.connect(
  host="localhost",
  user="jerry",
  password="password",
  database="pagenet"
)
mycursor = mydb.cursor()

# %%
# Store {state:label} {labe;:state} in state_label
sql = "select * from state_label;"
mycursor.execute(sql)
result = mycursor.fetchall()
state_label = {}
label_state = {}
for t in result:
    state, label = t
    state_label[state] = label
    label_state[label] = state
# print(state_label)
# print(label_state)

# %%
# Store {city: {state: label index}} dict
sql = "select * from us_city_state;"
mycursor.execute(sql)
result = mycursor.fetchall()
city_label_state = {}
for t in result:
    city, state = t
    city = city.lower()
    if state not in state_label:
        continue
    if city not in city_label_state:
        city_label_state[city] = {state_label[state]:state}
    else:
        city_label_state[city][state_label[state]] = state
print(city_label_state['Franklin'.lower()])
print(len(city_label_state))

# %%
# read id, idx, label, city, dup_states from us_pages_lgc.csv
lgc_nodes_info = []
idx_label = {}
idx_city = {}
with open('./data/raw_dir/us_pages_lgc.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    count = 0
    for r in reader:    
        id, idx, label, city, dup_states = r
        id, idx, label, dup_states = int(id), int(idx), int(label), int(dup_states)
        row = {'id':id, 'idx':idx, 'label':label, 'city':city.lower(), 'dup_states':dup_states}
        lgc_nodes_info.append(row)
        idx_label[idx] = label
        idx_city[idx] = city.lower()
        count += 1
        # if count == 10: break
print(len(lgc_nodes_info))

# %%
# change idx_label from label contains -1 to hop 2 bfs labels
hop_2_bfs_labels= pd.read_csv('./data/raw_dir/us_lgc_2_hop_bfs_voting_label.csv',
                      sep='\t',
                      header=None)
hop_2_bfs_label_count = {}
idx_label = {}
hop_2_bfs_labels = hop_2_bfs_labels.values
for n in range(5873395):
    i = int(hop_2_bfs_labels[n])
    idx_label[lgc_nodes_info[n]['idx']] = i
    if i not in hop_2_bfs_label_count:
        hop_2_bfs_label_count[i] = 1
    else:
        hop_2_bfs_label_count[i] += 1
l = sorted(hop_2_bfs_label_count.items(), key=lambda x:x[1], reverse=True)
print(l)
# print(idx_label.items())
# %%
# add properties to the lgc vertex.
print(lgc.vertex(60924672))

# %%
# define get_neighbors_most_label function among 1001 neighbors 

# Get bfs nodes array for 60924672 using lgc
# bfs_iter = gt.bfs_iterator(g=lgc, source=60924672, array=True)
# print(bfs_iter.shape)
# (5873394, 2)

def get_neighbors_most_label(node: int, hops: int = 2):
    '''Get the most frequent label of the neighbors with number of hops.
    '''
    count = 0
    bfs_iter = gt.bfs_iterator(g=lgc, source=node, array=False)
    vertex = lgc.vertex(node)
    neighbors, s, e = set(), set(), set()
    s.add(vertex)
    counting_array = [0 for i in range(53)]
    for edge in bfs_iter:
        start = edge.source()
        end = edge.target()
        # if gt.shortest_distance(lgc, vertex, end) == hops:
        #   break
        # if vertex == start:
        #     neighbors.add(end)
        # else:
        #     if start not in neighbors:
        #       break

        if start in s:
            e.add(end)
        else:
            hops -= 1
            if hops == 0: break
            s = e
            e = set()
            e.add(end)
        
        label = idx_label[int(end)]

        if label == -1: continue
        
        counting_array[label] += 1

        count += 1
        # if count == neighbor_count: break
    # print(count)
    # print(idx_city[node])
    candidate = city_label_state[idx_city[node]].keys()
    most, label = -1, -1
    for i in candidate:
        if most < counting_array[i]:
            most = counting_array[i]
            label = i
    arr_dict = {}
    for i in range(53):
        arr_dict[i] = counting_array[i]
    return label, candidate, arr_dict

# %%
print(get_neighbors_most_label(60924672))
print(get_neighbors_most_label(103))
print(get_neighbors_most_label(123))
print(get_neighbors_most_label(127))
print(get_neighbors_most_label(77))
print(get_neighbors_most_label(88))
# %%
# Update hop_2_bfs for largest component indice to mysql table us_edges column gt_lgc.
mydb = mysql.connector.connect(
  host="localhost",
  user="jerry",
  password="password",
  database="pagenet"
)
mycursor = mydb.cursor()

count = 0
for idx, label in idx_label.items():
    # if count > 60890100: print(i)
    sql = "update us_pages set hop_2_bfs={} where idx={};".format(label, idx)
    mycursor.execute(sql)
    count += 1
    if count % 100000 == 0:
        mydb.commit()
        print(count, "record updated.")
mydb.commit()
print(count, "record updated.")

# %%
for i in range(2, 28):
    for row in tqdm.tqdm(lgc_nodes_info):
        if row['label'] != -1: continue
        if row['dup_states'] != i: continue
        label = get_neighbors_most_label(row['idx'])
        row['label'] = label

    
# %%
# Update largest component indice to mysql table us_edges column gt_lgc.
mydb = mysql.connector.connect(
  host="localhost",
  user="jerry",
  password="password",
  database="pagenet"
)
mycursor = mydb.cursor()
sql_prefix = "update us_pages set gt_lgc=True where idx="
count = 0
for i in lgc_nodes_us:
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
