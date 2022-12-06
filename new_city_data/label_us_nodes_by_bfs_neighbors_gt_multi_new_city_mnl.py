# %% 
# Import packages
import csv
import multiprocessing as mp
from os import dup
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
edges_lgc_us = get_edge('../data/raw_dir/us_edges_lgc.csv')

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
# load states name create state label dict
# Store {state:label} {label:state} in state_label
state_label = {}
label_state = {}
statenamefile = './states.csv'
with open(statenamefile, 'r') as statename:
    cityreader = csv.reader(statename)
    counter = 0
    for row in cityreader:
        state = row[0]
        state_label[state] = counter
        label_state[counter] = state
        counter += 1

print(state_label)
print(label_state)

#%% 
# load city data in the value list, column name in the header map.
import csv
is_header = True
city_headers = {}
cities = {}
cityfilename = './us-cities-30k-noPR.csv'
with open(cityfilename, 'r') as cityfile:
    cityreader = csv.reader(cityfile)
    for row in cityreader:
        # load header
        if is_header:
            header_idx = 0
            for r in row:
                city_headers[r] = header_idx
                header_idx += 1
            is_header = False
            print("city headers", city_headers)
            continue
        # print(row)
        
        # store key value pair

        key = (row[city_headers['city']],row[city_headers['state_name']])

        if key in cities:
            if int(cities[key][city_headers['population']]) < int(row[city_headers['population']]):
                cities[key] = row
                # print(row)
        else:
            cities[key] = row
print(len(cities))
print(list(cities.items())[1])

# %%
# Store {city: {state: label index}} dict

cities_dup_states = {}
for key in cities.keys():
    city, state = key
    city = city.lower()
    if city not in cities_dup_states:
        cities_dup_states[city] = {state_label[state]: state}
    else:
        cities_dup_states[city][state_label[state]] = state
print(len(cities_dup_states))
print(len(list(cities_dup_states.values())[0]))
print(list(cities_dup_states.items())[0])

largest_dup_count = 0
for k, dictionary in cities_dup_states.items():
    if largest_dup_count < len(dictionary):
        largest_dup_count = len(dictionary)
print("largest_dup_count",largest_dup_count)


# %%
# read id, idx, label, city, dup_states from us_pages_lgc.csv
lgc_nodes_info = []
idx_label = {}
idx_city = {}
idx_index = {}
count_true_label = 0
with open('../data/raw_dir/us_pages_lgc.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    count = 0
    for r in reader:    
        id, idx, label, city, dup_states = r
        city = city.lower()
        id, idx, label, dup_states = int(id), int(idx), int(label), int(dup_states)
        new_dup_states = 0
        new_label = -1
        if city in cities_dup_states:
            new_dup_states = len(cities_dup_states[city])
            if new_dup_states == 1:
                new_label = list(cities_dup_states[city].keys())[0]
        if new_label != -1:
            count_true_label += 1
        row = {'id':id, 'idx':idx, 'label':new_label, 'city':city, 'dup_states':new_dup_states}
        lgc_nodes_info.append(row)
        idx_label[idx] = new_label
        idx_city[idx] = city
        idx_index[idx] = count
        count += 1
        # if count == 10: break
print(len(lgc_nodes_info))
print(count_true_label)
# 5873395
# 2114133
# 2114133 + 33266(washington) = 2147399
#%%
print(lgc_nodes_info[0:10])

# %%
# create shared_labels for processes
shared_labels = mp.Array('i', range(len(lgc_nodes_info)), lock=False)
for i in range(len(lgc_nodes_info)):
    shared_labels[i] = lgc_nodes_info[i]['label']
print(shared_labels[0:100])
# %%
# add properties to the lgc vertex.
print(lgc.vertex(60924672))

# %%
# define get_neighbors_most_label function
# for dup_states = 2 ~ 27

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
    s, e = set(), set()
    s.add(vertex)
    counting_array = [0 for i in range(50)]
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
        
        label = shared_labels[idx_index[int(end)]]

        if label == -1: continue
        
        counting_array[label] += 1

        count += 1
        # if count == neighbor_count: break
    # print(count)
    # print(idx_city[node])

    # find the candidate state with largest neighbor weight
    candidate = cities_dup_states[idx_city[node]].keys()
    most, label = -1, -1
    for i in candidate:
        if most < counting_array[i]:
            most = counting_array[i]
            label = i
    # arr_dict = {}
    # for i in range(53):
    #     arr_dict[i] = counting_array[i]
    return label, candidate#, arr_dict


def get_neighbors_most_label_for_dup_0(node: int, hops: int = 2):
    '''Get the most frequent label of the neighbors with number of hops.
    '''
    count = 0
    bfs_iter = gt.bfs_iterator(g=lgc, source=node, array=False)
    vertex = lgc.vertex(node)
    s, e = set(), set()
    s.add(vertex)
    counting_array = [0 for i in range(50)]
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
        
        label = shared_labels[idx_index[int(end)]]

        if label == -1: continue
        
        counting_array[label] += 1

        count += 1
        # if count == neighbor_count: break
    # print(count)
    # print(idx_city[node])
    # 
    # since the dup_state = 0, the city is not in the idx_city,
    # no respect candidate states, all the states are candidates.
    candidate = [i for i in range(50)]
    most, label = -1, -1
    for i in candidate:
        if most < counting_array[i]:
            most = counting_array[i]
            label = i
    # arr_dict = {}
    # for i in range(53):
    #     arr_dict[i] = counting_array[i]
    return label, candidate#, arr_dict

# %%
print(get_neighbors_most_label(60924672))

# %%
# get the split range for range(0, 5873395)
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

# range_list = list(split(range(5873395), 16))
# range_list = list(split(range(30), 16))

# %% split calculate nodes equaliy

def split_workload_equaliy(dup_count):
    index_lists = []
    total_index = []
    for i in range(len(lgc_nodes_info)):
        if dup_count == lgc_nodes_info[i]['dup_states']:
            total_index.append(i)
    index_lists = list(split(total_index, 16))
    return index_lists


# %%
# first round: define multi threading function for the 1st round bfs label assignment
def multi_th_func(split_range, dup_count):
    for n in split_range:
        row = lgc_nodes_info[n]
        if shared_labels[n] != -1: continue
        if row['dup_states'] != dup_count: continue
        if dup_count != 0:
            label, _= get_neighbors_most_label(row['idx'])
        else:
            label, _ = get_neighbors_most_label_for_dup_0(row['idx'])
        shared_labels[n] = label
    


# %%
# first round: run multi threading for dup_states = 2 ~ 28

for dup_count in range(2, 29):
    print("Starting 1st round multi-threading for dup_states = ", dup_count)
    range_list = split_workload_equaliy(dup_count=dup_count)
    # range_list = list(split(range(30), 16))
    print_str = "Len for range lists: "
    for i in range(16):
        print_str = print_str + " " + str(len(range_list[i]))
    print(print_str)
    pr_list = []
    for i in range(16):
        pr = mp.Process(target=multi_th_func,
                        args=[range_list[i], dup_count],
                        name='th_'+str(i))
        pr_list.append(pr)
    
    for th in pr_list:
        th.start()
    for th in pr_list:
        th.join()
    print("Ending 1st round multi-threading for dup_states = ", dup_count)


# %%
# first round: run multi threading for dup_states = 0 
dup_count = 0
print("Starting multi-threading for dup_states = ", dup_count)
range_list = split_workload_equaliy(dup_count=dup_count)
# range_list = list(split(range(30), 16))
print_str = "Len for range lists: "
for i in range(16):
    print_str = print_str + " " + str(len(range_list[i]))
print(print_str)
pr_list = []
for i in range(16):
    pr = mp.Process(target=multi_th_func,
                    args=[range_list[i], dup_count],
                    name='th_'+str(i))
    pr_list.append(pr)

for th in pr_list:
    th.start()
for th in pr_list:
    th.join()
print("Ending multi-threading for dup_states = ", dup_count)

# %%
print(shared_labels[0:40])
print(list(idx_label.values())[0:40])

# %%
# first round: print the result label to us_lgc_2_hop_bfs_voting_label.csv for the first round
first_round_label_result = []
with open('./new_cities_2hop_bfs_1_round_mnl_id_idx_city_dupStates_trueLabel_1stRoundLabel.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(len(shared_labels)):
        first_round_label_result.append(shared_labels[i])
        node_info = lgc_nodes_info[i]
        row = [str(node_info['id']), str(node_info['idx']), node_info['city'],
               str(node_info['dup_states']), str(node_info['label']), str(shared_labels[i])]
        writer.writerow(row)
# %%
print(len(shared_labels), len(lgc_nodes_info))
# %%
# # first round: update us_pages for the first round
# count = 0
# for i in range(len(lgc_nodes_info)):
#     idx, hop2bfs = lgc_nodes_info[i]['idx'], shared_labels[i]
#     sql = 'update us_pages set hop2bfs={} where idx={}'.format(hop2bfs, idx)
#     mycursor.execute(sql)
#     count += 1
#     if count % 100000 == 0:
#         mydb.commit()
#         print(count, "record updated.")
# mydb.commit()
# print(count, "record updated.")
# %%
# 2nd round: for hop 2 bfs
def multi_th_func_no_neg_1(split_range, dup_count):
    for n in split_range:
        row = lgc_nodes_info[n]
        if row['label'] != -1: continue
        '''should use row['label'] = -1 to replace shared_labels[n] != -1.
        But shared_labels[n] != -1 didn't change the original ground truth labels (labels not equal to -1,
        which needs to be predicted). Means this 2 hop bfs labeling has very good result on the ground truth
        data, 100% correctness.  Wrong: because labels[n] != -1 means the dup states for the city is only one, 
        which means the candidatte state is only 1 in the get_neighbors_most_label(). 
        In 2nd round there should be no -1 in shared_labels, all labels != -1, all shared_labels will be skipped,
        must use row['label'] != -1
        '''
        # should use row['label'] to replace shared_labels[n] != -1 this didn't change the original ground truth label.
        # if shared_labels[n] != -1: continue 
        if row['dup_states'] != dup_count: continue
        if dup_count != 0:
            label, _= get_neighbors_most_label(row['idx'])
        else:
            label, _ = get_neighbors_most_label_for_dup_0(row['idx'])
        shared_labels[n] = label
# %%
# run multi threading for dup_states = 2-28
for dup_count in range(2, 29):
    print("Starting 2nd iteration multi-threading for dup_states = ", dup_count)
    range_list = split_workload_equaliy(dup_count=dup_count)
    # range_list = list(split(range(30), 16))
    print_str = "Len for range lists: "
    for i in range(16):
        print_str = print_str + " " + str(len(range_list[i]))
    print(print_str)
    pr_list = []
    for i in range(16):
        pr = mp.Process(target=multi_th_func_no_neg_1,
                        args=[range_list[i], dup_count],
                        name='th_'+str(i))
        pr_list.append(pr)
    
    for th in pr_list:
        th.start()
    for th in pr_list:
        th.join()
    print("Ending multi-threading for dup_states = ", dup_count)


# %%
# run multi threading for dup_states = 0
dup_count = 0
print("Starting 2nd iteration multi-threading for dup_states = ", dup_count)
range_list = split_workload_equaliy(dup_count=dup_count)
# range_list = list(split(range(30), 16))
print_str = "Len for range lists: "
for i in range(16):
    print_str = print_str + " " + str(len(range_list[i]))
print(print_str)
pr_list = []
for i in range(16):
    pr = mp.Process(target=multi_th_func_no_neg_1,
                    args=[range_list[i], dup_count],
                    name='th_'+str(i))
    pr_list.append(pr)

for th in pr_list:
    th.start()
for th in pr_list:
    th.join()
print("Ending multi-threading for dup_states = ", dup_count)
# %%
print(shared_labels[0:40])
print(list(idx_label.values())[0:40])

# %%
# print the result label to us_lgc_2_hop_bfs_voting_label.csv
with open('./new_cities_2hop_bfs_2_round_mnl_id_idx_city_dupStates_trueLabel_2ndRoundLabel_1stRoundLabel.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(len(shared_labels)):
        node_info = lgc_nodes_info[i]
        row = [str(node_info['id']), str(node_info['idx']), node_info['city'], str(node_info['dup_states']), 
               str(node_info['label']), str(shared_labels[i]), str(first_round_label_result[i])]
        writer.writerow(row)
# %%
# # 2nd round: update us_pages 
# count = 0
# for i in range(len(lgc_nodes_info)):
#     idx, hop2bfs_2nd_change = lgc_nodes_info[i]['idx'], shared_labels[i]
#     sql = 'update us_pages set hop2bfs_2nd_change={} where idx={}'.format(hop2bfs_2nd_change, idx)
#     mycursor.execute(sql)
#     count += 1
#     if count % 100000 == 0:
#         mydb.commit()
#         print(count, "record updated.")
# mydb.commit()
# print(count, "record updated.")
# %%
