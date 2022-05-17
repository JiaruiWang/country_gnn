# %%
# import packages
import igraph as ig
import pandas as pd
import random
import csv, tqdm
import torch
import mysql.connector
# %%
# load edge list csv
edge_df = pd.read_csv('./data/raw_dir/us_edges_lgc_header.csv',
                      sep='\t',
                      header=None)

# %%
# load page csv
page_df = pd.read_csv('./data/raw_dir/us_pages_lgc_header.csv',
                      sep='\t',
                      header=None)
page_df.columns = ['id', 'idx', 'state_label', 'city', 'dup_states_count']
# # %%
# print(page_df.shape)
# print(page_df[0:10])
# print(edge_df.shape)
# print(edge_df[0:10])
# print(page_df[2591:2592])
# print(page_df[2668:2669])
# print(page_df[4757:4758])
# %%
page_df = page_df.loc[:, ['idx', 'id', 'city', 'dup_states_count', 'state_label']]

# %%
# load louvain clustering labels (191 clusters) from csv file
louvain_191_df = pd.read_csv('./data/raw_dir/us_lgc_louvain_clusters.csv',
                      sep='\t',
                      header=None)
clusters_labels = {}

# %%
page_df.insert(5, 'louvain_191', louvain_191_df)

# %%
hop_2_bfs_labels= pd.read_csv('./data/raw_dir/us_lgc_2_hop_bfs_voting_label.csv',
                      sep='\t',
                      header=None)
page_df.insert(6, 'hop_2_bfs', hop_2_bfs_labels)
# %%
# load page df and edge df into directed graph
page_df_simp = page_df.loc[:, ['idx', 'state_label', 'louvain_191', 'hop_2_bfs']]

g = ig.Graph.DataFrame(edges=edge_df, directed=True, vertices=page_df_simp, use_vids=False)
# %%
print(g.vcount())
print([v for v in g.vs[0:10]])


# %%
# # create anchor nodes. First 52 nodes are university node from 52 states as one node anchor sets.
# one_node_anchor_pages = {
# 'The University of Alabama'                                             :'22360227546',
# 'UAA: University of Alaska Anchorage'                                   :'57576345235',
# 'The University of Arizona'                                             :'6096033309',
# 'University of Arkansas at Pine Bluff'                                  :'228480367906',
# 'California State University Maritime Academy - Cal Maritime'           :'166121435148',
# 'California State University, Los Angeles'                              :'123775222323',
# 'Colorado State University'                                             :'136298855614',
# 'Geography Department, University of Connecticut'                       :'100344880314319',
# 'University of Delaware Department of Music'                            :'134075779247',
# 'University of Florida'                                                 :'44496359631',
# 'University of Georgia'                                                 :'21657666681',
# 'University of Hawaii at Manoa'                                         :'42509314000',
# 'University of Idaho'                                                   :'8990210995',
# 'University of Illinois Computer Science'                               :'29311728027',
# 'University of Indianapolis'                                            :'167322460506',
# 'University of Iowa'                                                    :'41486668846',
# 'The University of Kansas'                                              :'44728562961',
# 'University of Kentucky College of Engineering'                         :'42873414959',
# 'University of Louisiana Monroe'                                        :'20058920070',
# 'University of Maine'                                                   :'6713903971',
# 'University of Maryland'                                                :'16972274487',
# 'University of Massachusetts'                                           :'72450458708',
# 'University of Michigan'	                                            :'21105780752',
# 'University of Minnesota'	                                            :'93415311269',
# 'University of Mississippi ~ Ole Miss '	                                :'56900048284',
# 'University of Missouri College of Arts and Science'                    :'55353702356',
# 'University of Montana'	                                                :'69748712645',
# 'University of Nebraska-Lincoln'	                                    :'7293092326',
# 'University of Nevada, Reno'	                                        :'8455795859',
# 'University of New Hampshire'	                                        :'56182416236',
# 'William Paterson University of New Jersey'	                            :'57701144178',
# 'The University of New Mexico - UNM'	                                :'21749746264',
# 'New York University'	                                                :'103256838688',
# 'The University of North Carolina at Chapel Hill'	                    :'140105122708',
# 'University of North Dakota'                                            :'9275555010',
# 'The Ohio State University'	                                            :'6711658857',
# 'The University of Oklahoma'	                                        :'10049970637',
# 'University of Oregon'	                                                :'10515469841',
# 'University of Pennsylvania'	                                        :'98508878775',
# 'University of Rhode Island'	                                        :'60011053571',
# 'University of South Carolina'	                                        :'41091071801',
# 'The University of South Dakota'                                        :'267498065938',
# 'University of Tennessee, Knoxville'	                                :'89769773068',
# 'The University of Texas at Austin'                                     :'245640871929',
# 'The University of Utah'                                                :'7576735635',
# 'University of Vermont'                                                 :'31152992636',
# 'University of Virginia'                                                :'12527153330',
# 'University of Washington'                                              :'8829726273',
# 'Georgetown University'                                                 :'8825331245',
# 'West Virginia University'                                              :'28580386540',
# 'University of Wisconsin Milwaukee'                                     :'14716860095',
# 'University of Wyoming College of Engineering and Applied Science'      :'174371532597',
# }

# mydb = mysql.connector.connect(
#   host="localhost",
#   user="jerry",
#   password="password",
#   database="pagenet"
# )
# mycursor = mydb.cursor()
# anchor_nodes = []
# for id in one_node_anchor_pages.values():
#     sql = 'select idx from us_pages where id={};'.format(id)
#     mycursor.execute(sql)
#     result = mycursor.fetchall()
#     anchor_nodes.append(result[0][0])
# print(anchor_nodes)

# # random select 26 two nodes set, 13 four nodes set, 6 eight nodes set, 3 sixteen nodes set,
# # 1 thirty two nodes set. total 232 random nodes other than first 52 nodes.

# rand_list = random.sample(range(page_df_simp.shape[0]), k=250)
# print(rand_list)
# print(len(anchor_nodes))
# count = 0
# for i in rand_list:
#     if len(anchor_nodes) == 284:
#         print('break at', count)
#         break
#     if page_df_simp.iloc[i]['idx'] in anchor_nodes:
#         print("find dup")
#         continue
#     else:
#         anchor_nodes.append(page_df_simp.iloc[i]['idx'])
#     count+=1
# print(len(anchor_nodes))
# print(anchor_nodes)

# # print the random nodes idx list to random_node_idx.csv
# with open('./data/raw_dir/random_node_idx.csv', 'w') as csvfile:
#     writer = csv.writer(csvfile)
#     for r in anchor_nodes:
#         # row = {'id':id, 'idx':idx, 'label':label, 'city':city.lower(), 'dup_states':dup_states}
#         row = str(r) 
#         writer.writerow([row])

# %%
# read in random node idx from file generated by above cell.
anchor_nodes = []
with open('./data/raw_dir/random_node_idx.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for r in reader:
        idx = int(r[0]) 
        anchor_nodes.append(idx)
# 284 * 3 needs two much memory, use 52 nodes only
anchor_nodes = anchor_nodes[0:52]
print(len(anchor_nodes))
print(anchor_nodes)
# %%
# update bfs distance
out_vertex_list, out_start_indices, out_parents = g.bfs(vid=anchor_nodes[0], mode='out')
print(len(out_vertex_list), len(out_start_indices), len(out_parents))
print(out_start_indices)
in_vertex_list, in_start_indices, in_parents = g.bfs(vid=anchor_nodes[0], mode='in')
print(len(in_vertex_list), len(in_start_indices), len(in_parents))
print(in_start_indices)
all_vertex_list, all_start_indices, all_parents = g.bfs(vid=anchor_nodes[0], mode='all')
print(len(all_vertex_list), len(all_start_indices), len(all_parents))
print(all_start_indices)
# %%
# test bfsiter
iter = g.bfsiter(vid=anchor_nodes[0], mode='all', advanced=True)
print(type(iter))
count = 0
for v, dis, p in iter:
    if count % 1000000 == 0: print(count, dis, v, p)
    count += 1
print(count)
# %%
path = g.get_shortest_paths(anchor_nodes[0], to=5577644, mode='all', output='vpath')
print(path)
# %%
print(g.vs.select(name=str(anchor_nodes[0])))
# %%
# create anchor distance list and idx to index of array dict
idx_index = {}
dist_list = []
idx_list = page_df_simp.loc[:, 'idx'].values.tolist()
print(len(idx_list))
print(idx_list[0:10])
print(type(idx_list))
for i in tqdm.tqdm(range(len(idx_list))):
    index_of_array, idx = i, idx_list[i]
    idx_index[idx] = index_of_array
    dist_list.append({idx:[0 for i in range(156)]})
for i in range(len(idx_list)):
    assert list(dist_list[i].keys())[0] == idx_list[i]
    assert idx_index[idx_list[i]] == i

# %%
# 
node = g.vs.select(name=20880852)
print(node[0])
# %%
nei = g.neighbors(node[0], mode='in')
# %%
print(len(nei))
# %%

# %%
for v, dis, p in iter:
    print(v)
    print(p)
    pass
# %%
#
anchor_nodes[7] = 2889465
for i in tqdm.tqdm(range(len(anchor_nodes))):
    # i: 0 ~ 51
    # if i == 7: continue
    vid = anchor_nodes[i]
    modes = ['in', 'out', 'all']
    for j in range(len(modes)):
        # j: 0 ~ 2
        mode = modes[j]
        col = i * 3 + j
        iter = g.bfsiter(vid=vid, mode=mode, advanced=True)
        for v, dis, p in iter:
            idx = v['name']
            index = idx_index[idx]
            dist_list[index][idx][col] = 1/(dis+1)

# %%
print(len(dist_list), len(list(dist_list[0].values())[0]))
print(dist_list[0])
# %%
# %%
# print the result label to dist_list_idx+52(in_out_all).csv
with open('./data/raw_dir/normalized_dist_list_idx+52_in_out_all.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for d in dist_list:
        key = list(d.keys())[0]
        values = list(d.values())[0]
        row_list = [key] + values
        # print(row_list)
        # print(len(row_list))
        # row = ",".join(str(i) for i in row_list)
        # print(row)
        # break
        writer.writerow(row_list)
# %%
# print the result label to dist_list_idx+52(in_out_all).csv
with open('./data/raw_dir/dist_list_idx+52_in_out_all.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for d in dist_list:
        key = list(d.keys())[0]
        values = list(d.values())[0]
        row_list = [key] + values
        # print(row_list)
        # print(len(row_list))
        # row = ",".join(str(i) for i in row_list)
        # print(row)
        # break
        writer.writerow(row_list)

# %%
df = pd.read_csv('./data/raw_dir/normalized_dist_list_idx+52_in_out_all.csv', sep=',', header=None)
print(df.iloc[0:3])
dfd = df.iloc[:, 1:157]
print(dfd.shape)
print(dfd.iloc[0:3])
# %%
print(type(dfd.values[0,4]))
t=torch.Tensor([dfd.values[0,4]])
print(type(t))
print(torch.isinf(t))
# %%
