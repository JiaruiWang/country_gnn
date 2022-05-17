# %%
# import packages
from dis import code_info
import igraph as ig
import pandas as pd
import tqdm

# %%
# load edge list csv
edge_df = pd.read_csv('./data/raw_dir/us_edges_lgc_header.csv',
                      sep='\t',
                      header=None)
# %%
print(edge_df.shape)
print(edge_df[0:10])
# %%
# load page csv
page_all_df = pd.read_csv('./data/raw_dir/us_pages_lgc_header.csv',
                      sep='\t',
                      header=None)
page_all_df.columns = ['id', 'idx', 'state_label', 'city', 'dup_states_count']
# %%

page_df = page_all_df.loc[:, ['idx']]
print(page_df.shape)
print(page_df[0:10])
# %%
# load louvain clustering labels (191 clusters) from csv file
# louvain_191_df = pd.read_csv('./data/raw_dir/us_lgc_louvain_clusters.csv',
#                       sep='\t',
#                       header=None)
# clusters_labels = {}
# for n in louvain_191_df.values:
#     i = int(n)
#     if i not in clusters_labels:
#         clusters_labels[i] = 1
#     else:
#         clusters_labels[i] += 1

# l = sorted(clusters_labels.items(), key=lambda x:x[1], reverse=True)
# print(l)
# print(louvain_191_df.shape)
# print(louvain_191_df[0:10])
# # %%
# page_df.insert(5, 'louvain_191', louvain_191_df)

# %%
hop_2_bfs_labels= pd.read_csv('./data/raw_dir/us_lgc_2_hop_bfs_voting_label_correct_2nd.csv',
                      sep='\t',
                      header=None)
hop_2_bfs_label_count = {}
for n in hop_2_bfs_labels.values:
    i = int(n)
    if i not in hop_2_bfs_label_count:
        hop_2_bfs_label_count[i] = 1
    else:
        hop_2_bfs_label_count[i] += 1
l = sorted(hop_2_bfs_label_count.items(), key=lambda x:x[1], reverse=True)
print(l)
# print(hop_2_bfs_label_count)
page_df.insert(1, 'hop_2_bfs', hop_2_bfs_labels)
# %%
print(page_df.shape)
print(page_df[0:10])

# %%
# create color dict for visualization
color_palette = ig.drawing.colors.ClusterColoringPalette(54)
color_label_order = [4,8,32,44,13,48,38,30,22,2,51,33,18,21,5,
                     9,14,36,47,41,20,35,45,23,11,15,0,37,12,
                     16,31,25,43,26,24,19,1,6,17,3,42,29,46,40,
                     52,50,27,49,34,28,7,10,39,-1]
color_dict = {}
for i in range(54):
    color_dict[color_label_order[i]] = color_palette.get(i)
print(color_dict)

# %%
# visualization style. TODO:too slow, need to rewrite
color_list = []
count = 0
for i in tqdm.tqdm(range(5873395)):
    color_list.append(color_dict[page_df.iloc[i][1]])
    count += 1
    # if count == 5: break
# %%
print(len(color_list))
color_df = pd.Series(color_list).to_numpy()
print(color_df.shape)
print(type(color_df))
print(color_df[0:10])
page_df.insert(2, 'color', color_df)
# %%
print(page_df[0:10])
# %%
# load page df and edge df into directed graph
g = ig.Graph.DataFrame(edges=edge_df, directed=True, vertices=page_df, use_vids=False)
# %%
print(g.vcount())
print([v for v in g.vs[0:10]])
# %%
print(g.diameter(directed=True, unconn=False))
print(g.diameter(directed=False, unconn=False))

# %%
visual_style = {}
visual_style['vertex_color'] = g.vs['color']
visual_style['vertex_label'] = g.vs['hop_2_bfs']
# %%
visual_style["layout"] = g.layout("lgl")

# %%
true_label_with_null = visual_style['vertex_color']
hop_2_bfs_color = [color_dict[label] for label in g.vs['hop_2_bfs']]
# %%
ig.plot(g, **visual_style)
# %%
