# %%
# Import packages
from os import setegid
import csv
from tkinter.ttk import LabeledScale
import sknetwork as skn
import numpy as np
import pandas as pd
from sknetwork.data import karate_club, painters, movie_actor
from sknetwork.classification import KNN, Propagation
from sknetwork.embedding import GSVD, LouvainNE, Spectral
from sknetwork.clustering import Louvain
# %%
# Load us graph
directed = skn.data.load_edge_list(file='./data/raw_dir/us_edges_lgc.csv',
                                   directed=True,
                                   delimiter='\t')

# undirected = skn.data.load_edge_list(file='./data/raw_dir/us_edges_lgc.csv',
#                                      directed=False,
#                                      delimiter='\t')
# %%
# Print graph
print(directed)
# print(undirected)



# # %%
# seeds = {}
# with open('./data/raw_dir/us_pages_lgc.csv', 'r') as csvfile:
#     reader = csv.reader(csvfile, delimiter='\t')    
#     count  = -1
#     for r in reader:    
#         id, idx, label, city, dup_states = r
#         label = int(label)
#         count+=1
#         if label == -1: continue
#         seeds[count] = label
        
#         # if count == 10: break

# print(count)
# print(len(seeds))
# print(list(seeds.items())[0:10],list(seeds.items())[2331777:2331787])
# %%

# %%
louvain_for_clustering = Louvain()

louvain_clusters = louvain_for_clustering.fit_transform(directed.adjacency)
# %%
# %%
print(louvain_clusters.shape)
# %%
print(type(louvain_clusters))
# %%
# pd.DataFrame(labels).to_csv('./data/raw_dir/us_lgc_knn_gsvd_8_20.csv')
# np.savetxt(fname='./data/raw_dir/us_lgc_louvain_clusters.csv', 
#            X=louvain_clusters,
#            delimiter='\t') 
with open('./data/raw_dir/us_lgc_louvain_clusters.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    for r in louvain_clusters:
        row = str(r) 
        writer.writerow([row])
# %%

clusters = {}
for i in louvain_clusters:
    if i not in clusters:
        clusters[i] = 1
    else:
        clusters[i] += 1
l = sorted(clusters.items())
print(l)

# %%
# visualization
from IPython.display import SVG
image = skn.visualization.svg_graph(directed.adjacency, labels=louvain_clusters)
SVG(image)
# %%
