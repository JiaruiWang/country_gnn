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



# %%
seeds = {}
with open('./data/raw_dir/us_pages_lgc.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, csvfile, delimiter='\t')
    count  = -1
    for r in reader:    
        id, idx, label, city, dup_stats = r
        label = int(label)
        count+=1
        if label == -1: continue
        seeds[count] = label
        
        # if count == 10: break

print(count)
print(len(seeds))
print(list(seeds.items())[0:10],list(seeds.items())[2331777:2331787])
# # %%
# knn = KNN(GSVD(8), n_neighbors=20)
# labels = knn.fit_transform(directed.adjacency, seeds)
# # %%
# print(labels.shape)
# # %%
# print(type(labels))
# # %%
# # pd.DataFrame(labels).to_csv('./data/raw_dir/us_lgc_knn_gsvd_8_20.csv')
# with open('./data/raw_dir/us_lgc_knn_gsvd_8_20.csv', 'w') as csvfile:
#     writer = csv.writer(csvfile)
#     for r in labels:
#         row = str(r.item()) 
#         writer.writerow([row])

# # %%
# knn_gsvd_8_20_labels = labels
# print(knn_gsvd_8_20_labels.shape)


# %%
louvain = LouvainNE(n_components=16)
knn = KNN(louvain, n_neighbors=101)
louvain_16_101_labels = knn.fit_transform(directed.adjacency, seeds)
# %%
# %%
print(louvain_16_101_labels.shape)
# %%
print(type(louvain_16_101_labels))
# %%
# pd.DataFrame(labels).to_csv('./data/raw_dir/us_lgc_knn_gsvd_8_20.csv')
with open('./data/raw_dir/us_lgc_knn_louvain_16_101_labels.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    for r in louvain_16_101_labels:
        row = str(r.item()) 
        writer.writerow([row])
# %%
