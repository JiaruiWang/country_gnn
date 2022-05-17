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
    reader = csv.reader(csvfile, delimiter='\t')    
    count  = -1
    for r in reader:    
        id, idx, label, city, dup_states = r
        label = int(label)
        count+=1
        if label == -1: continue
        seeds[count] = label
        
        # if count == 10: break

print(count)
print(len(seeds))
print(list(seeds.items())[0:10],list(seeds.items())[2331777:2331787])
# %%

# %%
louvain = LouvainNE(n_components=32)
knn = KNN(louvain, n_neighbors=501)
knn_louvain_32_501_labels = knn.fit_transform(directed.adjacency, seeds)
# %%
# %%
print(knn_louvain_32_501_labels.shape)
# %%
print(type(knn_louvain_32_501_labels))
# %%
# pd.DataFrame(labels).to_csv('./data/raw_dir/us_lgc_knn_gsvd_8_20.csv')
with open('./data/raw_dir/us_lgc_knn_louvain_32_501_labels.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    for r in knn_louvain_32_501_labels:
        row = str(r.item()) 
        writer.writerow([row])
# %%
