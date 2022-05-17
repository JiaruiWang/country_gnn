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
        id, idx, label, city, dup_stats = r
        label = int(label)
        count+=1
        if label == -1: continue
        seeds[count] = label
        
        # if count == 10: break

print(count)
print(len(seeds))
print(list(seeds.items())[0:10],list(seeds.items())[2331777:2331787])
# %%
knn = KNN(Spectral(n_components=32), n_neighbors=1001)
knn_spectral_32_1001_labels = knn.fit_transform(directed.adjacency, seeds)
# %%
print(knn_spectral_32_1001_labels.shape)
# %%
print(type(knn_spectral_32_1001_labels))
# %%
# pd.DataFrame(labels).to_csv('./data/raw_dir/us_lgc_knn_gsvd_8_20.csv')
with open('./data/raw_dir/us_lgc_knn_spectral_32_1001_.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    for r in knn_spectral_32_1001_labels:
        row = str(r.item()) 
        writer.writerow([row])

# %%
