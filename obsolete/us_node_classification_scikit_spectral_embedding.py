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
spectral = Spectral(n_components=32)

spectral_32_embedding = spectral.fit_transform(directed.adjacency)
# %%
# %%
print(spectral_32_embedding.shape)
# %%
print(type(spectral_32_embedding))
# %%
# pd.DataFrame(labels).to_csv('./data/raw_dir/us_lgc_knn_gsvd_8_20.csv')
np.savetxt(fname='./data/raw_dir/us_lgc_spectral_32_embedding.csv', 
           X=spectral_32_embedding,
           delimiter='\t') 

# %%
