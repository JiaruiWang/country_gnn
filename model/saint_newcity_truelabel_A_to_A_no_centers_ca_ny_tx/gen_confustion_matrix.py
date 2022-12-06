# %% Import packages
import pandas as pd
import os.path as osp
import numpy as np

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib as mtl
mtl.style.use('ggplot')

# %%
pl_class_names = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT",\
                  "DE", "FL", "GA", "HI", "ID", "IL", "IN",\
                  "IA", "KS", "KY", "LA", "ME", "MD", "MA",\
                  "MI", "MN", "MS", "MO", "MT", "NE", "NV",\
                  "NH", "NJ", "NM", "NY", "NC", "ND", "OH",\
                  "OK", "OR", "PA", "RI", "SC", "SD", "TN",\
                  "TX", "UT", "VT", "VA", "WA", "WV", "WI",\
                  "WY", "DC"]

mnl_class_names = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT",\
                   "DE", "FL", "GA", "HI", "ID", "IL", "IN",\
                   "IA", "KS", "KY", "LA", "ME", "MD", "MA",\
                   "MI", "MN", "MS", "MO", "MT", "NE", "NV",\
                   "NH", "NJ", "NM", "NY", "NC", "ND", "OH",\
                   "OK", "OR", "PA", "RI", "SC", "SD", "TN",\
                   "TX", "UT", "VT", "VA", "WA", "DC", "WV",\
                   "WI", "WY"]

# %%
# file_path1 = '../new_city_data/us_pages_lgc_true_label_51_label.csv'
# twolabel = pd.read_csv(file_path1, sep='\t', header=None)
# file_path2 = '../new_city_data/us_pages_lgc_idx_id_mask_label_state.csv'
# fivelabel = pd.read_csv(file_path2, sep='\t', header=None)
# mnl_path = '../model/saint_all_label/saint_id_y_pred_51probability_setseed_test1.csv'
# mnl_out = pd.read_csv(mnl_path, sep=',', header=None)
pl_path = './saint_newcity_truelabel_A_to_A_no_centers_48_infer_A_id_y_pred_51probability.csv'
pl_out = pd.read_csv(pl_path, sep=',', header=None)
print(pl_out.shape)
# (2147399, 54)
# %%
# mnl_y = mnl_out.values[:,1:2].flatten()
# mnl_pred = mnl_out.values[:,2:3].flatten()

pl_y = pl_out.values[:,1:2].flatten()
pl_pred = pl_out.values[:,2:3].flatten()
# print(mnl_y)
# print(mnl_pred)
print(pl_y)
print(pl_pred)

#%%
# Compute confusion matrix
cnf_matrix = confusion_matrix(pl_y, pl_pred, normalize='true')
cnf_matrix = cnf_matrix * 100
np.savetxt('truelabel_no_centers_cm_2d.csv', cnf_matrix, fmt='%2d', delimiter=',')
# cnf_matrix = confusion_matrix(mnl_y, mnl_pred, normalize='true')
# cnf_matrix = cnf_matrix * 100
# np.savetxt('mnl_cm_1f.csv', cnf_matrix, fmt='%0.1f', delimiter=',')

# %%