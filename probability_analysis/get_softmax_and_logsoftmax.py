# %% Import packages
import pandas as pd
import os.path as osp
import numpy as np
from scipy.special import softmax

# %%
file = '../model/saint_all_label/saint_idx_id_mask_label_state_id_y_pred_51probability.csv'
data = pd.read_csv(file, sep=',', header=None)
print(data)
# %%
idx = data.iloc[:, 0:1]
id = data.iloc[:, 1:2]
mask = data.iloc[:, 2:3]
label = data.iloc[:, 3:4]
state = data.iloc[:, 4:5]
pred = data.iloc[:, 7:8]
probability = data.iloc[:, 8:59]
# %%
print(idx)
print(id)
print(mask)
print(label)
print(state)
print(pred)
print(probability)
#%%
np.set_printoptions(precision=5)
n = probability.values
print(type(n), n.shape)
print(n.shape)
m = softmax(n, axis=1)

print(m.shape)
#%%
print(m[0:20])
# %%
