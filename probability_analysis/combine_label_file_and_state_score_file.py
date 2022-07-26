# %% Import packages
import pandas as pd
import os.path as osp
import numpy as np


# %%
file_path1 = '../data/raw_dir/us_pages_lgc_true_label_51_label.csv'
twolabel = pd.read_csv(file_path1, sep='\t', header=None)
file_path2 = '../data/raw_dir/us_pages_lgc_idx_id_mask_label_state.csv'
fivelabel = pd.read_csv(file_path2, sep='\t', header=None)
out_pa = '../model/saint_all_label/saint_id_y_pred_51probability_setseed_test1.csv'
out1 = pd.read_csv(out_pa, sep=',', header=None)
# %%
two = twolabel.values
five = fivelabel.values
out = out1.values
print(two)
print(five)
print(out)
print(type(two[0][0]),type(two[0][1]))
print(type(five[0][0]),type(five[0][1]),type(five[0][2]),
        type(five[0][3]),type(five[0][4]))

# %%
print(out1.iloc[0:10][0:10])
print(fivelabel.iloc[0:10])
#%%
for i in range(5873395):
    if out1.iat[i,0] != fivelabel.iat[i,1]:
        print(i)
#%%
total = pd.concat([fivelabel, out1], axis=1)
print(total[0:10])
#%%
for i in range(5873395):
    if total.iat[i,1] != total.iat[i,5]:
        print(i)

#%%
outputfile = "../model/saint_all_label/saint_idx_id_mask_label_state_id_y_pred_51probability_setseed.csv"
total.to_csv(outputfile, sep=',', header=False, index=False)
# %%
count_mask = 0
count_label = 0
for i in range(5873395):
    if (two[0][0] == -1 and five[0][2] == 0) or (two[0][0] != -1 and five[0][2] == 1):
        count_mask += 1
    if (two[0][1] == five[0][3]):
        count_label += 1
    if (two[0][1] != five[0][3]):
        print("line ", i, " lable not same.")
print(count_mask, count_label)
# %%
