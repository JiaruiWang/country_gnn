# %%
from decimal import DivisionByZero
# import matplotlib as mp
import pandas as pd
import numpy as np
# %%
raw_features = pd.read_csv('/home/jerry/Documents/country_gnn/comparison_jason/raw_features.csv', index_col=0)
raw_id = pd.read_csv('/home/jerry/Documents/country_gnn/comparison_jason/raw_id.csv', index_col=0)
features = pd.read_csv('/home/jerry/Documents/country_gnn/comparison_jason/features.csv', index_col=0)
features_id = pd.read_csv('/home/jerry/Documents/country_gnn/comparison_jason/features_id.csv', index_col=0)
y = pd.read_csv('/home/jerry/Documents/country_gnn/comparison_jason/y.csv', index_col=0)

print(raw_features.shape, raw_id.shape, features.shape, features_id.shape, y.shape)
# %%
print(y[0:10])
# %%
count, division = np.histogram(y, bins=[i+0.5 for i in range(-1, 50)])
print(count, division)
# %%
id_to_index = {}
for i in range(features_id.shape[0]):
    id = int(features_id.iloc[i].values.tolist()[0])
    id_to_index[id] = i 
# %%
print(int(features_id.iloc[0].values.tolist()[0]))
print(type(features_id.iloc[0].values))
print(type(features_id.iloc[0].values.tolist()[0]))
print(type(str(features_id.iloc[0].values.tolist()[0])))
print(str(features_id.iloc[0].values.tolist()[0]))
print(id_to_index[10150152322365192])
print(features.iloc[4044].values.tolist())
# %%
row_data_id = pd.read_csv('/home/jerry/Documents/public_page_graph/Adv_BFS_with_dup_states/BFS_output/id_Alabama.csv', 
                            header = None, error_bad_lines=False, skip_blank_lines=True,
                                #    dtype=str,
                                   low_memory=False,
                                   na_values=0, 
                                   keep_default_na=False)
print(row_data_id.iloc[0].values.tolist()[0])
print(type(row_data_id.iloc[0].values))
print(type(row_data_id.iloc[0].values.tolist()[0]))
print(type(str(row_data_id.iloc[0].values.tolist()[0])))
# %%
i = 0
print(features_id[i], y[i], features[i])