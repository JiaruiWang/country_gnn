# %% Import packages
import pandas as pd
import os.path as osp
import numpy as np
import jenkspy
import csv
from scipy.special import softmax

# %%
# Read in probabi
file = '../model/saint_all_label/saint_idx_id_mask_label_state_id_y_pred_51probability_setseed.csv'
data = pd.read_csv(file, sep=',', header=None)
#%%
print(data)
# %%
# split columns
idx_list = data.iloc[:, 0:1].T.values.tolist()[0]
id_list = data.iloc[:, 1:2].T.values.tolist()[0]
mask_list = data.iloc[:, 2:3].T.values.tolist()[0]
label_list = data.iloc[:, 3:4].T.values.tolist()[0]
state_list = data.iloc[:, 4:5].T.values.tolist()[0]
pred_list = data.iloc[:, 7:8].T.values.tolist()[0]
scoresdf = data.iloc[:, 8:59]

# print(idx)
# print(id)
# print(mask)
# print(label)
# print(state)
# print(pred)
# print(scores)
#%%
# Apply softmax to state scores.
np.set_printoptions(precision=5)
scores = scoresdf.values
print(type(scores), scores.shape)
softmax_probability = softmax(scores, axis=1)

#%%
# Read in state name to order list
statefile = '../data/raw_dir/state_classes.csv'
state_data = pd.read_csv(statefile, sep=',', header=None)
# print(data.T.values[0].tolist(), type(data.T.values[0].tolist()))
state_order = state_data.T.values[0].tolist()
print(state_order)

#%%
# Define 2 breaks jenks natural breaks
jnk = jenkspy.JenksNaturalBreaks(nb_class=2)

#%%
# Apply jenks to softmax probability and store data
possible_count = []
probability_threshold = []
possible_states = []
states_probability = []
for i in range(5873395):
# for i in range(10):
    jnk.fit(softmax_probability[i:i+1][0])
    # print(softmax_probability[i:i+1][0])
    label_one_row = jnk.labels_.tolist()
    states = []
    for i in range(51):
        if label_one_row[i] == 1:
            states.append(state_order[i])
    possible_states.append(states)
    probs = jnk.groups_[1].tolist()
    states_probability.append(probs)
    possible_count.append(len(probs))
    # print(type(jnk.labels_.tolist()), type(jnk.groups_[1].tolist()), type(jnk.inner_breaks_))
    probability_threshold.append(jnk.inner_breaks_[0])
#%%
print(softmax_probability[5555666:5555669])
print(possible_count[5555666:5555669])
print(probability_threshold[5555666:5555669])
print(possible_states[5555666:5555669])
print(states_probability[5555666:5555669])
print(label_list[5555666:5555669])
print(state_list[5555666:5555669])
print(pred_list[5555666:5555669])

#%%
outputfile = '../model/saint_all_label/idx_id_mask_label_state_pred_state_threshold_counts_possible_states.csv'
with open(outputfile, 'w') as file:
    csvwriter = csv.writer(file)
    for i in range(5873395):
        row = [idx_list[i], id_list[i], mask_list[i], label_list[i], state_list[i], \
            pred_list[i], state_order[pred_list[i]], probability_threshold[i], possible_count[i]]
        for j in range(len(possible_states[i])):
            row.append(possible_states[i][j]),
            row.append(states_probability[i][j])
        csvwriter.writerow(row)

#%%
inputfile = '../model/saint_all_label/idx_id_mask_label_state_pred_state_threshold_counts_possible_states.csv'
with open(inputfile, 'r') as file:
    csvreader = csv.reader(file)
    for row in csvreader:
        idx = int(row[0])
        id = int(row[1])
        mask = int(row[2])
        label = int(row[3])
        label_state = str(row[4])
        pred = int(row[5])
        pred_state = str(row[6])
        threshold = float(row[7])
        counts = int(row[8])
        states = []
        probs = []
        for i in range(counts):
            states.append(row[9+i*2])
            probs.append(float(row[9+i*2+1]))
        print(idx, id, mask, label, label_state, pred, pred_state, threshold, counts, states, probs)
# %%
