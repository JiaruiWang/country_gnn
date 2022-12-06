# %% Import packages
import pandas as pd
import os.path as osp
import numpy as np
import jenkspy
import csv
from scipy.special import softmax
#%%
pl_class_names = ["AL", "AK", "AZ", "AR", "CO", "CT",\
                  "DE", "GA", "HI", "ID", "IN",\
                  "IA", "KS", "KY", "LA", "ME", "MD", "MA",\
                  "MI", "MN", "MS", "MO", "MT", "NE", "NV",\
                  "NH", "NJ", "NM", "NC", "ND", "OH",\
                  "OK", "OR", "RI", "SC", "SD", "TN",\
                  "UT", "VT", "VA", "WA", "WV", "WI",\
                  "WY", "DC"]
pl_state_names = ["Alabama","Alaska","Arizona","Arkansas","Colorado","Connecticut",
                  "Delaware","Georgia","Hawaii","Idaho","Indiana","Iowa","Kansas",
                  "Kentucky","Louisiana","Maine","Maryland","Massachusetts","Michigan","Minnesota",
                  "Mississippi","Missouri","Montana","Nebraska","Nevada","New Hampshire","New Jersey",
                  "New Mexico","North Carolina","North Dakota","Ohio","Oklahoma","Oregon",
                  "Rhode Island","South Carolina","South Dakota","Tennessee","Utah",
                  "Vermont","Virginia","Washington","West Virginia","Wisconsin","Wyoming","Washington D.C.",]

# %%
# Read in probabi
# file = '../model/saint_population_label_all_label/saint_inference_output_id_y_pred_51probability.csv'
file = './saint_newcity_truelabel_A_to_A_no_6_centers_45_infer_AB_id_y_pred_45probability.csv'
data = pd.read_csv(file, sep=',', header=None)
#%%
print(data)
# %%
# split columns
id_list = data.iloc[:, 0:1].T.values.tolist()[0]
label_list = data.iloc[:, 1:2].T.values.tolist()[0]
pred_list = data.iloc[:, 2:3].T.values.tolist()[0]
scoresdf = data.iloc[:, 3:48]

print(id_list[0:10])
print(label_list[0:10])
print(pred_list[0:10])
print(scoresdf[0:10])
#%%
# Apply softmax to state scores.
np.set_printoptions(precision=5)
scores = scoresdf.values
print(type(scores), scores.shape)
softmax_probability = softmax(scores, axis=1)

#%%
# Read in state name to order list. New data DC is 50, Old data DC is 47
# statefile = '../../data/raw_dir/state_classes.csv'
# state_data = pd.read_csv(statefile, sep=',', header=None)
# # print(data.T.values[0].tolist(), type(data.T.values[0].tolist()))
# state_order = state_data.T.values[0].tolist()
# print(state_order)
# print(pl_state_names)
state_order = pl_state_names
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
    for i in range(45):
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
print(pred_list[5555666:5555669])

#%%
outputfile = './id_label_state_pred_state_threshold_counts_possible_states_no_6_centers.csv'
with open(outputfile, 'w') as file:
    csvwriter = csv.writer(file)
    for i in range(5873395):
        row = [id_list[i], label_list[i], state_order[label_list[i]], \
            pred_list[i], state_order[pred_list[i]], probability_threshold[i], possible_count[i]]
        for j in range(len(possible_states[i])):
            row.append(possible_states[i][j]),
            row.append(states_probability[i][j])
        csvwriter.writerow(row)

#%%
iter = 100
inputfile = './id_label_state_pred_state_threshold_counts_possible_states_no_6_centers.csv'
with open(inputfile, 'r') as file:
    csvreader = csv.reader(file)
    for row in csvreader:
        if iter == 0:
            break
        iter -= 1
        id = int(row[0])
        label = int(row[1])
        label_state = str(row[2])
        pred = int(row[3])
        pred_state = str(row[4])
        threshold = float(row[5])
        counts = int(row[6])
        states = []
        probs = []
        for i in range(counts):
            states.append(row[7+i*2])
            probs.append(float(row[7+i*2+1]))
        print(id, label, label_state, pred, pred_state, threshold, counts, states, probs)
# %%
