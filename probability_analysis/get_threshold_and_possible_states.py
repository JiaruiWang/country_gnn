# %% Import packages
import pandas as pd
import os.path as osp
import numpy as np
import jenkspy
import csv
from scipy.special import softmax

# %%
# Read in probabi
file = '/home/jerry/Documents/country_gnn/model/saint_all_label/saint_idx_id_mask_label_state_id_y_pred_51probability_setseed.csv'
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
# Read in state name to order list. New data DC is 50, Old data DC is 47
statefile = '/home/jerry/Documents/country_gnn/data/raw_dir/state_classes.csv'
state_data = pd.read_csv(statefile, sep=',', header=None)
# print(data.T.values[0].tolist(), type(data.T.values[0].tolist()))
state_order = state_data.T.values[0].tolist()
print(state_order)

#%%
# Define 2 breaks jenks natural breaks
jnk = jenkspy.JenksNaturalBreaks(n_classes=2)

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
# build target source map
targets_infile = '/var/lib/mysql-files/pagenet_newcity_us_lgc_edges_reverse_id-target_id-source.csv'
targets = {}
sources_infile = '/var/lib/mysql-files/pagenet_newcity_us_lgc_edges_id-source_id-target.csv'
sources = {}
iter = 2
with open(targets_infile, 'r') as file:
    csvreader = csv.reader(file, delimiter='\t')
    for row in csvreader:
        # if iter == 0:
        #     break
        # iter -= 1
        target = int(row[0])
        source = int(row[1])
        if target not in targets:
            targets[target] = [source]
        else:
            targets[target].append(source)

with open(sources_infile, 'r') as file:
    csvreader = csv.reader(file, delimiter='\t')
    for row in csvreader:
        # if iter == 0:
        #     break
        # iter -= 1
        source = int(row[0])
        target = int(row[1])
        if source not in sources:
            sources[source] = [target]
        else:
            sources[source].append(target)

print(len(targets))
print(len(sources))
#%%
# sort the key list
targets_list = list(targets.keys())
targets_list.sort()
sources_list = list(sources.keys())
sources_list.sort()
#%%
count = 0
for i in range(len(targets_list)):
    count += len(targets[targets_list[i]])
print(count)
count = 0
for i in range(len(sources_list)):
    count += len(sources[sources_list[i]])
print(count)        

#%%
# write sorted file

# outfile = '/var/lib/mysql-files/pagenet_newcity_us_lgc_edges_reverse_id-target_id-source_sorted.csv'
# with open(outfile, 'w') as file:
#     csvwriter = csv.writer(file, delimiter='\t')
#     for i in range(len(key_list)):
#         for j in range(len(targets[key_list[i]])):
#             row = [key_list[i], targets[key_list[i]][j]]
#             csvwriter.writerow(row)
#%%
print(len(targets[5281959998]))
print(len(id_list))

#%%
# read page full info
import mysql.connector 
mydb = mysql.connector.connect(
  host="localhost",
  user="jerry",
  password="0000",
  database="pagenet_newcity"
)
mycursor = mydb.cursor()

outputfile = '/home/jerry/Documents/country_gnn/model/saint_all_label/us_lgc_page_id_name_category_city_likespage_fan_outward_inward.csv'
with open(outputfile, 'w') as file:
    csvwriter = csv.writer(file, delimiter='\t')
    for i in range(len(id_list)):
        sql = "select id, name, category, city, likespage, fan\
            from world_page_full where id={};".format(id_list[i])
        mycursor.execute(sql)
        result = mycursor.fetchall()
        if (len(result) > 1):
            print(id_list[i], result)
        row = list(result[0])
        if id_list[i] in sources:
            row.append(len(sources[id_list[i]]))
        else:
            row.append(0)
        if id_list[i] in targets:
            row.append(len(targets[id_list[i]]))
        else:
            row.append(0)
        csvwriter.writerow(row)

#%% 
# read 
inputfile = '/home/jerry/Documents/country_gnn/model/saint_all_label/us_lgc_page_id_name_category_city_likespage_fan_outward_inward.csv'
with open(inputfile, 'r') as file:
    csvreader = csv.reader(file, delimiter='\t')
    counter = 0
    for row in csvreader:
        id = int(row[0])
        name = str(row[1])
        category = str(row[2])
        city = str(row[3])
        likespage = int(row[4])
        fan = int(row[5])
        outward = int(row[6])
        inward = int(row[7])
        if id != id_list[counter]:
            print(counter, id, id_list[counter])
        counter += 1
print(counter)

#%%
outputfile = '/home/jerry/Documents/country_gnn/model/saint_all_label/idx_id_mask_label_state_pred_state_threshold_counts_possible_states.csv'
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
iter = 100
inputfile = '/home/jerry/Documents/country_gnn/model/saint_all_label/idx_id_mask_label_state_pred_state_threshold_counts_possible_states.csv'
with open(inputfile, 'r') as file:
    csvreader = csv.reader(file)
    for row in csvreader:
        if iter == 0:
            break
        iter -= 1
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
