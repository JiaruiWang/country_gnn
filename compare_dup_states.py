# %%
# import packages
from unittest import result
import mysql.connector
import pandas as pd
import csv

# %%
# Initialize mysql connector
mydb = mysql.connector.connect(
  host="localhost",
  user="jerry",
  password="password",
  database="pagenet"
)
mycursor = mydb.cursor()

# %%
# Store {state:label} in state_label
sql = "select * from state_label;"
mycursor.execute(sql)
result = mycursor.fetchall()
state_label = {}
label_state = {}
for t in result:
    state, label = t
    state_label[state] = label
    label_state[label] = state
print(state_label)
print(label_state)

# %%
# Store {city: {state: label index}} dict
sql = "select * from us_city_state;"
mycursor.execute(sql)
result = mycursor.fetchall()
city_label_state = {}
for t in result:
    city, state = t
    city = city.lower()
    if state not in state_label:
        continue
    if city not in city_label_state:
        city_label_state[city] = {state_label[state]:state}
    else:
        city_label_state[city][state_label[state]] = state
print(city_label_state['Franklin'.lower()])
print(len(city_label_state))

# %%
# read id, idx, label, city, dup_states from us_pages_lgc.csv
lgc_nodes = []
with open('./data/raw_dir/us_pages_lgc.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    count = 0
    for r in reader:    
        id, idx, label, city, dup_states = r
        id, idx, label, dup_states = int(id), int(idx), int(label), int(dup_states)
        row = {'id':id, 'idx':idx, 'label':label, 'city':city.lower(), 'dup_states':dup_states}
        lgc_nodes.append(row)
        count += 1
        # if count == 10: break
print(len(lgc_nodes))

# %%
# read predicted labels from gsvd embedding knn predictions.
# the classification took 380 min to run gsvd_8_20
# the classification took 961 min to run gsvd_16_61
with open('./data/raw_dir/us_lgc_knn_gsvd_16_61.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    count = 0
    for r in reader:    
        pred = int(r[0])
        lgc_nodes[count]['gsvd_label'] = pred
        count += 1
print(count)


# %%
# read predicted labels from spectral embedding knn predictions.
# the classification took 150 min to run
with open('./data/raw_dir/us_lgc_knn_spectral_16_61_.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    count = 0
    for r in reader:    
        pred = int(r[0])
        lgc_nodes[count]['spectral_label'] = pred
        count += 1
print(count)


# %%
# read predicted labels from propagation_50 predictions.
with open('./data/raw_dir/us_lgc_propagation_50_.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    count = 0
    for r in reader:    
        pred = int(r[0])
        lgc_nodes[count]['propagation_50'] = pred
        count += 1
print(count)

# %%
# read predicted labels from propagation_200 predictions.
with open('./data/raw_dir/us_lgc_propagation_200_.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    count = 0
    for r in reader:    
        pred = int(r[0])
        lgc_nodes[count]['propagation_200'] = pred
        count += 1
print(count)

# %%
# read predicted labels from propagation_400 predictions.
with open('./data/raw_dir/us_lgc_propagation_400_.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    count = 0
    for r in reader:    
        pred = int(r[0])
        lgc_nodes[count]['propagation_400'] = pred
        count += 1
print(count)

# %%
# read predicted labels from 2 hop 1 round.
with open('./data/raw_dir/us_lgc_2_hop_bfs_voting_label_correct_1st.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    count = 0
    for r in reader:    
        pred = int(r[0])
        lgc_nodes[count]['hop_2_bfs'] = pred
        count += 1
print(count)
# %%
# read predicted labels from 2 hop 2nd round with seed change.
with open('./data/raw_dir/us_lgc_2_hop_bfs_voting_label_2nd_with_seed_change.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    count = 0
    for r in reader:    
        pred = int(r[0])
        lgc_nodes[count]['hop_2_bfs_2nd_seed_change'] = pred
        count += 1
print(count)
# %%
print(len(lgc_nodes))
print(city_label_state['Orange City'.lower()])

# %%
# examine the prediction acc for gsvd, spectral
gsvd_pred_miss, gsvd_seed_miss = 0, 0
spectral_pred_miss, spectral_seed_miss = 0, 0
propagation_50_pred_miss, propagation_50_seed_miss = 0, 0
propagation_200_pred_miss, propagation_200_seed_miss = 0, 0
propagation_400_pred_miss, propagation_400_seed_miss = 0, 0
hop_2_bfs_pred_miss, hop_2_bfs_seed_miss = 0, 0
hop_2_bfs_2nd_seed_change_pred_miss, hop_2_bfs_2nd_seed_change_seed_miss = 0, 0
gsvd_pred_miss_array = [0 for i in range(30)]
gsvd_pred_count_array = [1 for i in range(30)]
spectral_pred_miss_array = [0 for i in range(30)]
spectral_pred_count_array = [1 for i in range(30)]
propagation_50_pred_miss_array = [0 for i in range(30)]
propagation_50_pred_count_array = [1 for i in range(30)]
propagation_200_pred_miss_array = [0 for i in range(30)]
propagation_200_pred_count_array = [1 for i in range(30)]
propagation_400_pred_miss_array = [0 for i in range(30)]
propagation_400_pred_count_array = [1 for i in range(30)]
hop_2_bfs_pred_miss_array = [0 for i in range(30)]
hop_2_bfs_pred_count_array = [1 for i in range(30)]
hop_2_bfs_2nd_seed_change_pred_miss_array = [0 for i in range(30)]
hop_2_bfs_2nd_seed_change_pred_count_array = [1 for i in range(30)]
total_seed, total_pred, total = 2331787, 3541608, 5873395
for row in lgc_nodes:
    if row['dup_states'] == 0: continue
    if row['label'] != -1:
        if row['label'] != row['gsvd_label']: gsvd_seed_miss += 1
        if row['label'] != row['spectral_label']: spectral_seed_miss += 1
        if row['label'] != row['propagation_50']: propagation_50_seed_miss += 1
        if row['label'] != row['propagation_200']: propagation_200_seed_miss += 1
        if row['label'] != row['hop_2_bfs']: hop_2_bfs_seed_miss += 1
        if row['label'] != row['hop_2_bfs_2nd_seed_change']: hop_2_bfs_2nd_seed_change_seed_miss += 1
    else:
        if row['gsvd_label'] not in city_label_state[row['city']]: 
            gsvd_pred_miss += 1
            gsvd_pred_miss_array[row['dup_states']] += 1
        if row['spectral_label'] not in city_label_state[row['city']]: 
            spectral_pred_miss += 1
            spectral_pred_miss_array[row['dup_states']] += 1
        if row['propagation_50'] not in city_label_state[row['city']]: 
            propagation_50_pred_miss += 1
            propagation_50_pred_miss_array[row['dup_states']] += 1
        if row['propagation_200'] not in city_label_state[row['city']]: 
            propagation_200_pred_miss += 1
            propagation_200_pred_miss_array[row['dup_states']] += 1
        if row['hop_2_bfs'] not in city_label_state[row['city']]: 
            hop_2_bfs_pred_miss += 1
            hop_2_bfs_pred_miss_array[row['dup_states']] += 1
        if row['hop_2_bfs'] not in city_label_state[row['city']]: 
            hop_2_bfs_2nd_seed_change_pred_miss += 1
            hop_2_bfs_2nd_seed_change_pred_miss_array[row['dup_states']] += 1
    gsvd_pred_count_array[row['dup_states']] += 1
    spectral_pred_count_array[row['dup_states']] += 1
    propagation_50_pred_count_array[row['dup_states']] += 1
    propagation_200_pred_count_array[row['dup_states']] += 1
    propagation_400_pred_count_array[row['dup_states']] += 1
    hop_2_bfs_2nd_seed_change_pred_count_array[row['dup_states']] += 1
    hop_2_bfs_2nd_seed_change_pred_count_array[row['dup_states']] += 1
    pass
print('gsvd     seed miss: ', gsvd_seed_miss, ' pred miss: ', gsvd_pred_miss, gsvd_pred_miss/total_pred)
print('spectral seed miss: ', spectral_seed_miss, ' pred miss: ', spectral_pred_miss, spectral_pred_miss/total_pred)
print('propagation_50 seed miss: ', propagation_50_seed_miss, ' pred miss: ', propagation_50_pred_miss, propagation_50_pred_miss/total_pred)
# print('propagation_200 seed miss: ', propagation_200_seed_miss, ' pred miss: ', propagation_200_pred_miss, propagation_200_pred_miss/total_pred)
# print('propagation_400 seed miss: ', propagation_400_seed_miss, ' pred miss: ', propagation_400_pred_miss, propagation_400_pred_miss/total_pred)
print('hop_2_bfs seed miss: ', hop_2_bfs_seed_miss, ' pred miss: ', hop_2_bfs_pred_miss, hop_2_bfs_pred_miss/total_pred)
print('hop_2_bfs_2nd_seed_change seed miss: ', hop_2_bfs_2nd_seed_change_seed_miss, ' pred miss: ', hop_2_bfs_2nd_seed_change_pred_miss, hop_2_bfs_2nd_seed_change_pred_miss/total_pred)
# %%
for i in range(30):
    print(i, 'gsvd', "{:.2f}".format(gsvd_pred_miss_array[i]/gsvd_pred_count_array[i]),
             'spectral', "{:.2f}".format(spectral_pred_miss_array[i]/spectral_pred_count_array[i]),
             'propagation_50', "{:.2f}".format(propagation_50_pred_miss_array[i]/propagation_50_pred_count_array[i]),
            #  'propagation_200', "{:.2f}".format(propagation_200_pred_miss_array[i]/propagation_200_pred_count_array[i]),
            #  'propagation_400', "{:.2f}".format(propagation_400_pred_miss_array[i]/propagation_400_pred_count_array[i]),
             'hop_2_bfs', "{:.2f}".format(hop_2_bfs_pred_miss_array[i]/hop_2_bfs_pred_count_array[i]),
             'hop_2_bfs_2nd_seed_change', "{:.2f}".format(hop_2_bfs_2nd_seed_change_pred_miss_array[i]/hop_2_bfs_2nd_seed_change_pred_count_array[i]))
# %%
# %%
