# %%

import mysql.connector
import pandas as pd
import tqdm

# %%
# Initialize mysql connector
mydb = mysql.connector.connect(
  host="localhost",
  user="jerry",
  password="0000",
  database="pagenet"
)
mycursor = mydb.cursor()

# %%
# Add label indice to table state_label.

# query = "select state from state_label;"
# mycursor.execute(query)
# result = mycursor.fetchall()
# count = 0
# for t in result:
#     # print(t) ('Alabama',)
#     sql = "update state_label set label={} where state='{}';".format(count,t[0])
#     mycursor.execute(sql)
#     count += 1
# mydb.commit()
# print(count, "record updated.")

# %%
# Store {state:label} in state_dict

# sql = "select * from state_label;"
# mycursor.execute(sql)
# result = mycursor.fetchall()
# state_dict = {}
# for t in result:
#     state, label = t
#     state_dict[state] = label
# print(state_dict)

# %%
# Add state label indice to table us_pages

# sql = "select idx, state from us_pages where state is not Null;"
# mycursor.execute(sql)
# result = mycursor.fetchall()
# count = 0
# for t in result:
#     idx, state = t
#     label = state_dict[state]
#     sql = "update us_pages set label={} where idx={};".format(label, idx)
#     mycursor.execute(sql)
#     count += 1
#     if count % 100000 == 0:
#         mydb.commit()
#         print(count, "record updated.")
# mydb.commit()
# print(count, "record updated.")

# %%
# add bfs 2nd round label to table us_lgc_pages

df = pd.read_csv('./data/raw_dir/us_lgc_2_hop_bfs_voting_label_correct_2nd.csv', sep='\t', header=None)
# print(df.shape)
# v = int(df.iloc[0].values)
# print(type(v), v)

count = 0
for i in range(df.shape[0]):
    v = int(df.iloc[i].values)
    sql = 'update us_lgc_pages set bfs_label={} where new_idx={};'.format(v, i)
    mycursor.execute(sql)
    count += 1
    if count % 100000 == 0:
        mydb.commit()
        print(count, "record updated.")
mydb.commit()
print(count, "record updated.")

# %%
# add id, world_idx, new_idx to table us_lgc_edges

world_idx_df = pd.read_csv('./data/raw_dir/us_edges_lgc.csv', sep='\t', header=None)
new_idx_df = pd.read_csv('./data/raw_dir/us_edges_lgc_relabeled.csv', sep='\t', header=None)
print(world_idx_df.shape)
s, e = int(world_idx_df.iloc[0][0]), int(world_idx_df.iloc[0][1])
print(s,e)
# %%
sql = 'select id from us_lgc_pages where world_idx={}'.format(0)
mycursor.execute(sql)
result = mycursor.fetchall()
id_s = int(result[0][0])
print(id_s)
# %%
count = 0
for i in tqdm.tqdm(range(world_idx_df.shape[0])):
    world_s, world_t = int(world_idx_df.iloc[i][0]), int(world_idx_df.iloc[i][1])
    new_s, new_t = int(new_idx_df.iloc[i][0]), int(new_idx_df.iloc[i][1])

    sql = 'select id from us_lgc_pages where world_idx={}'.format(world_s)
    mycursor.execute(sql)
    result = mycursor.fetchall()
    id_s = int(result[0][0])

    sql = 'select id from us_lgc_pages where world_idx={}'.format(world_t)
    mycursor.execute(sql)
    result = mycursor.fetchall()
    id_t = int(result[0][0])

    sql = 'INSERT INTO us_lgc_edges VALUES ({}, {}, {}, {}, {}, {});'.format(
             id_s, id_t, world_s, world_t, new_s, new_t)
    mycursor.execute(sql)
    count += 1
    if count % 1000000 == 0:
        mydb.commit()
        print(count, "record updated.")
mydb.commit()
print(count, "record updated.")
# %%
