#%%
import os

#%%
path = './dot_pts/test_neg_5/'
dir_list = os.listdir(path)
#%%
dir_list.sort()
dir_list = dir_list[2:]
print(len(dir_list))
# print(dir_list)
# %%
tuple_list = []
for i in range(len(dir_list)):
    if dir_list[i][9] == '_':
        epoch = dir_list[i][0:9]
        auc = dir_list[i][-17:-3]
        loss = dir_list[i][10:-18]
    else:
        epoch = dir_list[i][0:10]
        auc = dir_list[i][-17:-3]
        loss = dir_list[i][11:-18]

    tuple_list.append((epoch, loss, auc))
print(tuple_list)
# %%
ordered = sorted(tuple_list, key=lambda x:x[1])
for t in ordered:
    print(t)
# %%
