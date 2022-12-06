#%%
import csv
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
#%%
# read in predictions
total_pred = []
total_edge_label = []
total_edge_start_id = []
total_edge_end_id = []
total_edge_start_id_state_label = []
total_edge_end_id_state_label = []

inputfile = './mnl_label_link_pred_saint_label_dot_0.9083_pred_label_startid_statelabel_endid_statelabel_neg1.0.csv'
with open(inputfile, 'r') as file:
    csvreader = csv.reader(file, delimiter='\t')
    for row in csvreader:
        # print(row)
        total_pred.append(float(row[0]))
        total_edge_label.append(int(float(row[1])))
        total_edge_start_id.append(int(row[2]))
        total_edge_start_id_state_label.append(int(row[3]))
        total_edge_end_id.append(int(row[4]))
        total_edge_end_id_state_label.append(int(row[5]))
        
#%%
print(len(total_pred))
print(len(total_edge_label))
print(len(total_edge_start_id))
print(len(total_edge_end_id))
print(len(total_edge_start_id_state_label))
print(len(total_edge_end_id_state_label))
'''
negative sampling ratio=2.0
253440
253440
253440
253440
253440
253440
negative sampling ratio=1
168960
168960
168960
168960
168960
168960
negative sampling ratio=1 test=0.01
1689610
1689610
1689610
1689610
1689610
1689610
'''
# %%
# get roc and plot
auc_score = roc_auc_score(total_edge_label, total_pred)
print('roc_auc_score', auc_score)
'''
negative sampling ratio=2.0
0.9094696644974806
negative sampling ratio=1
0.9090527696845946
negative sampling ratio=1 test=0.01
roc_auc_score 0.9094279805216651
'''
fpr, tpr, thresholds = roc_curve(total_edge_label, total_pred)

plt.figure()
lw = 2
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.2f)" % auc_score,
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()
#%%
# find threshold index
threshold = 0
diff = 0
index = 0
for i in range(len(fpr)):
    if tpr[i] - fpr[i] > diff:
        diff = tpr[i] - fpr[i]
        index = i
print(index, tpr[index], fpr[index], thresholds[index])
'''
negative sampling ratio=2.0 
19733 0.8044862689393939 0.12356178977272728 0.8349145650863647
negative sampling ratio=1.0
14259 0.8085819128787879 0.12927320075757576 0.8260908126831055
negative sampling ratio=1 test=0.01
141452 0.812157835240085 0.1332638892998976 0.8167164325714111
'''
# %%
# apply threshold
plabel = []
for i in range(len(total_pred)):
    if total_pred[i] >= thresholds[index]:
        plabel.append(1)
    else:
        plabel.append(0)
    # if total_edge_start_id[i] == 180673242056613 and total_edge_end_id[i] == 153854904646050:
    #     print(total_pred[i], plabel[i], total_edge_label[i])

print(accuracy_score(total_edge_label, plabel))
'''
negative sampling ratio=2.0
0.8524542297979798
negative sampling ratio=1.0
0.839654356060606
negative sampling ratio=1 test=0.01
0.8394469729700937
'''
# %%
# seperate labels
'''
                         actual positive(1)   actual negative(0)
prediction positive(1)   True Positive(TP)    False Positive(0)
prediction negative(0)   False Negative(FN)   True Negative(0)
'''
true_posi_index = []
false_posi_index = []
true_nega_index = []
false_nega_index = []
for i in range(len(plabel)):
    if total_edge_label[i] == 1 and plabel[i] == 1:
        true_posi_index.append(i)
    elif total_edge_label[i] == 1 and plabel[i] == 0:
        false_nega_index.append(i)
    elif total_edge_label[i] == 0 and plabel[i] == 1:
        false_posi_index.append(i)
    elif total_edge_label[i] == 0 and plabel[i] == 0:
        true_nega_index.append(i)
    else:
        print("exception: ", i, plabel[i], total_edge_label[i])
#%%
print(len(true_posi_index))
print(len(false_posi_index))
print(len(true_nega_index))
print(len(false_nega_index))

'''
# negative sampling ratio=2
67963
20877
148083
16517 84480
# negative sampling ratio=1
68309
10921
73559
16171
negative sampling ratio=1 test=0.01
686115
112582
732223
158690
'''
#%%
# get inner state edge and inter state edge counts
inner_state_edge_total = 0
inter_state_edge_total = 0
for i in range(len(total_edge_label)):
    if total_edge_label[i] == 0:
        continue
    # print(total_edge_start_id_state_label[i], total_edge_end_id_state_label[i])
    if total_edge_start_id_state_label[i] == total_edge_end_id_state_label[i]:
        inner_state_edge_total += 1
    else:
        inter_state_edge_total += 1
total = inner_state_edge_total + inter_state_edge_total
print(inner_state_edge_total, inner_state_edge_total/total)
print(inter_state_edge_total, inter_state_edge_total/total)
# %%
true_posi_dict = {}
for i in true_posi_index:
    if (total_edge_start_id_state_label[i], total_edge_end_id_state_label[i]) not in true_posi_dict:
        true_posi_dict[(total_edge_start_id_state_label[i], total_edge_end_id_state_label[i])] = [
            (total_edge_start_id[i], total_edge_end_id[i])
        ]
    else:
        true_posi_dict[(total_edge_start_id_state_label[i], total_edge_end_id_state_label[i])].append(
            (total_edge_start_id[i], total_edge_end_id[i])
        )
true_posi_key_edgecount_list = []
true_posi_edgecount_key_list = []
for key, edges in true_posi_dict.items():
    # print(key, len(edges))
    true_posi_key_edgecount_list.append((key, len(edges)))
    true_posi_edgecount_key_list.append((len(edges), key))
true_posi_key_edgecount_list.sort()
# print(key_edgecount_list)
true_posi_edgecount_key_list.sort(reverse=True)
# print(len(edgecount_key_list), edgecount_key_list)
true_posi_inner_state_edge_count = 0
true_posi_inter_state_edge_count = 0
for i in range(len(true_posi_edgecount_key_list)):
    if true_posi_edgecount_key_list[i][1][0] == true_posi_edgecount_key_list[i][1][1]:
        true_posi_inner_state_edge_count += true_posi_edgecount_key_list[i][0]
    else:
        true_posi_inter_state_edge_count += true_posi_edgecount_key_list[i][0]
total = true_posi_inner_state_edge_count + true_posi_inter_state_edge_count
print(true_posi_inner_state_edge_count, true_posi_inner_state_edge_count/total)
print(true_posi_inter_state_edge_count, true_posi_inter_state_edge_count/total)
# %%
false_nega_dict = {}
for i in false_nega_index:
    if (total_edge_start_id_state_label[i], total_edge_end_id_state_label[i]) not in false_nega_dict:
        false_nega_dict[(total_edge_start_id_state_label[i], total_edge_end_id_state_label[i])] = [
            (total_edge_start_id[i], total_edge_end_id[i])
        ]
    else:
        false_nega_dict[(total_edge_start_id_state_label[i], total_edge_end_id_state_label[i])].append(
            (total_edge_start_id[i], total_edge_end_id[i])
        )
false_nega_key_edgecount_list = []
false_nega_edgecount_key_list = []
for key, edges in false_nega_dict.items():
    # print(key, len(edges))
    false_nega_key_edgecount_list.append((key, len(edges)))
    false_nega_edgecount_key_list.append((len(edges), key))
false_nega_key_edgecount_list.sort()
# print(key_edgecount_list)
false_nega_edgecount_key_list.sort(reverse=True)
# print(len(edgecount_key_list), edgecount_key_list)
false_nega_inner_state_edge_count = 0
false_nega_inter_state_edge_count = 0
for i in range(len(false_nega_edgecount_key_list)):
    if false_nega_edgecount_key_list[i][1][0] == false_nega_edgecount_key_list[i][1][1]:
        false_nega_inner_state_edge_count += false_nega_edgecount_key_list[i][0]
    else:
        false_nega_inter_state_edge_count += false_nega_edgecount_key_list[i][0]
total = false_nega_inner_state_edge_count + false_nega_inter_state_edge_count
print(false_nega_inner_state_edge_count, false_nega_inner_state_edge_count/total)
print(false_nega_inter_state_edge_count, false_nega_inter_state_edge_count/total)
'''
mis labeled true edges
negative sampling ratio=2
false negative edges 
9640    6877  16517
0.5836  0.4164
negative sampling ratio=1
false negative edges
9599    6572  16171
0.5935  0.4064

0.8030
0.8169
negative sampling ratio=1 test=0.01
93596   65094  158690
0.5898  0.4101
'''
# %%
