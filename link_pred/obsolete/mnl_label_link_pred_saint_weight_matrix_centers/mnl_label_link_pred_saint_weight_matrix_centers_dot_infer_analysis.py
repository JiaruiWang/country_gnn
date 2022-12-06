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

inputfile = './mnl_label_link_pred_saint_weight_matrix_centers_dot_0.9087_pred_label_startid_statelabel_endid_statelabel_neg1.0.csv'
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
48802 0.5776751893939394
35678 0.4223248106060606
negative sampling ratio=1  test=0.01
1689610
1689610
1689610
1689610
1689610
1689610
486490 0.5758607015820218
358315 0.4241392984179781

'''


# %%
# get roc and plot
auc_score = roc_auc_score(total_edge_label, total_pred)
print('roc_auc_score', auc_score)
'''
negative sampling ratio=2.0 
roc_auc_score 0.9082277757208539
negative sampling ratio=1
roc_auc_score 0.9096192094927209
negative sampling ratio=1 test=0.01
roc_auc_score 0.907771109871169
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
23687 0.8076822916666667 0.13241595643939394 0.7184221148490906
negative sampling ratio=1.0
15347 0.8005326704545455 0.12084517045454546 0.7338989973068237
negative sampling ratio=1 test=0.01
167958 0.8066760968507526 0.13176295121359366 0.7172802686691284

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
0.8476167929292929
negative sampling ratio=1.0
0.83984375
negative sampling ratio=1 test=0.01
0.8374565728185794
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
negative sampling ratio=2.0
68233
22373
146587
16247
negative sampling ratio=1.0
67629
10209
74271
16851 84480
negative sampling ratio=1 test=0.01
681484
111314
733491
163321
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
negative sampling ratio=2
false negative edges
9580    6667  16247
0.5896  0.4103
negative sampling ratio=1
false negative edges
10081   6770  16851
0.5982  0.4017

0.7934
0.8102

negative sampling ratio=1 test=0.01
96727   66594  163321
0.5922  0.4077
'''
# %%
