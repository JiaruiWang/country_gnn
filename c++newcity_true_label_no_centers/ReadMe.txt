1. Generate feature vectors from anchored seeds

~/code/c++/crawler_v2/latestCode/state_labeler/BFS_from_anchored_pages/run.sh > 20171219.log
~/code/c++/crawler_v2/latestCode/state_labeler/BFS_from_anchored_pages/run_2lists.sh > 20171222_2lists.log
memory: 8% ~= 120*.08 ~= 10G memory
time: 20:40 starts 23:46 ends ~= 3 hours

Output directory is /public_page_graph/Adv_BFS/BFS_output/
Total page count: 15,115,446
Other page count: 12,685,090 (pages with country labels other than U.S.)
U.S. pages count:  2,430,356 (contains U.S. pages whose city belongs to only one state in the U.S.,
                              U.S. pages whose city belongs to many states in the U.S. are thrown away.)


2. Clean all-zero feature vectors which are not reached by any seeds(anchors)

rm -f /data1/yclin123/public_page_graph/BFS_output/Guam.csv
rm -f /data1/yclin123/public_page_graph/BFS_output/Puerto Rico.csv

~/code/python2/state_labeler/python clean_data.py > 20171219_cl.log

Output directory is /public_page_graph/Adv_BFS/BFS_output_cleaned/
Total page count: 6,851,911
Other page count: 5,842,776 (pages with country labels other than U.S.)
U.S. pages count: 1,009,135 (contains U.S. pages whose city belongs to only one state in the U.S.,
                              U.S. pages whose city belongs to many states in the U.S. are thrown away.)

3. Select min2 feature vectors from each state and others
~/code/python2/state_labeler/python select_fv_min2.py > 20171221_sl_min2.log

    a. Calulate the minimum non-zero value (smallest) of the 102 dimension distance vector to 51 anchors for each page.
    b. For other pages, only keep smallest >= 5 the arbitrary threshold for other pages, throw away the rest pages.
    c. For U.S. pages, only keep smallest <= 3 the arbitrary threshold for U.S. pages, throw away the rest pages.

Output directory is /public_page_graph/Adv_BFS/BFS_output__min2_selected/
       Jason didn't modify page_id files in this directory.

Total page count:   641,407
Other page count:   100,000 (only keep 100,000 pages out of 3,107,168 pages, throw the rest. Pages with country labels other than U.S.)
U.S. pages count:   541,407 (contains U.S. pages whose city belongs to only one state in the U.S.,
                              U.S. pages whose city belongs to many states in the U.S. are thrown away.)


4. select N feature vectors from each state and others (N <= common minimum number, Nevada:824, N=800)
~/code/python2/state_labeler/python select_fv.py > 20171221_sl.log

Nevada has the minimum 824 pages. Jason set N = 800, means select_fv.py will just select first 800 pages from each class.

Output directory is /public_page_graph/Adv_BFS/BFS_output_min2_selected_with_id_first800/
Jerry: Copy the directory to /home/jerry/Documents/jason_code/python code/state_labeler/adv/features_adv_min2_selected800/

Total page count: 40,800
Other page count:    800 (pages with country labels other than U.S.)
U.S. pages count: 40,000 (50 states * 800 = 40,000
                            Contains U.S. pages whose city belongs to only one state in the U.S.,
                            U.S. pages whose city belongs to many states in the U.S. are thrown away.)


5. upload feature vector files into docker ./labeller/features/
6. replace others.csv by the first N feature vectors instead

7. run ML classifier
./labeller/feature_ml-master.ipynb

Jason splits the 40,800 pages, into 80% training set and 20% testing set.

The classifiers are scikit learn built in functions:
    clf1 = GaussianNB()
    clf2 = AdaBoostClassifier()
    clf3 = tree.DecisionTreeClassifier()
    clf4 = RandomForestClassifier() 

classifier: RandomForestClassifier()
classifier
             precision    recall  f1-score   support (number of pages for testing)

          0       0.85      0.85      0.85       181
          1       0.87      0.92      0.89       154
          2       0.81      0.87      0.84       156
          3       0.88      0.87      0.87       174
          4       0.56      0.85      0.67       151
          5       0.81      0.91      0.86       158
          6       0.82      0.78      0.80       151
          7       0.95      0.90      0.92       177
          8       0.77      0.88      0.82       161
          9       0.81      0.80      0.81       147
         10       0.93      0.96      0.94       161
         11       0.91      0.91      0.91       172
         12       0.73      0.82      0.77       160
         13       0.88      0.96      0.92       161
         14       0.93      0.89      0.91       154
         15       0.85      0.86      0.86       162
         16       0.83      0.74      0.78       161
         17       0.87      0.89      0.88       148
         18       0.93      0.89      0.91       169
         19       0.84      0.80      0.82       185
         20       0.83      0.86      0.85       177
         21       0.91      0.90      0.91       162
         22       0.84      0.83      0.83       154
         23       0.93      0.86      0.89       167
         24       0.82      0.77      0.80       168
         25       0.94      0.94      0.94       132
         26       0.95      0.85      0.90       163
         27       0.88      0.70      0.78       159
         28       0.91      0.83      0.87       163
         29       0.80      0.80      0.80       147
         30       0.94      0.96      0.95       152
         31       0.63      0.66      0.65       160
         32       0.85      0.92      0.88       166
         33       0.90      0.88      0.89       145
         34       0.83      0.77      0.80       154
         35       0.93      0.89      0.91       169
         36       0.94      0.91      0.93       149
         37       0.80      0.84      0.82       165
         38       0.90      0.84      0.87       157
         39       0.94      0.87      0.91       175
         40       0.93      0.92      0.92       166
         41       0.88      0.83      0.86       151
         42       0.76      0.82      0.79       157
         43       0.86      0.86      0.86       154
         44       0.93      0.89      0.91       159
         45       0.84      0.87      0.86       143
         46       0.86      0.81      0.83       155
         47       0.91      0.76      0.83       161
         48       0.91      0.84      0.87       177
         49       0.91      0.84      0.87       137
         50       0.76      0.88      0.82       173

avg / total       0.86      0.85      0.86      8160


step 2:
U.S. pages count: 1,009,135 
             precision    recall  f1-score   support

          0       0.58      0.85      0.69      9019
          1       0.66      0.92      0.77      5815
          2       0.76      0.83      0.79     25350
          3       0.39      0.85      0.53      3209
          4       0.89      0.76      0.82    176819
          5       0.80      0.80      0.80     19806
          6       0.44      0.82      0.57      4540
          7       0.38      0.87      0.53      1359
          8       0.92      0.79      0.85     86547
          9       0.66      0.78      0.72     14380
         10       0.81      0.88      0.84     11967
         11       0.83      0.87      0.85     10760
         12       0.81      0.82      0.82     54917
         13       0.83      0.84      0.84     17950
         14       0.87      0.84      0.85     10106
         15       0.79      0.85      0.82      9661
         16       0.47      0.79      0.59      3567
         17       0.90      0.89      0.89     20615
         18       0.77      0.89      0.82      5845
         19       0.64      0.77      0.70     11167
         20       0.86      0.80      0.83     18490
         21       0.89      0.85      0.87     26258
         22       0.72      0.81      0.76     10700
         23       0.75      0.83      0.79      5154
         24       0.58      0.78      0.67      7282
         25       0.83      0.93      0.88      8088
         26       0.53      0.84      0.65      1694
         27       0.14      0.80      0.24      1536
         28       0.61      0.84      0.70      3435
         29       0.77      0.78      0.78     26262
         30       0.84      0.88      0.86     10179
         31       0.83      0.72      0.77     92543
         32       0.87      0.80      0.83     20895
         33       0.63      0.91      0.74      1618
         34       0.68      0.75      0.72      9732
         35       0.86      0.86      0.86     15875
         36       0.75      0.82      0.79      9701
         37       0.89      0.77      0.82     35231
         38       0.57      0.86      0.69      2949
         39       0.84      0.80      0.82     11206
         40       0.80      0.89      0.84      3825
         41       0.65      0.77      0.70      7205
         42       0.91      0.82      0.86     65949
         43       0.73      0.83      0.78     13130
         44       0.73      0.87      0.80      3712
         45       0.76      0.78      0.77     14387
         46       0.92      0.83      0.87     51140
         47       0.45      0.83      0.58      2273
         48       0.88      0.86      0.87     22348
         49       0.54      0.81      0.65      2939
         50       0.00      0.00      0.00         0

avg / total       0.83      0.80      0.81   1009135

step 1:


             precision    recall  f1-score   support

          0       0.03      0.89      0.06     25740
          1       0.61      0.42      0.50     13317
          2       0.76      0.73      0.74     57075
          3       0.44      0.28      0.34     11172
          4       0.89      0.35      0.50    409149
          5       0.80      0.39      0.52     44211
          6       0.46      0.31      0.37     12823
          7       0.48      0.70      0.57      3321
          8       0.91      0.35      0.50    210077
          9       0.64      0.31      0.42     40905
         10       0.80      0.42      0.55     27453
         11       0.82      0.41      0.55     23801
         12       0.82      0.39      0.53    121531
         13       0.84      0.39      0.53     41323
         14       0.85      0.70      0.77     25726
         15       0.02      0.45      0.04     22961
         16       0.54      0.28      0.37     11433
         17       0.89      0.77      0.82     49844
         18       0.79      0.39      0.52     14370
         19       0.63      0.32      0.42     29364
         20       0.85      0.34      0.49     46039
         21       0.83      0.37      0.51     64087
         22       0.74      0.34      0.47     27306
         23       0.75      0.30      0.43     14939
         24       0.54      0.30      0.38     21355
         25       0.83      0.83      0.83     16647
         26       0.59      0.30      0.40      5432
         27       0.19      0.54      0.28      4462
         28       0.30      0.64      0.40      8380
         29       0.79      0.31      0.45     68857
         30       0.83      0.41      0.55     22947
         31       0.83      0.31      0.45    215337
         32       0.86      0.35      0.50     51522
         33       0.64      0.33      0.44      4737
         34       0.74      0.56      0.63     29615
         35       0.85      0.38      0.53     37691
         36       0.73      0.34      0.47     24476
         37       0.88      0.34      0.49     87589
         38       0.60      0.34      0.43      7982
         39       0.86      0.33      0.47     29750
         40       0.81      0.78      0.80      8905
         41       0.67      0.30      0.41     20656
         42       0.92      0.34      0.50    168287
         43       0.77      0.66      0.71     29224
         44       0.73      0.72      0.73      8061
         45       0.77      0.34      0.47     36672
         46       0.91      0.42      0.57    106982
         47       0.47      0.30      0.37      6790
         48       0.87      0.39      0.54     52957
         49       0.63      0.64      0.63      7076
         50       0.00      0.00      0.00         0

avg / total       0.81      0.39      0.51   2430356
