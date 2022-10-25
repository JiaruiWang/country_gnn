1. generate feature vectors from anchored seeds
~/code/c++/crawler_v2/latestCode/state_labeler/BFS_from_anchored_pages/run.sh > 20171219.log
~/code/c++/crawler_v2/latestCode/state_labeler/BFS_from_anchored_pages/run_2lists.sh > 20171222_2lists.log
memory: 8% ~= 120*.08 ~= 10G memory
time: 20:40 starts 23:46 ends ~= 3 hours

2. clean all-zero feature vectors which are not reached by any seeds
rm -f /data1/yclin123/public_page_graph/BFS_output/Guam.csv
rm -f /data1/yclin123/public_page_graph/BFS_output/Puerto Rico.csv
~/code/python2/state_labeler/python clean_data.py > 20171219_cl.log
3. select min2 feature vectors from each state and others
~/code/python2/state_labeler/python select_fv_min2.py > 20171221_sl_min2.log
4. select N feature vectors from each state and others (N <= common minimum number)
~/code/python2/state_labeler/python select_fv.py > 20171221_sl.log
5. upload feature vector files into docker ./labeller/features/
6. replace others.csv by the first N feature vectors instead
7. run ML classifier
./labeller/feature_ml-master.ipynb

precision    recall  f1-score   support

           0       0.02      0.95      0.03     25740
           1       0.83      0.43      0.56     13317
           2       0.84      0.40      0.54     57075
           3       0.61      0.28      0.39     11172
           4       0.90      0.35      0.50    409149
           5       0.85      0.39      0.53     44211
           6       0.61      0.32      0.42     12823
           7       0.59      0.38      0.46      3321
           8       0.93      0.36      0.51    210077
           9       0.74      0.32      0.44     40905
          10       0.84      0.42      0.56     27453
          11       0.85      0.42      0.56     23801
          12       0.82      0.38      0.52    121531
          13       0.86      0.39      0.54     41323
          14       0.89      0.35      0.51     25726
          15       0.84      0.38      0.52     22961
          16       0.58      0.29      0.39     11433
          17       0.90      0.40      0.55     49844
          18       0.84      0.39      0.54     14370
          19       0.69      0.32      0.43     29364
          20       0.88      0.35      0.50     46039
          21       0.89      0.38      0.53     64087
          22       0.77      0.35      0.48     27306
          23       0.79      0.32      0.46     14939
          24       0.65      0.29      0.41     21355
          25       0.86      0.52      0.65     16647
          26       0.72      0.63      0.67      5432
          27       0.20      0.29      0.24      4462
          28       0.65      0.37      0.47      8380
          29       0.81      0.33      0.47     68857
          30       0.86      0.42      0.56     22947
          31       0.81      0.33      0.47    215337
          32       0.87      0.36      0.51     51522
          33       0.70      0.34      0.46      4737
          34       0.72      0.28      0.41     29615
          35       0.87      0.39      0.54     37691
          36       0.77      0.35      0.48     24476
          37       0.89      0.35      0.50     87589
          38       0.58      0.35      0.44      7982
          39       0.85      0.34      0.49     29750
          40       0.83      0.42      0.56      8905
          41       0.67      0.31      0.42     20656
          42       0.91      0.35      0.51    168287
          43       0.73      0.41      0.53     29224
          44       0.73      0.42      0.54      8061
          45       0.80      0.34      0.48     36672
          46       0.91      0.43      0.58    106982
          47       0.54      0.30      0.39      6790
          48       0.87      0.39      0.54     52957
          49       0.62      0.36      0.46      7076
          50       0.00      0.00      0.00         0

    accuracy                           0.37   2430356
   macro avg       0.74      0.37      0.48   2430356
weighted avg       0.84      0.37      0.50   2430356
