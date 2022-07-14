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
