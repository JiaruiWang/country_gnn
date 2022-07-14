D1=/home/jerry/Documents/public_page_graph/
# out=$D1/Adv_BFS_with_dup_states/BFS_output/
out=$D1/Adv_BFS_same/BFS_output/
D2=$D1/20171217/

echo $D1
echo $out
echo $D2
#clean previous outputs
#rm -rf ./output/*
rm -f $out/*
#./BFS_seeds ./test/node.csv ./test/edge.csv ./test/seed.csv ./test/state.csv ./output/
./BFS_seeds\
    $D2/page.csv\
    $D2/edge.csv\
    $D2/seed_no_DC.csv\
    $D2/state.csv\
    $out\
    ./state_classes.csv
