#clean previous outputs
rm -rf ./output/*
./count_neighbor ../data/raw_dir/us_pages_lgc_idx_id_mask_label_state.csv ../data/raw_dir/us_edges_lgc_relabeled.csv ../data/raw_dir/state_classes.csv ./output/

