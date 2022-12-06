#%%
import csv
import pandas as pd

#%%
def get_y_label(file_path: str):
    '''Node-level ground-truth labels as 243 country classes.

    Args:
        file_path: A string of file path for the c_label.csv contains country labels.
    Return:
        y: A torch.Tensor with shape (num_nodes).
    '''
    '''
    us_y = self.get_y_label('./new_city_data/new_cities_2hop_bfs_2_round_id_idx_city_dupStates_trueLabel_mostPopulationLabel_2ndRoundDupstate0_1stRoundDupstate0.csv')
    4846711747,0,san francisco,1,4,4,4,4
    5246634919,20,redwood city,1,4,4,4,4
    5281959998,23,new york,1,31,31,31,31
    5340039981,32,virginia beach,1,45,45,45,45
    5352734382,34,new york,1,31,31,31,31
    5381334667,42,oakland,16,-1,4,4,4
    5417674986,53,washington,18,-1,50,50,50
    5459524943,77,syracuse,7,-1,31,31,31
    5461604986,78,west valley city,1,43,43,43,43
    5466504237,81,inglewood,2,-1,4,4,4
    '''

    df = pd.read_csv(file_path, sep=',', header=None)
    trueLabel_df = df.iloc[:, 4:5]
    city_df = df.iloc[:, 2:3]
    count_washington = 0
    count_true_label = 0
    
    city = city_df.values
    trueLabel = trueLabel_df.values
    print('trueLabel[0:10]', trueLabel[0:10])
    print('city[0:10]', city[0:10])
    for i in range(len(city)):
        # print(city_df.values[i][0], trueLabel_df.values[i][0])
        if city[i][0] == 'washington':
            trueLabel[i][0] = 50
            count_washington += 1
        if trueLabel[i][0] != -1:
            count_true_label += 1
    print('count_washington',count_washington, 'count_true_label',count_true_label)
    # df.to_csv('./data/raw_dir/us_pages_lgc_with_new_label.csv', sep='\t', index=True,
    #           header=False)
    y = trueLabel.T[0]

    mask = (y != -1)
    print('y.shape',y.shape)
    print('y[0:10]',y[0:10])
    print('mask.shape',mask.shape)
    print('mask[0:10]',mask[0:10])
    print('mask.sum()', mask.sum())

    return (y, mask)
#%%
us_y, true_mask = get_y_label('../new_city_data/new_cities_2hop_bfs_2_round_pl_id_idx_city_dupStates_trueLabel_mostPopulationLabel_2ndRoundDupstate0_1stRoundDupstate0.csv')

# %%
degreefile = '../data/raw_dir/us_lgc_page_id_name_category_city_likespage_fan_outward_inward.csv'
df = pd.read_csv(degreefile, sep='\t', header=None)
out_degree = df.iloc[:, 6:7].values
in_degree = df.iloc[:, 7:8].values
undirected_degree = out_degree + in_degree
out_degree = out_degree.T[0]
in_degree = in_degree.T[0]
undirected_degree = undirected_degree.T[0]
#%%
print(out_degree.shape, in_degree.shape, undirected_degree.shape, type(out_degree))
print(out_degree.min(), out_degree.max(), out_degree[0:10])
print(in_degree.min(), in_degree.max(), in_degree[0:10])
print(undirected_degree.min(), undirected_degree.max(), undirected_degree[0:10])

# %%
labeled_un_degree = []
unlabeled_un_degree = []
labeled_in_degree = []
unlabeled_in_degree = []
labeled_out_degree = []
unlabeled_out_degree = []
for i in range(len(undirected_degree)):
    if true_mask[i]:
        labeled_un_degree.append(undirected_degree[i])
        labeled_in_degree.append(in_degree[i])
        labeled_out_degree.append(out_degree[i])
    else:
        unlabeled_un_degree.append(undirected_degree[i])
        unlabeled_in_degree.append(in_degree[i])
        unlabeled_out_degree.append(out_degree[i])   
# %%
import matplotlib.pyplot as plt
def plot_hist(x, title):
    f = plt.figure()
    f.set_figwidth(8)
    f.set_figheight(4)
    f.set_dpi(100)
    plt.hist(x)
    plt.xlabel('page degree', fontsize=16) 
    plt.ylabel('number of pages', fontsize=16) 
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yscale('log')
    # plt.xscale('log')
    # plt.title(title)
    plt.show()
# %%
plot_hist(unlabeled_un_degree, 'Histogram of the degrees of non-deterministic pages, dataset B')
#%%
plot_hist(labeled_un_degree, 'Histogram of the degrees of deterministic pages, dataset A')
# %%
plot_hist(unlabeled_in_degree, 'unlabeled_in_degree')
plot_hist(labeled_in_degree, 'labeled_in_degree')
# %%
plot_hist(unlabeled_out_degree, 'unlabeled_out_degree')
plot_hist(labeled_out_degree, 'labeled_out_degree')
# %%
