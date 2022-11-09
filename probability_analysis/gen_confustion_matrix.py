# %% Import packages
import pandas as pd
import os.path as osp
import numpy as np

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib as mtl
mtl.style.use('ggplot')

# %%
pl_class_names = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT",\
                  "DE", "FL", "GA", "HI", "ID", "IL", "IN",\
                  "IA", "KS", "KY", "LA", "ME", "MD", "MA",\
                  "MI", "MN", "MS", "MO", "MT", "NE", "NV",\
                  "NH", "NJ", "NM", "NY", "NC", "ND", "OH",\
                  "OK", "OR", "PA", "RI", "SC", "SD", "TN",\
                  "TX", "UT", "VT", "VA", "WA", "WV", "WI",\
                  "WY", "DC"]

mnl_class_names = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT",\
                   "DE", "FL", "GA", "HI", "ID", "IL", "IN",\
                   "IA", "KS", "KY", "LA", "ME", "MD", "MA",\
                   "MI", "MN", "MS", "MO", "MT", "NE", "NV",\
                   "NH", "NJ", "NM", "NY", "NC", "ND", "OH",\
                   "OK", "OR", "PA", "RI", "SC", "SD", "TN",\
                   "TX", "UT", "VT", "VA", "WA", "DC", "WV",\
                   "WI", "WY"]

# %%
# file_path1 = '../new_city_data/us_pages_lgc_true_label_51_label.csv'
# twolabel = pd.read_csv(file_path1, sep='\t', header=None)
# file_path2 = '../new_city_data/us_pages_lgc_idx_id_mask_label_state.csv'
# fivelabel = pd.read_csv(file_path2, sep='\t', header=None)
mnl_path = '../model/saint_all_label/saint_id_y_pred_51probability_setseed_test1.csv'
mnl_out = pd.read_csv(mnl_path, sep=',', header=None)
pl_path = '../model/saint_population_label_all_label/saint_inference_output_id_y_pred_51probability.csv'
pl_out = pd.read_csv(pl_path, sep=',', header=None)
# %%
mnl_y = mnl_out.values[:,1:2].flatten()
mnl_pred = mnl_out.values[:,2:3].flatten()

pl_y = pl_out.values[:,1:2].flatten()
pl_pred = pl_out.values[:,2:3].flatten()
print(mnl_y)
print(mnl_pred)
print(pl_y)
print(pl_pred)

#%%

#%%
def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)
    np.savetxt('cm.csv', cm, fmt='%.2f', delimiter=',')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    """
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    """

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
#%%
# Compute confusion matrix
cnf_matrix = confusion_matrix(pl_y, pl_pred, normalize='true')
cnf_matrix = cnf_matrix * 100
np.savetxt('pl_cm_1f.csv', cnf_matrix, fmt='%0.1f', delimiter=',')
cnf_matrix = confusion_matrix(mnl_y, mnl_pred, normalize='true')
cnf_matrix = cnf_matrix * 100
np.savetxt('mnl_cm_1f.csv', cnf_matrix, fmt='%0.1f', delimiter=',')

#%%
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=pl_class_names,
#                         title='Confusion matrix, without normalization')
# fname = "pl_confusion"

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=pl_class_names, normalize=True,
                        title='Normalized confusion matrix')

#plt.show()
PNG_PATH = "./"
fname = "pl_confusion"
png_name = PNG_PATH + fname + ".png"
plt.savefig(png_name, dpi=300, format='png')

# %%