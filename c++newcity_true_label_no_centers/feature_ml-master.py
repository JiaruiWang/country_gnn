# %% import pymysql
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import matplotlib as mtl
mtl.style.use('ggplot')
import collections
import pandas as pd
from scipy.stats import mannwhitneyu
from datetime import datetime
import time
import csv
import math
import itertools
import statistics as stat
# %matplotlib inline

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
#from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn import tree
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier

# FV_PATH = "./features_adv_min2_selected800/"
FV_PATH = '/home/jerry/Documents/jason_code/python code/state_labeler/adv/features_adv_min2_selected800/'
# PNG_PATH = './png/'
PNG_PATH = "/home/jerry/Documents/jason_code/python code/state_labeler/adv/png/"

#FV_PER_STATE = 1000
FV_PER_STATE = 800
CLASS_CNT = 51
ROW_CNT = FV_PER_STATE * CLASS_CNT # 800*51 = 40800
NUM_SEED = 51
COL_CNT = NUM_SEED * 2 + CLASS_CNT * 2 # NUM_SEED * (inward, outward) # 204
class_names = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT",\
               "DE", "FL", "GA", "HI", "ID", "IL", "IN",\
               "IA", "KS", "KY", "LA", "ME", "MD", "MA",\
               "MI", "MN", "MS", "MO", "MT", "NE", "NV",\
               "NH", "NJ", "NM", "NY", "NC", "ND", "OH",\
               "OK", "OR", "PA", "RI", "SC", "SD", "TN",\
               "TX", "UT", "VT", "VA", "WA", "WV", "WI",\
               "WY", "OT"]

s2i_table = {}  # hash table for state to index
i2s_table = {}  # hash table for index to state

# use neighbor info only for feature
# zero_data = np.zeros(shape=(ROW_CNT, int(COL_CNT/2))) # 40800 * 102

zero_data = np.zeros(shape=(ROW_CNT, COL_CNT)) # 40800 * 204

data = pd.DataFrame(zero_data)
print(data.shape)
def readStateMap(fname):
    with open(fname, "r") as f:
        cnt = 0
        for line in f.readlines():
            word = line.rstrip()
            state = word
            s2i_table[state] = cnt
            i2s_table[cnt] = state
            #print("c2i_table[country]: %d, i2c_table[cnt]: %s"  \
            #%(c2i_table[country], i2c_table[cnt]))
            csv_name = FV_PATH + state + ".csv"
            print(csv_name)
            readFeatures(csv_name, cnt * FV_PER_STATE)
            cnt += 1

def readFeatures(fname, row_cnt):
    #print(data.iloc[row_cnt])
    row_data = pd.read_csv(fname, header = None)
    if row_data.empty:
        print("%s is empty! row_cnt == %d" %(fname, row_cnt))
    else:
        for i in range(FV_PER_STATE):
            # row assignment has to be done row by row because block row assignment has bug.
            
            data.iloc[row_cnt + i] = row_data.iloc[i]

            # use neighbor info only for feature
            # nprow = row_data.iloc[i,0:102].to_numpy()
            # nprow = row_data.iloc[i,102:204].to_numpy()
            # nprow = row_data.iloc[i,0:204].to_numpy()
            # if not np.all(np.isfinite(nprow)):
            #     print(nprow)
            # data.iloc[row_cnt + i] = nprow

        #print(data.iloc[row_cnt:row_cnt + 2])

def genY():
    Y = [0] * ROW_CNT
    for i in range(CLASS_CNT):
        Y[i * FV_PER_STATE:(i + 1) * FV_PER_STATE] = [i] * FV_PER_STATE
    #print(Y)
    return Y

def plot_confusion_matrix(cm, classes,
                          normalize=False,
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
    np.savetxt('cm.csv', cm, fmt='%4d', delimiter=',')

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

def classSub(clf, X, y, random_state, cv):
    #scores = cross_val_score(clf, X, y, cv = cv)
    #print(scores)
    #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    X_train, X_test, y_train, y_test = train_test_split(X, y, \
    test_size = .2, random_state = random_state)
    return X_train, X_test, y_train, y_test

def classFit(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    predicts = clf.predict(X_test)
    print(classification_report(y_test, predicts))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, predicts)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')
    """
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    """
    #plt.show()
    fname = "confusion"
    png_name = PNG_PATH + fname + ".png"
    plt.savefig(png_name, dpi=300, format='png')
    return predicts

def classifier(clf, X, y, random_state, cv):
    X_train, X_test, y_train, y_test = classSub(clf, X, y, random_state, cv)
    print("classifier")
    predicts = classFit(clf, X_train, X_test, y_train, y_test)

def featImportance(X, y, fname):
    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)
    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking: %s" %(fname))

    for f in range(X.shape[1]):
        if importances[indices[f]] >= 0.01:
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    #plt.title("Feature importances")
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Importance")
    plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    #plt.xlim([-1, X.shape[1]])
    plt.xlim([-1, 10.5])
    #plt.show()
    #fname = "feat_importance"
    png_name = PNG_PATH + fname + ".png"
    plt.savefig(png_name, dpi=300, format='png')

def dataFit():
    #random_state = np.random.RandomState(0)
    random_state = 7    # fixed seed: 5,
    y_list = genY()
    y = np.array(y_list)

    feat_data = data
    data2 = feat_data.values

    #featImportance(data2, y, "feat_importance")

    clf1 = GaussianNB()
    clf2 = AdaBoostClassifier()
    clf3 = tree.DecisionTreeClassifier()    # could be fine-tuned
    clf4 = RandomForestClassifier()         # could be fine-tuned
    #clf5 = GradientBoostingClassifier()     # could be fine-tuned

    
    print("classifier: GaussianNB()")
    classifier(clf1, data2, y, random_state, 5)
    print("classifier: AdaBoostClassifier()")
    classifier(clf2, data2, y, random_state, 5)
    print("classifier: DecisionTreeClassifier()")
    classifier(clf3, data2, y, random_state, 5)
    
    print("classifier: RandomForestClassifier()")
    classifier(clf4, data2, y, random_state, 5)
    #print("classifier: GradientBoostingClassifier()")
    #classifier(clf5, data2, y, random_state, 5)

readStateMap('/home/jerry/Documents/jason_code/python code/state_labeler/adv/state_classes.csv')
dataFit()

# %%
