# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import scipy.io as spio
from sklearn.neighbors import KNeighborsClassifier ## kNN
from sklearn import preprocessing ## Para estandarizar y normalizar
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab as pl
import random
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV

filename2='/home/bgx/trabajofinaldeprctica/datosOS14.mat'

#Load data from file return dataFrame
mat = spio.loadmat(filename2,squeeze_me=True)
datos = mat['datos'].tolist()
X=datos[0];Y=datos[1];
A=np.insert(X, X.shape[1], Y, axis=1)
idx = range(0, A.shape[0])
idy = list(range(0, A.shape[1]-1))
idy.append('label')
df = pd.DataFrame.from_records(A, columns=(idy), index=idx)
df.columns=['a','b','c','d','e','f','g','h','i','j','label']

## All features
X_data = df.values[:,0:10]
## Some features
#features = ['a','b', 'c','d', 'e',]
#X_data = df.ix[:,features].values
## Labels
Y_data = df.values[:,10]

results=[]
klabels=range(2,20,2)
mydf = pd.DataFrame(columns=list(klabels))
for j in range(100):
    dfr = df.reindex(np.random.permutation(df.index))
    ## Some features
    features = ['a','b', 'c','d', 'e',]
    X = dfr.ix[:,features].values
    #X = np.array(dfr.values[:,0:10])
    Y = data_df.values[:,10]
    X = preprocessing.scale(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=100, random_state=j)

    kscores =[]
    for n in klabels:
        clf = KNeighborsClassifier(n_neighbors=n)
        clf.fit(X_train, Y_train)
        preds = clf.predict(X_test)
        accuracy = np.where(preds==Y_test, 1, 0).sum() / float(len(X_test))
        #print "Neighbors: %d, Accuracy: %3f" % (n, accuracy)
        kscores.append(accuracy)
    mydf.loc[len(mydf)] = kscores
pl.plot(mydf.mean(axis=0))
pl.title("Accuracy with Increasing K")
pl.show()
#print df1.head()



## CROSS-VALIDATION
# check CV score for K
# search for an optimal value of K
k_range = range(2, 30, 2)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores.append(np.mean(cross_val_score(knn, X_data, Y_data, cv=5, scoring='accuracy')))
scores

# plot the K values (x-axis) versus the 5-fold CV score (y-axis)
plt.figure()
plt.plot(k_range, scores)

# automatic grid search for an optimal value of K
knn = KNeighborsClassifier()
k_range = range(2, 12, 2)
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid.fit(X_data, Y_data)

# check the results of the grid search
#print grid.grid_scores_
grid_mean_scores = [result[1] for result in grid.grid_scores_]
plt.figure()
plt.plot(k_range, grid_mean_scores)
#print grid.best_score_
#print grid.best_params_
#print grid.best_estimator_