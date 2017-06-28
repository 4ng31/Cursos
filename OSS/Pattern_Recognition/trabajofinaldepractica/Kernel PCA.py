# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%matplotlib inline

import pandas as pd
import scipy.io as spio
import numpy as np
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA
from matplotlib import pyplot as plt

def loadDataFrame(filename):
    ##Load data from file return dataFrame
    mat = spio.loadmat(filename,squeeze_me=True)
    datos = mat['datos'].tolist()
    X=datos[0];Y=datos[1];
    A=np.insert(X, X.shape[1], Y, axis=1)
    idx = range(0, A.shape[0])
    idy = list(range(0, A.shape[1]-1))
    idy.append('label')
    df = pd.DataFrame.from_records(A, columns=(idy), index=idx)
    df.columns=['a','b','c','d','e','f','g','h','i','j','label']
    return df


def mykpca(X, gamma, n_components):
    """
    Implementation of a RBF kernel PCA.

    Arguments:
        X: A MxN dataset as NumPy array where the samples are stored as rows (M),
           and the attributes defined as columns (N).
        gamma: A free parameter (coefficient) for the RBF kernel.
        n_components: The number of components to be returned.

    """
    # Calculating the squared Euclidean distances for every pair of points
    # in the MxN dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')

    # Converting the pairwise distances into a symmetric MxM matrix.
    mat_sq_dists = squareform(sq_dists)

    # Computing the MxM kernel matrix.
    K = exp(-gamma * mat_sq_dists)

    # Centering the symmetric NxN kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenvalues in descending order with corresponding 
    # eigenvectors from the symmetric matrix.
    eigvals, eigvecs = eigh(K)

    # Obtaining the i eigenvectors that corresponds to the i highest eigenvalues.
    X_pc = np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1)))

    return X_pc

def graph2features(X,Y):
    ## Graph by 2 features
 
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    plt.figure(2, figsize=(8, 6))
    plt.clf()

    ## Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

    plt.show()

def graphfeaturematrix(dataframe,classes):
    ### Plot Features Matrix    
    g = sns.PairGrid(dataframe, hue=classes)
    g.map_diag(plt.hist)
    g.map_offdiag(plt.scatter)
    g.add_legend()

def main():
    # Get Data
    filename='/home/bgx/trabajofinaldeprctica/datosOS14.mat'
    DF_data=loadDataFrame(filename)

    # on non-standardized data
    X = DF_data.values[:,:10]
    Y = DF_data.values[:,10]
    
    # Plot data by first 2 classes
    graph2features(X[:,:2],Y)

    NC=10

    ###  Stepwise KPCA
    X_pc = mykpca(X, gamma=15, n_components=NC)
    
    ### Toolbox KPCA
    scikit_kpca = KernelPCA(n_components=NC, kernel='rbf', gamma=15)
    X_skernpca = scikit_kpca.fit_transform(X)

    # Plot KPCA data by first 2 classes  
    graph2features(X_pc[:,:2],Y)
    graph2features(X_skernpca,Y)

    # Standarize Data
    std_scale = preprocessing.StandardScaler().fit(X)
    X_std = std_scale.transform(X)
    
    ###  Stepwise KPCA
    X_std_pc = mykpca(X_std, gamma=15, n_components=NC)
    
    ### Toolbox KPCA
    scikit_kpca = KernelPCA(n_components=NC, kernel='rbf', gamma=15)
    X_std_skernpca = scikit_kpca.fit_transform(X_std)

    # Plot KPCA data standarized (first 2 classes)
    graph2features(X_std_pc[:,:2],Y)
    graph2features(X_std_skernpca,Y)

main()

# <codecell>


# <codecell>


