# -*- coding: utf-8 -*-

import pandas as pd
import math
import numpy as np
import numpy.matlib
from numpy.linalg import inv
import scipy.io as spio
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

def norm_pdf_multivariate(x, mu, sigma):
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")

        norm_const = 1.0/ ( math.pow((2*np.pi),float(size)/2) * math.pow(det,1.0/2) )
        x_mu = np.matrix(x - mu)
        sigmaI = inv(sigma)        
        result = math.pow(math.e, -0.5 * (x_mu * sigmaI * x_mu.T))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")

def myqd(Xtrain,Xtest,Ytrain,Ytest):
    
    trainSet=np.c_[ Xtrain, Ytrain ]
    testSet=np.c_[ Xtest, Ytest ]
    
    dftrain = pd.DataFrame.from_records(trainSet)
    dftrain.columns=[np.arange(Xtest.shape[1]).tolist()+list(['label'])]

    testSet=np.c_[ Xtest, Ytest ]
    dftest = pd.DataFrame.from_records(testSet)
    dftest.columns=[np.arange(Xtest.shape[1]).tolist()+list(['label'])]
    
    X = dftrain[np.arange(Xtest.shape[1]).tolist()].values 
    y = dftrain['label'].values
    
    Xts = dftest[np.arange(Xtest.shape[1]).tolist()].values 
    Yts = dftest['label'].values

    n_samples, n_features = X.shape
    classes = np.unique(y)
    n_classes = classes.size
    classes_indices = [(y == c).ravel() for c in classes]

    counts = np.array(ndimage.measurements.sum(np.ones(n_samples, dtype=y.dtype), y, index=classes))
    priors = counts / float(n_samples)

    means = []
    covs = []
    for group_indices in classes_indices:
        Xg = X[group_indices, :]
        meang = Xg.mean(0)   # Media de cada clase
        means.append(meang)  #  
        covg = np.cov(Xg,bias=True, rowvar=False)
        covs.append(covg)  ## Covarianza de cada clase

    prob_clases = []
    P1 = np.zeros(len(Xts))
    P2 = np.zeros(len(Xts))
    for i in range(0,len(Xts),1):
        P1[i]=norm_pdf_multivariate(Xts[i],means[0],covs[0])
        P2[i]=norm_pdf_multivariate(Xts[i],means[1],covs[1])
    

    PP = np.true_divide(P1,P2)*(priors[0]/priors[1])
    pred = np.zeros(len(Xts))
    for i in range(0,len(Xts),1):
        if PP[i]>=1:
            pred[i]=1;
        else:
            pred[i]=2;
            
    return (Xts,Yts, pred)
