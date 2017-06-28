# -*- coding: utf-8 -*-

## Modules
import numpy as np
import math
from scipy.stats import multivariate_normal

def parzen_window(X_train,Y_train,X_test,BW):

    Nlabels = X_train.shape[1]
    Y_hat = np.zeros(len(X_test))
    sigma = BW * np.eye(Nlabels, dtype=int)
  
    acum = 0;
    for i in range(0,len(X_test),1):
        for j in range(0,len(X_train),1):
            gauss = multivariate_normal.pdf(X_test[i], X_train[j], cov=sigma)
            if Y_train[j] == 1:
                acum = acum+ 1*gauss
            else:
                acum = acum+ (-1)*gauss
        if np.sign(acum) == np.sign(1):
            Y_hat[i] =  1;
        else:
            Y_hat[i] = 2;
    return Y_hat

