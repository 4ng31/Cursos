# -*- coding: utf-8 -*-
## Modules
import pandas as pd
import numpy as np
from numpy import linalg
import scipy.io as spio
from sklearn import preprocessing ## Para estandarizar y normalizar
from sklearn.cross_validation import train_test_split ## Para partir el set de datos
from sklearn import cross_validation ## Validacion cruzada
import cvxopt
import cvxopt.solvers

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))


def svmfit(X, y, kernel, PK, C=None):
    n_samples, n_features = X.shape
    if C is not None: C = float(C)
    # Gram matrix
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            if kernel == 'linear_kernel':
                K[i,j] = linear_kernel(X[i], X[j])
            if kernel == 'gaussian_kernel':
                K[i,j] = gaussian_kernel(X[i], X[j],PK)
            if kernel == 'polynomial_kernel':
                K[i,j] = polynomial_kernel(X[i], X[j],PK)

    P = cvxopt.matrix(np.outer(y,y) * K)
    q = cvxopt.matrix(np.ones(n_samples) * -1)
    A = cvxopt.matrix(y, (1,n_samples))
    b = cvxopt.matrix(0.0)

    if C is None:
        G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h = cvxopt.matrix(np.zeros(n_samples))
    else:
        tmp1 = np.diag(np.ones(n_samples) * -1)
        tmp2 = np.identity(n_samples)
        G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
        tmp1 = np.zeros(n_samples)
        tmp2 = np.ones(n_samples) * C
        h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

    # solve QP problem
    cvxopt.solvers.options['show_progress'] = False
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)

    # Lagrange multipliers
    a = np.ravel(solution['x'])
    
    # Support vectors have non zero lagrange multipliers
    tol = 1.0e-5
    idx = np.where(a>tol) ## Indice donde multiplicadores distintos de cero
    idx=idx[0]
    a = a[idx]            ## Multiplicadores distintos de cero
    sv_x = X[idx]         ## Elementos X 
    sv_y = y[idx]         ## Elementos Y 
    
    #print "%d support vectors out of %d points" % (len(a), n_samples)

    # Intercept
    b = 0
    for n in range(len(a)):
        b += sv_y[n]
        b -= np.sum(a * sv_y * K[idx[n],idx])
    b /= len(a)

    # Weight vector
    if kernel == linear_kernel:
        w = np.zeros(n_features)
        for n in range(len(self.a)):
            w += a[n] * sv_y[n] * sv[n]
    else:
        w = None
        
    return a,sv_y,sv_x,w,b
    

def project(X, kernel, a, sv_y, sv, w, b, PK):
    if w is not None:
        return np.dot(X, w) + b
    else:
        a = np.array(a)[np.newaxis]
        sv_y = np.array(sv_y)[np.newaxis]
        
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            for j in range(len(sv)):
                s = 0
                if kernel == 'linear_kernel':
                    s += a.T[j]*sv_y.T[j]*linear_kernel(X[i],sv[j]);
                if kernel == 'gaussian_kernel':
                    s += a.T[j]*sv_y.T[j]*gaussian_kernel(X[i],sv[j], PK);
                if kernel == 'polynomial_kernel':
                    s += a.T[j]*sv_y.T[j]*polynomial_kernel(X[i],sv[j], PK);
            y_predict[i] = s
        return y_predict + b

def predict(X, kernel, a, sv_y, sv, w, b, PK):
    return np.sign(project(X, kernel, a, sv_y, sv, w, b, PK))
