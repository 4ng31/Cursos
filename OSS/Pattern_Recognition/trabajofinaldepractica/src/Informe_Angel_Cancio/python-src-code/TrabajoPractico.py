# -*- coding: utf-8 -*-
#%matplotlib inline

#--------------------------------------------------------------------------
#
#    Pattern Recognition (OS2)
#
#    Profesores:
#                Dr. Beauseroy
#                Dr. Tomassi
#--------------------------------------------------------------------------
# Alumno: Cancio Montbrun, Angel Alberto
# Fecha: 31/05/2015
#--------------------------------------------------------------------------

## Modules
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import scipy.io as spio
from sklearn.neighbors import KNeighborsClassifier
from sklearn import grid_search
from sklearn import preprocessing ## Para estandarizar y normalizar
from sklearn import cross_validation
from sklearn.neighbors import KernelDensity
from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab as pl
import random
import math
import operator
from sklearn import svm

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.lda import LDA
from sklearn.qda import QDA
from scipy.stats import multivariate_normal

import sys
sys.path.append("/home/bgx/trabajofinaldeprctica/src/func") 
import myKNN
import myParzen
import myLD
import myQD
import mySVM
import myKPCA

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

def SplitDataFrame(data, Ntrain,SN=True):
    ## Suffle rows in DataFrame
    dfr = data.reindex(np.random.permutation(data.index))
    X = dfr.values[:,0:10]
    Y = dfr.values[:,10]
    N=X.shape[0]
    Ntest = N - Ntrain

    if SN:
        #Feature Scaling - Standardization && Normalization
        std_scale = preprocessing.StandardScaler().fit(X)
        X_std = std_scale.transform(X)
    else:
        X_std = X

    X_train, X_test, Y_train, Y_test = train_test_split(X_std, Y, test_size=Ntest, random_state=42)
    return (X_train, X_test, Y_train, Y_test)

def getAccuracy(X_test,Y_test, pred):
    #testSet=np.c_[ X_test, Y_test ]
    correct = incorrect = 0
    for x in range(len(Y_test)):
        #if testSet[x][-1] == pred[x]:
        if Y_test[x] == pred[x]:
            correct += 1
        else:
            incorrect += 1
        accuracy=(correct/float(len(Y_test))) * 100.0
        error=(incorrect/float(len(Y_test))) * 100.0
    return (accuracy, error)


def plot_step_lda(XX,YY):
 
    ax = plt.subplot(111)
    for label,marker,color in zip(range(1,3),('^', 's'),('blue', 'red')):
 
        plt.scatter(x=XX[:,0][YY == label],
                y=XX[:,1][YY == label],
                marker=marker,
                color=color,
                alpha=0.5,
                label=('Class')
                )
 
    plt.xlabel('X1')
    plt.ylabel('X2')
 
    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title('Data first 2 features')
 
    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
            labelbottom="on", left="off", right="off", labelleft="on")
 
    # remove axis spines
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False) 
    ax.spines["bottom"].set_visible(False) 
    ax.spines["left"].set_visible(False)
 
    plt.grid()
    plt.tight_layout
    plt.show()

def plot_step_qda(XX,YY):
 
    ax = plt.subplot(111)
    for label,marker,color in zip(range(1,3),('^', 's'),('blue', 'red')):
 
        plt.scatter(x=XX[:,0][YY == label],
                y=XX[:,1][YY == label],
                marker=marker,
                color=color,
                alpha=0.5,
                label=('Class')
                )
 
    plt.xlabel('X1')
    plt.ylabel('X2')
 
    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title('Data first 2 features')
 
    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
            labelbottom="on", left="off", right="off", labelleft="on")
 
    # remove axis spines
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False) 
    ax.spines["bottom"].set_visible(False) 
    ax.spines["left"].set_visible(False)
 
    plt.grid()
    plt.tight_layout
    plt.show()


def main():
    # Get Data
    filename='/home/bgx/trabajofinaldeprctica/datosOS14.mat'
    DF_data=loadDataFrame(filename)

    ## Sample train SIZE
    SIZE=179
   
    ## Get Xtrain, Xtest, Ytrain, Ytest - Standarization and Normalization SN=False
    Xtrain,Xtest,Ytrain,Ytest=SplitDataFrame(DF_data, SIZE,SN=True)

    ## Optimo usando el CrossValidation en el TrainSet
    ## Cambio el nombre de la segunda clase
    X=Xtrain
    Y=Ytrain
    ## KFold divides all the samples in math:K groups of samples, called folds .
    #Simple K-Fold cross validation. 10 folds. Return Indexes for Xtrain and Ytrain
    # Kfold = N => Leave One Out strategy
    folds=10
    kf_total = cross_validation.KFold(len(X), n_folds=folds, shuffle=True, random_state=8)
  
    for KPCA in (False,True):
        if KPCA:
            print 'KPCA True'
            print '#######################################################'
            print '########  Kernel Principal Component Analysis  ########'
            print '#######################################################'
        
            XX = np.vstack((Xtrain,Xtest))
    
            # Standardize
            X_std = preprocessing.StandardScaler().fit_transform(XX)

            # PCA
            #sklearn_pca = PCA(n_components=4)
            #X = sklearn_pca.fit_transform(X_std)
                          
            X = myKPCA.mykpca(XX, gamma=0.5, n_components=9)
                       
            #kpca = KernelPCA(kernel="poly", gamma=0.1, degree = 4, n_components=8)
            #X = kpca.fit_transform(XX)
            #X_back = kpca.inverse_transform(X_kpca)
            
            Xtrain = X[:SIZE,:]
            Xtest = X[SIZE:,:]
        else:
            print 'KPCA False'  

        ## Optimo usando el CrossValidation en el TrainSet
        ## Cambio el nombre de la segunda clase
        X=Xtrain
        Y=Ytrain
        print '#######################################################'
        print '########                 kNN                   ########'
        print '#######################################################'
        ## Tune
        K_vec=range(1,20+1,1)     ## Parametros K a evaluar
        Err_all_k = []
        ## Find Optimal K using CrossValidation on TrainSet
        for K in K_vec: 
            Err_k = np.array([])
            for train, test in kf_total:
                ## Test Classifier Over KFold
                pred=[]
                pred=myKNN.kNNClass(X[train],X[test],Y[train],Y[test],K)
                accuracy,error=getAccuracy(X[test],Y[test], pred) # Error para cada K en este Set
                Err_k = np.append(Err_k,error)
                # print('Accuracy: ' + repr(accuracy) + '%')
            Err_all_k.append(Err_k)
        meanEk=np.mean(np.vstack(Err_all_k),axis=1)  #Media de Error de c/K
        #print meanEk
        minindex=meanEk.argmin()
        BestK=K_vec[minindex]
    
        #print mean_err_knn_k
        print("Best K I found: {0}".format(BestK))   
    
        ## Same as above using Sklearn Tool
        parameters={'n_neighbors' : K_vec }
        knn=KNeighborsClassifier()
        clf=grid_search.GridSearchCV(knn,parameters,cv=5) #(CV is number of fold used for CrossValidation)
        clf.fit(Xtrain,Ytrain)
        #print("Best K: {0}".format(clf.best_estimator_.n_neighbors))
        print("Best K from toolbox: {0}".format(clf.best_params_['n_neighbors']))
        pred = clf.predict(Xtest)
        print("Test NN Toolbox")
        print('Accuracy: ' + repr(accuracy) + '%')

        
        ## Test Classifier
        pred=[]
        pred=myKNN.kNNClass(Xtrain,Xtest,Ytrain,Ytest,BestK)
        ## Results
        accuracy, error = getAccuracy(Xtest,Ytest, pred)

        print("Test myKNN with best K found: {0}".format(BestK))
        print('Accuracy: ' + repr(accuracy) + '%')

        print '#######################################################'
        print '########               Parzen                  ########'
        print '#######################################################'
    
        ## Tune
        BW_vec=np.arange(0.005, 1, 0.05)
        Err_all_bw = []
        ## Find Optimal BW using CrossValidation on TrainSet
        for bw in range(0,len(BW_vec),1): ## Find Optimal BW using CrossValidation on TrainSet
            sigma = BW_vec[bw]
            Err_bw = np.array([])
            for train, test in kf_total:
            ## Test Classifier Over KFold
                Yhat=[]
                Yhat = myParzen.parzen_window(X[train],Y[train],X[test],sigma)
                accuracy, error = getAccuracy(X[test],Y[test], Yhat)
                #print('Accuracy: ' + repr(accuracy) + '%')
                #The more training samples we have in the the training dataset, roughly speaking, 
                #the more accurate the estimation becomes (Central limit theorem) since we reduce 
                #the likelihood of encountering a sparsity of points for local regions - assuming 
                #that our training samples are i.i.d (independently drawn and identically distributed).
                accuracy,error=getAccuracy(X[test],Y[test], Yhat) # Error para cada BW en este Set
                Err_bw = np.append(Err_bw,error)
                # print('Accuracy: ' + repr(accuracy) + '%')
            Err_all_bw.append(Err_bw)
        meanEbw=np.mean(np.vstack(Err_all_bw),axis=1)  #Media de Error de c/K
        minindex=meanEbw.argmin()
        BestBW=BW_vec[minindex]
        #print mean_err_knn_k
        print("Best BW I found: {0}".format(BestBW))   
    
        # use grid search cross-validation to optimize the bandwidth
        params = {'bandwidth': np.logspace(-1, 1, 10)}
        clf=grid_search.GridSearchCV(KernelDensity(), params)
        clf.fit(Xtrain,Ytrain)
        print("Best BW from toolbox: {0}".format(clf.best_estimator_.bandwidth))
        
        # use the best estimator to compute the kernel density estimate
        # kde = clf.best_estimator_    
        Yhat = myParzen.parzen_window(Xtrain,Ytrain,Xtest,clf.best_estimator_.bandwidth)
        accuracy,error = getAccuracy(Xtest,Ytest, Yhat)
        print('Accuracy: ' + repr(accuracy) + '%')
    
        print("Test myParzen with best BW found: {0}".format(BestBW))
        Yhat = myParzen.parzen_window(Xtrain,Ytrain,Xtest,BestBW)
        accuracy,error = getAccuracy(Xtest,Ytest, Yhat)
        print('Accuracy: ' + repr(accuracy) + '%')
    
        print '#######################################################'
        print '########          Linear Discriminant          ########'
        print '#######################################################'
    
        Xts, Yts, pred = myLD.myld(Xtrain,Xtest,Ytrain,Ytest)
        accuracy, error = getAccuracy(Xts,Yts, pred)
        print('Accuracy: ' + repr(accuracy) + '%')
        print 'Real test data'
        plot_step_lda(Xts,Yts)
        print 'Predicted test data'
        plot_step_lda(Xts,pred)
    
        print '#######################################################'
        print '########        Quadratic Discriminant         ########'
        print '#######################################################'
    
        Xts,Yts, pred = myQD.myqd(Xtrain,Xtest,Ytrain,Ytest)

        accuracy, error = getAccuracy(Xts,Yts, pred)
        print('Accuracy: ' + repr(accuracy) + '%')
        print 'Real test data'
        plot_step_qda(Xts,Yts)
        print 'Predicted test data'
        plot_step_qda(Xts,pred)
    
        print '#######################################################'
        print '########        Support Vector Machine         ########'
        print '#######################################################'
    
        ## Cambio el nombre de la segunda clase
        Ytrain[Ytrain==2] = -1
        Ytest[Ytest==2]= -1

        ## Optimo usando el CrossValidation en el TrainSet
        #X=Xtrain
        #Y=Ytrain
    
        #Kernel='linear_kernel'
        Kernel='gaussian_kernel'
        #Kernel='polynomial_kernel'
        ####Kernel polinomico evaluar el grado
    
        
        Err_PK_C=[]
        PK=[0.1,0.2,0.5,1.0,1.5,2]
        PC=[0.1,0.2,0.5,1.0,1.5,2]
        for i in range(0,len(PK),1):
            Err_C = np.array([])
            for j in range(0,len(PC),1):
                Err_val = np.array([])
                for train, test in kf_total:
                    A, SV_Y, SV, W, B = mySVM.svmfit(X[train], Y[train],Kernel,PK[i], PC[j])  
                    ## Test Classifier Over KFold
                    if Kernel == 'linear_kernel':
                        hat = mySVM.predict(X[test], Kernel, A, SV_Y, SV, W, B, PK = False)
                        correct = np.sum(hat == Y[test])
                        #print "%d out of %d predictions correct" % (correct, len(hat))
                        accuracy, error = getAccuracy(X[test],Y[test], hat)
                        #print('Accuracy: ' + repr(accuracy) + '%')
                    if Kernel == 'gaussian_kernel':
                        hat = mySVM.predict(X[test], Kernel, A, SV_Y, SV, W, B, PK[i])
                        correct = np.sum(hat == Y[test])
                        #print "%d out of %d predictions correct" % (correct, len(hat))
                        accuracy, error = getAccuracy(X[test],Y[test], hat)
                        #print('Accuracy: ' + repr(accuracy) + '%')
                    if Kernel == 'polynomial_kernel':
                        hat = mySVM.predict(X[test], Kernel, A, SV_Y, SV, W, B, PK[i])
                        correct = np.sum(hat == Y[test])
                        #print "%d out of %d predictions correct" % (correct, len(hat))
                        accuracy, error = getAccuracy(X[test],Y[test], hat)
                        #print('Accuracy: ' + repr(accuracy) + '%')
                    Err_val = np.append(Err_val,error)
                meanEval=np.mean(Err_val,axis=0)
                Err_C = np.append(Err_C,meanEval)
            Err_PK_C.append(Err_C)
        PKC = np.vstack(Err_PK_C)
    
        ##Find Optimal PK and C
        minid = np.where(PKC == PKC.min())
        PKopt=PK[minid[0][0]]
        PCopt=PC[minid[1][0]]
    
        A, SV_Y, SV, W, B = mySVM.svmfit(Xtrain, Ytrain,Kernel,PKopt, PCopt)
        
        Yhat = mySVM.predict(Xtest, Kernel, A, SV_Y, SV, W, B,PKopt)
        correct = np.sum(Yhat == Ytest)
        accuracy, error = getAccuracy(Xtest,Ytest, Yhat)
        print('Accuracy: ' + repr(accuracy) + '%')
        
        
        ### With Toolbox
        # we create an instance of SVM and fit out data. We do not scale our
        # data since we want to plot the support vectors
        C = 1.0  # SVM regularization parameter
        svc = svm.SVC(kernel='linear', C=C).fit(Xtrain, Ytrain)
        rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(Xtrain, Ytrain)
        poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(Xtrain, Ytrain)
        lin_svc = svm.LinearSVC(C=C).fit(Xtrain, Ytrain)
        
        for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):

            Yhat = clf.predict(Xtest)
            accuracy, error = getAccuracy(Xtest,Ytest, Yhat)
            print('Accuracy: ' + repr(accuracy) + '%')
        
        ## Cambio el nombre de la segunda clase
        Ytrain[Ytrain== -1] = 2
        Ytest[Ytest== -1]= 2
    
main()

# <codecell>


# <codecell>


