# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%matplotlib inline

# -*- coding: utf-8 -*-
## Based in kNN example by Dr. Jason Brownlee
## Editor and Chief at MachineLearningMastery.com.

import random
import math
import operator
import numpy as np
import scipy.io as spio
import pylab as pl
import seaborn as sns


## Load data and split into training and test sets (67/33)
def loadDataset2(filename2, split):
	mat = spio.loadmat(filename2,squeeze_me=True)
	datos = mat['datos'].tolist()  
	X=datos[0]
	Y=datos[1]
	# add a column to A, at the end:
	A=np.insert(X, X.shape[1], Y, axis=1)
	#trainingSet=A[np.random.randint(A.shape[0],size=2),:]
	idx = range(0, A.shape[0])
	idx_train = np.random.choice(idx, size=split, replace=False)
	idx_test = np.setdiff1d(idx, idx_train)
	trainingSet = A[idx_train, :]
	testSet = A[idx_test, :]
	return (trainingSet,testSet)

## Measure euclidean distance between
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

## kNN
def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

## Vote the class and take majority
def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

## How well was predicted
def getAccuracy(testSet, predictions):
	correct = 0
	incorrect = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
		else:
			incorrect +=1
	accur=(correct/float(len(testSet))) * 100.0
	error=(incorrect/float(len(testSet))) * 100.0
	return (accur,error)

def main():
	# prepare data
	filename2='/home/bgx/trabajofinaldeprctica/datosOS14.mat'
	Nsamples = 100
	runs=range(10)
	print("Runs: %d, Ntrain: %d" % (len(runs),Nsamples))


	#kNN Implementation
	param=range(2, 11, 1) ## k nearest 
	Err = np.zeros(len(param)) ## Error array
	pl.ion()
	pl.show()
	for i in runs:
		trainingSet, testSet = loadDataset2(filename2, Nsamples)
		#kNN Implementation
		resultsX=[]
		resultsY=[]
		for j in param:
			# generate predictions
			predictions=[]
			for x in range(testSet.shape[0]):
				neighbors = getNeighbors(trainingSet, testSet[x], j)
				result = getResponse(neighbors)
				predictions.append(result)
			accuracy,error = getAccuracy(testSet, predictions)
			#print('K: '+repr(kvector[j]) +'  Accuracy: ' + repr(accuracy) + '%  Error: ' + repr(error) + '%')
			#Err[j]=Err[j]+float(error)
			resultsX.append(j)
			resultsY.append(accuracy)
		pl.plot(resultsX,resultsY)
		pl.title("Accuracy with Increasing K")
		pl.draw()
#	print "Errors: k=2 k=5 k=10"
#	print np.divide(Err, len(runs))

main()

# <codecell>


# <codecell>


