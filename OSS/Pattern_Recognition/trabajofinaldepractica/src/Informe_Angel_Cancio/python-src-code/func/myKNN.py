# -*- coding: utf-8 -*-
# K-nearest neighbors module

import numpy as np
import math
import operator

def kNNClass(X_train,X_test,Y_train,Y_test,k):
    trainSet=np.c_[ X_train, Y_train ]
    testSet=np.c_[ X_test, Y_test ]
    # generate predictions
    Predictions=[]
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainSet, testSet[x], k)
        result = getResponse(neighbors)
        Predictions.append(result)
    return Predictions

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

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

#if __name__ == "__main__":
#    import sys
#    kNNClass(int(sys.argv[1]))
