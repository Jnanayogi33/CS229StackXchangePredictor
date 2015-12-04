import numpy as np
import collections
from random import shuffle
from scipy import sparse
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score


def createConfusionMatrix(Y, Y_hat):
    matrix = collections.Counter()
    for i in range(len(Y_hat)): matrix[(Y[i], Y_hat[i])] += 1
    return matrix

def accuracy(matrix):
   numerator = 0.0
   denominator = 0.0
   for key in matrix.keys():
       if key[0] == key[1]: numerator += float(matrix[key])
       denominator += float(matrix[key])
   return numerator/denominator

def precision(matrix):
    precVals = []
    numCategories = max(x[0] for x in matrix.keys())+1
    for i in range(numCategories):
        numerator = float(matrix[(i,i)])
        denominator = float(sum(matrix[(j,i)] for j in range(numCategories)))
        if numerator == 0.0: precVals += [0.0]
        else: precVals += [numerator/denominator]
    return precVals

def recall(matrix):
    recallVals = []
    numCategories = max(x[0] for x in matrix.keys())+1
    for i in range(numCategories):
        numerator = float(matrix[(i,i)])
        denominator = float(sum(matrix[(i,j)] for j in range(numCategories)))
        if numerator == 0.0: recallVals += [0.0]
        else: recallVals += [numerator/denominator]
    return recallVals

def listAverage(rawlist):
    print(rawlist)
    return [sum([x[i] for x in rawlist])/len(rawlist) for i in range(len(rawlist[0]))]

def jointShuffle(X, Y):
    index = range(len(Y))
    shuffle(index)
    X = sparse.csr_matrix(X.toarray()[index])
    Y = np.array(Y)[index].tolist()
    return X, Y

def nFoldValidation(X,Y,folds,model):
    X, Y = jointShuffle(X, Y)

    trainAccuracies = 0.0
    testAccuracies = 0.0
    trainPrecision = np.array([0.0,0.0,0.0])
    testPrecision = np.array([0.0,0.0,0.0])
    trainRecall = np.array([0.0,0.0,0.0])
    testRecall = np.array([0.0,0.0,0.0])

    for i in range(folds):

        lowSplit = i*len(Y)/folds
        highSplit = (i+1)*len(Y)/folds

        trainX = X[range(0,lowSplit) + range(highSplit,X.shape[0])]
        testX = X[lowSplit:highSplit]
        trainY = Y[0:lowSplit] + Y[highSplit:]
        testY = Y[lowSplit:highSplit]

        model.fit(trainX, trainY)

        trainY_hat = model.predict(trainX)
        testY_hat = model.predict(testX)

        trainAccuracies += accuracy_score(trainY, trainY_hat)
        testAccuracies += accuracy_score(testY, testY_hat)
        trainPrecision += precision_score(trainY, trainY_hat, average=None)
        testPrecision += precision_score(testY, testY_hat, average=None)
        trainRecall += recall_score(trainY, trainY_hat, average=None)
        testRecall += recall_score(testY, testY_hat, average=None)

    return trainAccuracies/folds, testAccuracies/folds, trainPrecision/folds, testPrecision/folds, trainRecall/folds, testRecall/folds

