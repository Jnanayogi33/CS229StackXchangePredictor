import numpy as np
import featureExtractors as FE
from sklearn.naive_bayes import MultinomialNB
import collections

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
    numCategories = max(x[0] for x in matrix.keys())
    for i in range(numCategories + 1):
        numerator = float(matrix[(i,i)])
        denominator = float(sum(matrix[(j,i)] for j in range(numCategories)))
        precVals += [numerator/denominator]
    return precVals

def recall(matrix):
    recallVals = []
    numCategories = max(x[0] for x in matrix.keys())
    for i in range(numCategories + 1):
        numerator = float(matrix[(i,i)])
        denominator = float(sum(matrix[(i,j)] for j in range(numCategories)))
        recallVals += [numerator/denominator]
    return recallVals

def nFoldValidation(X,Y,folds,model):
    X, Y = FE.jointShuffle(X, Y)
    trainScores = collections.Counter()
    testScores = collections.Counter()
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

        trainConfusion = createConfusionMatrix(trainY, trainY_hat)
        testConfusion = createConfusionMatrix(testY, testY_hat)

        for key in trainConfusion.keys(): trainScores[key] += trainConfusion[key]
        for key in testConfusion.keys(): testScores[key] += testConfusion[key]

    return accuracy(testScores), precision(testScores), recall(testScores)