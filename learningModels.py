import numpy as np
import featureExtractors as FE
from sklearn.naive_bayes import MultinomialNB

def nFoldValidation(X,Y,folds,model):
    X, Y = FE.jointShuffle(X, Y)
    trainScores = []
    testScores = []
    for i in range(folds):

        lowSplit = i*len(Y)/folds
        highSplit = (i+1)*len(Y)/folds

        trainX = X[range(0,lowSplit) + range(highSplit,X.shape[0])]
        testX = X[lowSplit:highSplit]
        trainY = Y[0:lowSplit] + Y[highSplit:]
        testY = Y[lowSplit:highSplit]

        model.fit(trainX, trainY)
        trainScores += [model.score(trainX, trainY)]
        testScores += [model.score(testX, testY)]

    return trainScores,testScores