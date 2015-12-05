import numpy as np
import collections
from random import shuffle
from scipy import sparse
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split

def nFoldValidation(X,Y,folds,model,name):
    accum_trainY = []
    accum_trainY_hat = []
    accum_testY = []
    accum_testY_hat = []
    for i in range(folds):
        trainX, testX, trainY, testY = train_test_split(X, Y, test_size=1.0/folds)
        model.fit(trainX, trainY)
        trainY_hat = list(model.predict(trainX))
        testY_hat = list(model.predict(testX))
        accum_trainY += trainY
        accum_trainY_hat += trainY_hat
        accum_testY += testY
        accum_testY_hat += testY_hat
    print "Performance on training set for", name
    print classification_report(accum_trainY, accum_trainY_hat)
    print "Performance on test set for", name
    print classification_report(accum_testY, accum_testY_hat)
