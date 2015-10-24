from collections import Counter
from sklearn.feature_extraction import DictVectorizer
import postParseUtils as PU
from random import shuffle
import numpy as np

def tokenCountByAnswer(posts):
    counts = []
    for i in range(len(posts)):
        if posts[i]['PostTypeId'] == PU.QUESTION: continue
        curr = Counter()
        for word in posts[i]['Title']: curr[word] += 1
        for word in posts[i]['Body']: curr[word] += 1
        counts += [curr]
    return counts

def vectorizeCounts(counts):
    v = DictVectorizer(sparse=True)
    X = v.fit_transform(counts)
    return X

def rightPlaceInList(candidate, list):
    for i in range(len(list)):
        if candidate <= list[i]: return i
    return len(list)

def labelAnswerByScoreSplits(posts, splits):
    Y = []
    for i in range(len(posts)):
        if posts[i]['PostTypeId'] == PU.QUESTION: continue
        Y += [rightPlaceInList(posts[i]['Score'], splits)]
    return Y

def jointShuffle(X, Y):
    index = range(len(Y))
    shuffle(index)
    X = X[index]
    Y = np.array(Y)[index].tolist()
    return X, Y