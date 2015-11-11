import postParseUtils as PU
import postTokenizerUtils as TU
import featureExtractors as FE
import learningModels as LM
import targetCreation as TC
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
import numpy as np

#General control variables
nfolds = 5
file = './stackExchangeData/Apple/Posts.xml'

# print "Extract post data from xml"
# posts = PU.extractPosts(file)
# PU.printBasicStats(posts)
#
# print "Parse, tokenize posts"
# posts = TU.stripHTML(posts)
# posts = TU.stripPunctuation(posts) #except apostrophe
# posts = TU.toLower(posts)
# posts = TU.tokenize(posts)
# # posts = TU.stemTokens(posts)
#
# print "Save posts"
# PU.savePosts(posts, './stackExchangeData/Apple/posts.pk1')
#
# print "Load pre-parsed posts"
# posts = PU.loadPosts('./stackExchangeData/Apple/posts.pk1')
#
#
# print "Generate features (X): unigram counts, image existence dummy variable, Q&A pair cosine similarity"
# X = FE.tokenCountByAnswer(posts)
# X = FE.addToListofDicts(X, FE.includeImage(posts), "##IMAGE##")
# # X = FE.addToListofDicts(X, FE.cosScore(posts), "##COSINESIM##")
# X = FE.vectorizeCounts(X)
# PU.savePosts(X, 'currentXvector')
#
#
# print "Create adjusted data labels (Y)"
adjustedScores = TC.getAdjustedScores(file)
stdev = np.std(adjustedScores)
scoreSplits = [-stdev, stdev]
Y = [FE.rightPlaceInList(score, scoreSplits) for score in adjustedScores]
PU.savePosts(Y,'currentYvector')

X = PU.loadPosts('currentXvector')
Y = PU.loadPosts('currentYvector')

print "Run n-folds validation and output resulting scores"
print LM.nFoldValidation(X,Y,nfolds,MultinomialNB(alpha=1.0))
print LM.nFoldValidation(X,Y,nfolds,MultinomialNB(alpha=2.0))
print LM.nFoldValidation(X,Y,nfolds,MultinomialNB(alpha=5.0))
print LM.nFoldValidation(X,Y,nfolds,MultinomialNB(alpha=10.0))
print LM.nFoldValidation(X,Y,nfolds,svm.LinearSVC(C=1.0))
print LM.nFoldValidation(X,Y,nfolds,svm.LinearSVC(C=0.01))
print LM.nFoldValidation(X,Y,nfolds,svm.LinearSVC(C=0.0001))