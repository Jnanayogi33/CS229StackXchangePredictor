import postParseUtils as PU
import featureExtractors as FE
import learningModels as LM
import targetCreation as TC
import numpy as np
from scipy import sparse


###################################################################################################

print "Step 1: Get ready: Set variables, obtain post data from xml and split into types"
nfolds = 5
cache = './stackExchangeData/'
file = './stackExchangeData/Apple/Posts.xml'
posts = PU.extractPosts(file)
# questionPosts = [post for post in posts if post['PostTypeId'] == '1']
# answerPosts = [post for post in posts if post['PostTypeId'] == '2']


### For testing only! ###
questionPosts = [posts[0], posts[2]]
answers1 = [post for post in posts if post['PostTypeId'] == '2' and post['ParentId'] == '1']
answers2 = [post for post in posts if post['PostTypeId'] == '2' and post['ParentId'] == '3']
answerPosts = answers1 + answers2


###################################################################################################

print "Step 2: Generate features and save as you go (since very compute intensive):"

# Get topic tag-based features: log of total number, binary tag existence by type
X_topicSum, X_topicBinary = FE.getTopicTagFeatures(questionPosts, answerPosts)
PU.saveSparseCSR(X_topicSum, cache + 'X_topicSum.npz')
PU.saveSparseCSR(X_topicBinary, cache + 'X_topicBinary.npz')


# Get XML tag-based features: log of total number, binary tag existence by type, count number of tags by type
X_xmlSum, X_xmlBinary, X_xmlCounts = FE.getXMLTagFeatures(answerPosts)
PU.saveSparseCSR(X_xmlSum, cache + 'X_xmlSum.npz')
PU.saveSparseCSR(X_xmlBinary, cache + 'X_xmlBinary.npz')
PU.saveSparseCSR(X_xmlCounts, cache + 'X_xmlCount.npz')


# Get raw text meta features utilizing spacy toolkit: 
#  - qMeta: question log num chars, words, sentences, sum of question words, binary existence of question words,
#  - aMeta: answer log num chars, words, sentences
#  - qaSim: word2vec cosine similarity with question
#  - qVect: full word2vec average vector for question (300 features)
#  - aVect: full word2vec average vector for answer (300 features)
X_qMeta, X_aMeta, X_qaWord2VecSim, X_qVect, X_aVect = FE.getMetaFeatures(questionPosts, answerPosts)
PU.saveSparseCSR(X_qMeta, cache + 'X_qMeta.npz')
PU.saveSparseCSR(X_aMeta, cache + 'X_aMeta.npz')
PU.saveSparseCSR(X_qaWord2VecSim, cache + 'X_qaWord2VecSim.npz')
PU.saveSparseCSR(X_qVect, cache + 'X_qVect.npz')
PU.saveSparseCSR(X_aVect, cache + 'X_aVect.npz')


# Get raw text word-specific features utilizing sklearn toolkit:
#  - answer unigram, bigram words: binary existence, counts, tfidf frequency
#  - tfidf cosine similarity between question and answer
X_wordBinary, X_wordCounts, X_wordTfidf, X_qaTfidfSim = FE.getWordFeatures(questionPosts, answerPosts)
PU.saveSparseCSR(X_wordBinary, cache + 'X_wordBinary.npz')
PU.saveSparseCSR(X_wordCounts, cache + 'X_wordCounts.npz')
PU.saveSparseCSR(X_wordTfidf, cache + 'X_wordTfidf.npz')
PU.saveSparseCSR(X_qaTfidfSim, cache + 'X_qaTfidfSim.npz')


# Certain features are in raw count form. Here we create their logged versions where useful
X_xmlCountsLog = FE.sparseLog(X_xmlCounts)
X_wordCountsLog = FE.sparseLog(X_wordCounts)


# For subset of features where relevant, combine into X vector set with only interaction terms
#  - returns X features with only first order interaction terms of the form x_1 * x_2
#  - remember number of features scales exponentially in degree, so limit this!
X_interactionTopicWord = FE.getInteractionMatrix([X_topicBinary, X_wordTfidf])
X_interactionTopicVector = FE.getInteractionMatrix([X_topicBinary, X_aVect])
X_interactionTopicWordBinary = FE.getInteractionMatrix([X_topicBinary, X_wordBinary])
X_interactionTopicWordCounts = FE.getInteractionMatrix([X_topicBinary, X_wordCounts])


###################################################################################################

print "Step 3: Create adjusted data labels (Y)"

adjustedScores = TC.getAdjustedScores(file, answerPosts)
stdev = np.std(adjustedScores)
scoreSplits = [-stdev, stdev]
Y = [FE.rightPlaceInList(score, scoreSplits) for score in adjustedScores]
PU.savePosts(Y, cache + 'Y')
# Y = PU.loadPosts(cache + 'Y')


###################################################################################################

print "Step 4: Run n-folds validation and output results with variety of different settings, features"


# Start with simplest classifier and small subset of features
#  - Only binary features provided based on bernoulli's requirements
#  - To prevent it only voting for the most common class, assume class priors are uniform
from sklearn.naive_bayes import BernoulliNB
X = sparse.hstack([X_topicBinary, X_xmlBinary, X_wordBinary, X_interactionTopicWordBinary])
print LM.nFoldValidation(X,Y,nfolds,BernoulliNB(alpha=1.0, fit_prior=False))
# print LM.nFoldValidation(X,Y,nfolds,BernoulliNB(alpha=2.0, fit_prior=False))
# print LM.nFoldValidation(X,Y,nfolds,BernoulliNB(alpha=5.0, fit_prior=False))
# print LM.nFoldValidation(X,Y,nfolds,BernoulliNB(alpha=10.0, fit_prior=False))


# Next simple classifier and small subset of features
#  - Only count features per Multinomial Naive Bayes requirements
#  - To prevent it only voting for the most common class, assume class priors are uniform
from sklearn.naive_bayes import MultinomialNB
X = sparse.hstack([X_xmlCounts, X_wordCounts, X_interactionTopicWordCounts])
print LM.nFoldValidation(X,Y,nfolds,MultinomialNB(alpha=1.0, fit_prior=False))
# print LM.nFoldValidation(X,Y,nfolds,MultinomialNB(alpha=2.0, fit_prior=False))
# print LM.nFoldValidation(X,Y,nfolds,MultinomialNB(alpha=5.0, fit_prior=False))
# print LM.nFoldValidation(X,Y,nfolds,MultinomialNB(alpha=10.0, fit_prior=False))


# Logistic regression with regularization known to be good at text classifcation according to Andrew!
#  - Advantage because does not make assumptions on feature distribution, so provide more diverse features
#  - Balance classes evenly so voter won't always vote for most common class
from sklearn.linear_model import LogisticRegression
X = sparse.hstack([X_interactionTopicWord, X_wordCountsLog, X_xmlCountsLog, X_xmlSum, X_topicSum, X_qaWord2VecSim, X_qaTfidfSim, X_aMeta, X_qMeta])
print LM.nFoldValidation(X,Y,nfolds,LogisticRegression(C=1.0, class_weight='balanced'))
# print LM.nFoldValidation(X,Y,nfolds,LogisticRegression(C=0.1, class_weight='balanced'))
# print LM.nFoldValidation(X,Y,nfolds,LogisticRegression(C=0.01, class_weight='balanced'))
# print LM.nFoldValidation(X,Y,nfolds,LogisticRegression(C=0.001, class_weight='balanced'))


# SVM classifier
#  - Typically requires features that vary within the same range (not linear invariant)
#  - Balance classes evenly so voter won't always vote for most common class
X = sparse.hstack([X_wordTfidf, X_aVect, X_interactionTopicWord, X_interactionTopicVector, X_qaWord2VecSim, X_qaTfidfSim])
from sklearn import svm
print LM.nFoldValidation(X,Y,nfolds,svm.LinearSVC(C=1.0, class_weight='balanced'))
# print LM.nFoldValidation(X,Y,nfolds,svm.LinearSVC(C=0.1, class_weight='balanced'))
# print LM.nFoldValidation(X,Y,nfolds,svm.LinearSVC(C=0.01, class_weight='balanced'))
# print LM.nFoldValidation(X,Y,nfolds,svm.LinearSVC(C=0.001, class_weight='balanced'))