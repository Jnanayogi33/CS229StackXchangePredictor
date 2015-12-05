import postParseUtils as PU
import featureExtractors as FE
import learningModels as LM
import targetCreation as TC
import numpy as np
from scipy import sparse
import os.path


###################################################################################################

print "Step 1: Get ready: Set variables, obtain post data from xml and split into types"
nfolds = 5
cache = './stackExchangeData/'
file = './stackExchangeData/Apple/Posts.xml'
posts = PU.extractPosts(file)
questionPosts = [post for post in posts if post['PostTypeId'] == '1']
answerPosts = [post for post in posts if post['PostTypeId'] == '2']


### For testing! Takes subset of questions as all their children posts ###
# questionPosts = [post for post in posts if post['PostTypeId'] == '1' and int(post['Id']) < 50]
# parentIDs = [post['Id'] for post in questionPosts]
# answerPosts = [post for post in posts if post['PostTypeId'] == '2' and post['ParentId'] in parentIDs]


###################################################################################################

print "Step 2: Generate features and save as you go (since very compute intensive):"

# Get topic tag-based features: log of total number, binary tag existence by type
if os.path.isfile(cache + 'X_topicSum.npz') and os.path.isfile(cache + 'X_topicBinary.npz'):
	X_topicSum = PU.loadSparseCSR(cache + 'X_topicSum.npz')
	X_topicBinary = PU.loadSparseCSR(cache + 'X_topicBinary.npz')
else:
	X_topicSum, X_topicBinary = FE.getTopicTagFeatures(questionPosts, answerPosts)
	PU.saveSparseCSR(X_topicSum, cache + 'X_topicSum.npz')
	PU.saveSparseCSR(X_topicBinary, cache + 'X_topicBinary.npz')



# Get XML tag-based features: log of total number, binary tag existence by type, count number of tags by type
if os.path.isfile(cache + 'X_xmlSum.npz') and os.path.isfile(cache + 'X_xmlBinary.npz') and os.path.isfile(cache + 'X_xmlCounts.npz'):
	X_xmlSum = PU.loadSparseCSR(cache + 'X_xmlSum.npz')
	X_xmlBinary = PU.loadSparseCSR(cache + 'X_xmlBinary.npz')
	X_xmlCounts = PU.loadSparseCSR(cache + 'X_xmlCounts.npz')
else:
	X_xmlSum, X_xmlBinary, X_xmlCounts = FE.getXMLTagFeatures(answerPosts)
	PU.saveSparseCSR(X_xmlSum, cache + 'X_xmlSum.npz')
	PU.saveSparseCSR(X_xmlBinary, cache + 'X_xmlBinary.npz')
	PU.saveSparseCSR(X_xmlCounts, cache + 'X_xmlCounts.npz')


# Get raw text meta features utilizing spacy toolkit: 
#  - qMeta: question log num chars, words, sentences, sum of question words, binary existence of question words,
#  - aMeta: answer log num chars, words, sentences
#  - qaSim: word2vec cosine similarity with question
#  - qVect: full word2vec average vector for question (300 features)
#  - aVect: full word2vec average vector for answer (300 features)
if os.path.isfile(cache + 'X_qMeta.npz') and os.path.isfile(cache + 'X_aMeta.npz') and os.path.isfile(cache + 'X_qaWord2VecSim.npz') \
	and os.path.isfile(cache + 'X_qVect.npz') and os.path.isfile(cache + 'X_aVect.npz'):
	X_qMeta = PU.loadSparseCSR(cache + 'X_qMeta.npz')
	X_aMeta = PU.loadSparseCSR(cache + 'X_aMeta.npz')
	X_qaWord2VecSim = PU.loadSparseCSR(cache + 'X_qaWord2VecSim.npz')
	X_qVect = PU.loadSparseCSR(cache + 'X_qVect.npz')
	X_aVect = PU.loadSparseCSR(cache + 'X_aVect.npz')
else:
	X_qMeta, X_aMeta, X_qaWord2VecSim, X_qVect, X_aVect = FE.getMetaFeatures(questionPosts, answerPosts)
	PU.saveSparseCSR(X_qMeta, cache + 'X_qMeta.npz')
	PU.saveSparseCSR(X_aMeta, cache + 'X_aMeta.npz')
	PU.saveSparseCSR(X_qaWord2VecSim, cache + 'X_qaWord2VecSim.npz')
	PU.saveSparseCSR(X_qVect, cache + 'X_qVect.npz')
	PU.saveSparseCSR(X_aVect, cache + 'X_aVect.npz')


# Get raw text word-specific features utilizing sklearn toolkit:
#  - answer unigram, bigram words: binary existence, counts, tfidf frequency
#  - tfidf cosine similarity between question and answer
if os.path.isfile(cache + 'X_wordBinary.npz') and os.path.isfile(cache + 'X_wordCounts.npz') and \
	os.path.isfile(cache + 'X_wordTfidf.npz') and os.path.isfile(cache + 'X_qaTfidfSim.npz'):
	X_wordBinary = PU.loadSparseCSR(cache + 'X_wordBinary.npz')
	X_wordCounts = PU.loadSparseCSR(cache + 'X_wordCounts.npz')
	X_wordTfidf = PU.loadSparseCSR(cache + 'X_wordTfidf.npz')
	X_qaTfidfSim = PU.loadSparseCSR(cache + 'X_qaTfidfSim.npz')
else:
	X_wordBinary, X_wordCounts, X_wordTfidf, X_qaTfidfSim = FE.getWordFeatures(questionPosts, answerPosts)
	PU.saveSparseCSR(X_wordBinary, cache + 'X_wordBinary.npz')
	PU.saveSparseCSR(X_wordCounts, cache + 'X_wordCounts.npz')
	PU.saveSparseCSR(X_wordTfidf, cache + 'X_wordTfidf.npz')
	PU.saveSparseCSR(X_qaTfidfSim, cache + 'X_qaTfidfSim.npz')


# Certain features are in raw count form. Here we create their logged versions where useful
X_xmlCountsLog = FE.sparseLog(X_xmlCounts)
X_wordCountsLog = FE.sparseLog(X_wordCounts)


# Reduce dimensionality of certain feature sets to make creating interaction terms more tractable/computable
#  - Use Truncated singular value decomposition (similar to PCA but works on sparse matrices)
if os.path.isfile(cache + 'X_wordTfidfSVD300.npz'): X_wordTfidfSVD300 = PU.loadSparseCSR(cache + 'X_wordTfidfSVD300.npz')
else: 
	X_wordTfidfSVD300 = FE.reduceDimensions(X_wordTfidf, dim=300)
	PU.saveSparseCSR(X_wordTfidfSVD300, cache + 'X_wordTfidfSVD300.npz')
X_qTypeSVD5 = FE.reduceDimensions(sparse.hstack([X_topicBinary, X_qVect, X_qMeta]), dim=5)


# For subset of features where relevant, combine into X vector set with only interaction terms
#  - returns X features with only first order interaction terms of the form x*y where x comes from matrix X and y comes from different matrix Y
#  - WARNING: takes too long whenever try to create interaction terms for bigger sparse vectors like X_wordCounts (and takes too much memory)
#  - Currently addressing by only creating interaction terms for dimension-reduced matrices
if os.path.isfile(cache + 'X_interactionQtypeWordTfidf.npz'): X_interactionQtypeWordTfidf = PU.loadSparseCSR(cache + 'X_interactionQtypeWordTfidf.npz')
else: 
	X_interactionQtypeWordTfidf = FE.getInteractionMatrix([X_qTypeSVD5, X_wordTfidfSVD300])
	PU.saveSparseCSR(X_interactionQtypeWordTfidf, cache + 'X_interactionQtypeWordTfidf.npz')


###################################################################################################

print "Step 3: Create adjusted data labels (Y)"

if os.path.isfile(cache + 'Y'): Y = PU.loadPosts(cache + 'Y')
else: 
	adjustedScores = TC.getAdjustedScores(file, answerPosts)
	stdev = np.std(adjustedScores)
	scoreSplits = [-stdev, stdev]
	Y = [FE.rightPlaceInList(score, scoreSplits) for score in adjustedScores]
	PU.savePosts(Y, cache + 'Y')


###################################################################################################

print "Step 4: Run n-folds validation and output results with variety of different settings, features"


# Start with simplest classifier and small subset of features
#  - Only binary features provided based on bernoulli's requirements
#  - To prevent it only voting for the most common class, assume class priors are uniform
from sklearn.naive_bayes import BernoulliNB
X = sparse.hstack([X_wordBinary, X_xmlBinary])
LM.nFoldValidation(X,Y,nfolds,BernoulliNB(alpha=1.0), "Bernoulli NB, alpha = 1.0, binary features only")
LM.nFoldValidation(X,Y,nfolds,BernoulliNB(alpha=2.0), "Bernoulli NB, alpha = 2.0, binary features only")
LM.nFoldValidation(X,Y,nfolds,BernoulliNB(alpha=5.0), "Bernoulli NB, alpha = 5.0, binary features only")
LM.nFoldValidation(X,Y,nfolds,BernoulliNB(alpha=10.0), "Bernoulli NB, alpha = 10.0, binary features only")


# Next simple classifier and small subset of features
#  - Only count features per Multinomial Naive Bayes requirements
#  - To prevent it only voting for the most common class, assume class priors are uniform
from sklearn.naive_bayes import MultinomialNB
X = sparse.hstack([X_xmlCounts, X_wordCounts])
LM.nFoldValidation(X,Y,nfolds,MultinomialNB(alpha=1.0), "Multinomial NB, alpha = 1.0, count features only")
LM.nFoldValidation(X,Y,nfolds,MultinomialNB(alpha=2.0), "Multinomial NB, alpha = 2.0, count features only")
LM.nFoldValidation(X,Y,nfolds,MultinomialNB(alpha=5.0), "Multinomial NB, alpha = 5.0, count features only")
LM.nFoldValidation(X,Y,nfolds,MultinomialNB(alpha=10.0), "Multinomial NB, alpha = 10.0, count features only")


# Logistic regression with regularization known to be good at text classifcation according to Andrew!
#  - Advantage because does not make assumptions on feature distribution, so provide more diverse, nonlinear features
#  - Balance classes evenly so voter won't always vote for most common class
from sklearn.linear_model import LogisticRegression
X = sparse.hstack([X_interactionQtypeWordTfidf, X_qaWord2VecSim, X_qaTfidfSim, X_aMeta, X_qMeta])
LM.nFoldValidation(X,Y,nfolds,LogisticRegression(C=1.0, class_weight='balanced'), "LogReg, C = 1.0, mix of mostly nonlinear features")
LM.nFoldValidation(X,Y,nfolds,LogisticRegression(C=0.1, class_weight='balanced'), "LogReg, C = 0.1, mix of mostly nonlinear features")
LM.nFoldValidation(X,Y,nfolds,LogisticRegression(C=0.01, class_weight='balanced'), "LogReg, C = 0.01, mix of mostly nonlinear features")
LM.nFoldValidation(X,Y,nfolds,LogisticRegression(C=0.001, class_weight='balanced'), "LogReg, C = 0.001, mix of mostly nonlinear features")


# SVM classifier
#  - Typically requires features that vary within the same range (not linear invariant)
#  - Balance classes evenly so voter won't always vote for most common class
from sklearn import svm
X = sparse.hstack([X_wordTfidf, X_aVect, X_qaWord2VecSim, X_qaTfidfSim])
LM.nFoldValidation(X,Y,nfolds,svm.LinearSVC(C=1.0, class_weight='balanced'), "Linear SVM classifier, C = 1.0, mix of mostly nonlinear features < 1.0")
LM.nFoldValidation(X,Y,nfolds,svm.LinearSVC(C=0.1, class_weight='balanced'), "Linear SVM classifier, C = 0.1, mix of mostly nonlinear features < 1.0")
LM.nFoldValidation(X,Y,nfolds,svm.LinearSVC(C=0.01, class_weight='balanced'), "Linear SVM classifier, C = 0.01, mix of mostly nonlinear features < 1.0")
LM.nFoldValidation(X,Y,nfolds,svm.LinearSVC(C=0.001, class_weight='balanced'), "Linear SVM classifier, C = 0.001, mix of mostly nonlinear features < 1.0")