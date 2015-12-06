from collections import Counter
from sklearn.feature_extraction import DictVectorizer
import postParseUtils as PU
import numpy as np
import math, bs4, os, scipy
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import TruncatedSVD
from scipy import sparse
from spacy.en import English, LOCAL_DATA_DIR


# Get data ready
data_dir = os.environ.get('SPACY_DATA', LOCAL_DATA_DIR)
nlp = English(data_dir=data_dir)
questionWords = ['who', 'what', 'when', 'where', 'how', 'why', 'which']


# General helper for extracting all tags from a piece of text and creating one list of tags per post
def getTags(soup):
    tags = []
    for child in soup.children:
        if type(child) is bs4.element.Tag:
            tags += ['TAG' + child.name]
            tags += getTags(child)
    return tags


# helper takes matrix (sparse or non-sparse) and returns log in sparse form. 
#  - Add 1 to smooth out zeroes. Assume no negative inputs
#  - If receive sparse matrix assume super large so switch to COO matrix constructor
def sparseLog(matrix):
	if type(matrix) == scipy.sparse.csr.csr_matrix:
		coo = sparse.coo_matrix(matrix)
		newData = np.log(coo.data + 1.0)
		coo = sparse.coo_matrix((newData, (coo.row, coo.col)), shape=coo.shape)
		return sparse.csr_matrix(coo)
	elif type(matrix) == np.ndarray or type(matrix) == np.matrixlib.defmatrix.matrix:
		return sparse.csr_matrix(np.log(matrix + 1.0))
	elif type(matrix) == list:
		return sparse.csr_matrix(np.log(np.array(matrix) + 1.0))
	else: return None


# Get topic tag-based features: log of total number, binary tag existence by type
def getTopicTagFeatures(questionPosts, answerPosts):
	tagsByQuestion = {}
	for qPost in questionPosts:
		curr = BeautifulSoup(qPost['Tags'])
		tagsByQuestion[qPost['Id']] = " ".join(getTags(curr))
	answerTags = []
	for aPost in answerPosts:
		answerTags += [tagsByQuestion[aPost['ParentId']]]
	X_binary = CountVectorizer(binary=True).fit_transform(answerTags)
	X_sums = sparseLog(X_binary.sum(axis=1))
	return X_sums, X_binary


# Get XML tag-based features: log total number, binary tag existence by type, number of tags by type
def getXMLTagFeatures(posts):
	tagLists = []
	for i in range(len(posts)):
	    curr = BeautifulSoup(posts[i]['Body'])
	    tagLists += [" ".join(getTags(curr))]
	X_binary = CountVectorizer(binary=True).fit_transform(tagLists)
	X_counts = CountVectorizer().fit_transform(tagLists)
	X_sums = sparseLog(X_counts.sum(axis=1))
	return X_sums, X_binary, X_counts


# Get raw text meta features utilizing spacy toolkit: 
#  - qMeta: question log num chars, words, sentences, sum of question words, binary existence of question words,
#  - aMeta: answer log num chars, words, sentences
#  - qaSim: word2vec cosine similarity with question
#  - qVect: full word2vec average vector for question
#  - aVect: full word2vec average vector for answer
def getMetaFeatures(questionPosts, answerPosts):
	dataByQuestion = {}
	for qPost in questionPosts:
		curr = nlp(BeautifulSoup(qPost['Body']).get_text())
		numChars = math.log(len(curr.text) + 1)
		numWords = math.log(len(list(curr)) + 1)
		numSents = math.log(len(list(curr.sents)) + 1)
		qWordSum = math.log(len([tok for tok in curr if tok.lemma_ in questionWords]) + 1)
		qWordData = [qWord in curr.text.lower() for qWord in questionWords]
		dataByQuestion[qPost['Id']] = [numChars, numWords, numSents, qWordSum] + qWordData + [curr]
	qMeta = []
	aMeta = []
	qaSim = []
	qVectors = []
	aVectors = []
	for aPost in answerPosts:
		curr = nlp(BeautifulSoup(aPost['Body']).get_text())
		qData = dataByQuestion[aPost['ParentId']]
		numChars = math.log(len(curr.text) + 1)
		numWords = math.log(len(list(curr)) + 1)
		numSents = math.log(len(list(curr.sents)) + 1)
		qMeta += [qData[:11]]
		aMeta += [[numChars, numWords, numSents]]
		qaSim += [[qData[11].similarity(curr)]]
		qVectors += [list(qData[11].vector)]
		aVectors += [list(curr.vector)]
	return sparse.csr_matrix(qMeta), sparse.csr_matrix(aMeta), sparse.csr_matrix(qaSim), sparse.csr_matrix(qVectors), sparse.csr_matrix(aVectors)


# Get raw text word-specific features utilizing sklearn toolkit:
#  - answer unigram, bigram words: binary existence, counts, tfidf frequency
#  - tfidf cosine similarity between question and answer
def getWordFeatures(questionPosts, answerPosts):
	answerText = [BeautifulSoup(aPost['Body']).get_text() for aPost in answerPosts]
	questionText = [BeautifulSoup(qPost['Body']).get_text() for qPost in questionPosts]
	aTfidfVectorizer = TfidfVectorizer(ngram_range=(1,2))
	aBinary = CountVectorizer(binary=True, ngram_range=(1,2)).fit_transform(answerText)
	aCounts = CountVectorizer(ngram_range=(1,2)).fit_transform(answerText)
	aTfidf = aTfidfVectorizer.fit_transform(answerText)
	qVectorDict = {}
	for i, qPost in enumerate(questionPosts): 
		qVectorDict[qPost['Id']] = aTfidfVectorizer.transform([questionText[i]])
	qaSim = []
	for i, aPost in enumerate(answerPosts): 
		qaSim += [linear_kernel(qVectorDict[aPost['ParentId']], aTfidf[i, :])[0]]
	return aBinary, aCounts, aTfidf, sparse.csr_matrix(qaSim)


# Get a simplified set of word features just for answer vectors
#  - Only include unigrams, bigrams that appear in at least 100 posts out of 90,000 (--> 100x smaller feature vectors)
#  - Lemmatize text before processing it (at risk of changing original meaning)
def getShrunkenWordFeatures(answerPosts):
	answerLemmas = []
	for aPost in answerPosts:
		curr = FE.nlp(FE.BeautifulSoup(aPost['Body']).get_text())
		answerLemmas += [" ".join([tok.lemma_ for tok in curr]).replace(u'\n',u'')]
	minRatio = 100.0/len(answerLemmas)
	aTfidf = TfidfVectorizer(ngram_range=(1,2), min_df=minRatio).fit_transform(answerLemmas)
	aBinary = CountVectorizer(binary=True, ngram_range=(1,2), min_df=minRatio).fit_transform(answerLemmas)
	aCounts = CountVectorizer(ngram_range=(1,2), min_df=minRatio).fit_transform(answerLemmas)
	return aBinary, aCounts, aTfidf


# Reduce dimension of matrix using PCA:
#  - assume given sparse matrix and will return sparse matrix
def reduceDimensions(matrix, dim=100):
	svd=TruncatedSVD(n_components=dim)
	result = sparse.csr_matrix(svd.fit_transform(matrix))
	print 'variance explained:', np.sum(svd.explained_variance_ratio_)
	return result


# Helper: get first order interaction terms between two sparse matrices only, but they must be small!
def getInteractions2Matrices(matA, matB):
	matCombined = []
	matA = matA.toarray()
	matB = matB.toarray()
	for i in range(matA.shape[1]):
		print 'Interaction matrix progress:', i, '/', matA.shape[1]
		for j in range(matB.shape[1]):
			curr = np.prod((matA[:,i], matB[:,j]), axis=0)
			curr = sparse.csr_matrix(curr)
			if j == 0: combined = curr
			else: combined = sparse.vstack([combined, curr])
		matCombined += [combined]
	matCombined = sparse.vstack(matCombined)
	return matCombined.transpose()


# Get first degree interaction terms between matrices in a list of matrices
#  - Each term is of the form x_1 * x_2 where x_1 is from one matrix, x_2 is from a different one
#  - Assume incoming matrices sparse, outgoing ones also sparse (csr), or else would break!
def getInteractionMatrix(xList):
	matCombined = None
	for i in range(len(xList)):
		for j in range(i+1,len(xList)):
			if matCombined == None: matCombined = getInteractions2Matrices(xList[i], xList[j])
			else: matCombined = sparse.hstack([matCombined, getInteractions2Matrices(xList[i], xList[j])]) 
	return matCombined


# Helper function places candidate in right part of list based on given threshold values
def rightPlaceInList(candidate, list):
    for i in range(len(list)):
        if candidate < list[i]: return i
    return len(list)


# Helper function that gives a weight to each sample to ensure balanced distribution
def getSampleWeights(Y):
	numSamples = float(len(Y))
	numClasses = len(list(set(Y)))
	weights = [numSamples/(np.bincount(Y)[i]) for i in range(numClasses)]
	return [weights[y] for y in Y]