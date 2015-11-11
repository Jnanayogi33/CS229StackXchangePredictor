from collections import Counter
from sklearn.feature_extraction import DictVectorizer
import postParseUtils as PU
from random import shuffle
import numpy as np

def tokenCountByAnswer(posts):
    counts = []
    for i in range(len(posts)):
        if posts[i]['PostTypeId'] == PU.ANSWER:
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

def addToListofDicts(listofdicts, list, label):
	print len(listofdicts), len(list)
	for i in range(len(listofdicts)):
		listofdicts[i][label] = list[i]
	return listofdicts

## includeImage() create a feature to indicate if any image was included
## in the answer. it returns a list of 0/1, which corresponds each answer
## in the dataset 
def includeImage(posts):
    output = []
    for i in range(len(posts)):
        if posts[i]['PostTypeId'] == PU.ANSWER:
            output += [1 * ("http://i.stack.imgur.com" in posts[i]['Body'])]
    return output

##** calculate cosine similarity between answer and corresponding question
## idf creates idf for each word in the answers, and 
## returns a word dictionary with value idf
def idf(posts):
	import math
	ct_answer = 0
	df_dict = Counter()
	for i in range(len(posts)):
		if posts[i]['PostTypeId'] == PU.ANSWER:
			ct_answer += 1
			unique_words = set(posts[i]['Body'])
			for word in list(unique_words):
				df_dict[word] += 1
	idf_dict = {}
	for wd in df_dict:
		idf_dict[wd] = math.log(ct_answer/df_dict[wd])
	return idf_dict

## createTFIdf_Q creates a dictionary of tf for each quesiton 
## 
def createTFIdf_Q(posts):
	idf_dict = idf(posts)
	tfIdf_dict = {}
	for i in range(len(posts)):
		tf_temp = Counter()
		if posts[i]['PostTypeId'] == PU.QUESTION:
			for i_word in posts[i]['Body']:
				tf_temp[i_word] += 1
			tfidf_temp = {}
			for i_tf in tf_temp:
				tfidf_temp[i_tf] =  tf_temp[i_tf]
			tfIdf_dict[posts[i]['Id']] = tfidf_temp
	return tfIdf_dict

## createTFIdf_A, creates a dictionary of tf-idf for each answer.
def createTFIdf_A(posts):
	idf_dict = idf(posts)
	tfIdf_dict = {}
	for i in range(len(posts)):
		tf_temp = Counter()
		if posts[i]['PostTypeId'] == PU.ANSWER:
			for i_word in posts[i]['Body']:
				tf_temp[i_word] += 1
			tfidf_temp = {}
			for i_tf in tf_temp:
				tfidf_temp[i_tf] =  tf_temp[i_tf] * idf_dict.get(i_tf, 0)
			tfIdf_dict[posts[i]['Id']] = tfidf_temp
	return tfIdf_dict

## cosScore measures similarity between question and it's answer
## it creates a cosine score fore each pair of question and its
## Answer

def cosScore(posts):
	cos_list = []
	tfIdf_Q = createTFIdf_Q(posts)
	tfIdf_A = createTFIdf_A(posts)
	for i in range(len(posts)):
		if posts[i]['PostTypeId'] == PU.ANSWER:
			vector_Q = tfIdf_Q[posts[i]['ParentId']]
			vector_A = tfIdf_A[posts[i]['Id']]
			word_common = list(set(vector_Q) & set(vector_A))
			t1 = np.sum([vector_Q[i_w]*vector_A[i_w] for i_w in word_common])
			norm_Q = np.sqrt(np.sum([i_q**2 for i_q in vector_Q.values()]))
			norm_A = np.sqrt(np.sum([i_a**2 for i_a in vector_A.values()]))
			cos = float(t1)/(norm_Q * norm_A)
			cos_list += [cos]
	return cos_list

##** Cosine similarity -- end

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