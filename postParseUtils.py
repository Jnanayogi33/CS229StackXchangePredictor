import xml.etree.ElementTree as ET
import pickle
import numpy as np
from scipy.sparse import csr_matrix

QUESTION = "1"
ANSWER = "2"

def getXMLRoot(inputSource):
    tree = ET.parse(inputSource)
    return tree.getroot()

def extractPosts(inputSource):
    print "1. Creating parse tree"
    tree = ET.parse(inputSource)
    root = tree.getroot()

    print "2. Extracting text, converting to dictionary form"
    posts = []
    for post in root:
        currEntry = {}
        currEntry['Id'] = post.get('Id')
        currEntry['PostTypeId'] = post.get('PostTypeId')
        currEntry['ParentId'] = post.get('ParentId')
        currEntry['AcceptedAnswerId'] = post.get('AcceptedAnswerId')
        currEntry['CreationDate'] = post.get('CreationDate')
        currEntry['Score'] = post.get('Score')
        currEntry['ViewCount'] = post.get('ViewCount')
        currEntry['Body'] = post.get('Body')
        currEntry['OwnerUserId'] = post.get('OwnerUserId')
        currEntry['LastEditorUserId'] = post.get('LastEditorUserId')
        currEntry['LastEditorDisplayName'] = post.get('LastEditorDisplayName')
        currEntry['LastEditDate'] = post.get('LastEditDate')
        currEntry['LastActivityDate'] = post.get('LastActivityDate')
        currEntry['CommunityOwnedDate'] = post.get('CommunityOwnedDate')
        currEntry['ClosedDate'] = post.get('ClosedDate')
        currEntry['Title'] = post.get('Title')
        currEntry['Tags'] = post.get('Tags')
        currEntry['AnswerCount'] = post.get('AnswerCount')
        currEntry['CommentCount'] = post.get('CommentCount')
        currEntry['FavoriteCount'] = post.get('FavoriteCount')

        #Carry out basic conversions of string numbers into manipulable ints
        if currEntry['Score'] != None: currEntry['Score'] = int(currEntry['Score'])
        else: currEntry['Score'] = 0
        if currEntry['ViewCount'] != None: currEntry['ViewCount'] = int(currEntry['ViewCount'])
        else: currEntry['ViewCount'] = 0
        if currEntry['AnswerCount'] != None: currEntry['AnswerCount'] = int(currEntry['AnswerCount'])
        else: currEntry['AnswerCount'] = 0
        if currEntry['CommentCount'] != None: currEntry['CommentCount'] = int(currEntry['CommentCount'])
        else: currEntry['CommentCount'] = 0
        if currEntry['FavoriteCount'] != None: currEntry['FavoriteCount'] = int(currEntry['FavoriteCount'])
        else: currEntry['FavoriteCount'] = 0

        #Convert text entries into '' if does not exist
        if currEntry['Title'] == None: currEntry['Title'] = ''
        if currEntry['Body'] == None: currEntry['Body'] = ''

        #Save into posts list
        posts += [currEntry]

    print "3. Returning posts"
    return posts

def savePosts(posts, outputDest):
    postsOutput = open(outputDest, 'wb')
    pickle.dump(posts, postsOutput)
    postsOutput.close()

def loadPosts(inputSource):
    postsInput = open(inputSource, 'rb')
    posts = pickle.load(postsInput)
    postsInput.close()
    return posts

def getAllAnswerScores(posts):
    answerScores = []
    for post in posts:
        if post['PostTypeId']==ANSWER: answerScores += [post['Score']]
    return answerScores

# Save function for sparse matrices
def saveSparseCSR(array, filename):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

# Load function for sparse matrices
def loadSparseCSR(filename):
    if filename[-4:] != '.npz':
        filename = filename + '.npz'
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

def getTotalQuestions(posts):
    return sum(1 for post in posts if post['PostTypeId'] == QUESTION)
def getTotalAnswers(posts):
    return sum(1 for post in posts if post['PostTypeId'] == ANSWER)
def getAverageQuestionScore(posts):
    return float(sum(post['Score'] for post in posts if post['PostTypeId'] == QUESTION))/float(getTotalQuestions(posts))
def getAverageAnswerScore(posts):
    return float(sum(post['Score'] for post in posts if post['PostTypeId'] == ANSWER))/float(getTotalAnswers(posts))
def getAnswerScoreStDev(posts):
    return np.std(getAllAnswerScores(posts))

def printBasicStats(posts):
    print "Some basic statistics:"
    print "-------------------------------------------------------"
    numZeroScoreAnswers = sum(1 for post in posts if (post['Score'] == 0) and (post['PostTypeId'] == ANSWER))

    print "Total Questions:", getTotalQuestions(posts)
    print "Total Answers:", getTotalAnswers(posts)
    print "Average Question Score:", getAverageQuestionScore(posts)
    print "Average Answer Score:", getAverageAnswerScore(posts)
    print "Number Zero-Score Answers:", numZeroScoreAnswers
