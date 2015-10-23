import xml.etree.ElementTree as ET
import pickle

QUESTION = "1"
ANSWER = "2"

#'./stackExchangeData/Apple/Posts.xml'
def extractPosts(inputSource):
    print "Extracting posts and making dictionary. Hang on this may take a bit:"
    print "1. Creating parse tree"
    tree = ET.parse(inputSource)
    root = tree.getroot()

    print "2. Extracting text, converting to dictionary form"
    posts = []
    for post in root:
        currEntry = {}

        #Extract information in raw form
        currEntry['Id'] = post.get('Id')
        currEntry['PostTypeId'] = post.get('PostTypeId')
        currEntry['ParentID'] = post.get('ParentID')
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

        #Save into posts list
        posts += [currEntry]

    print "3. Returning posts"
    return posts

# './stackExchangeData/Apple/posts.pk1'
def savePosts(outputDest):
    print "Saving posts into dictionary file"
    postsOutput = open(outputDest, 'wb')
    pickle.dump(posts, postsOutput)
    postsOutput.close()

# './stackExchangeData/Apple/posts.pk1'
def loadPosts(inputSource):
    print "Loading saved posts dictionary file"
    postsInput = open(inputSource, 'rb')
    posts = pickle.load(postsInput)
    postsInput.close()
    return posts

def printBasicStats(posts):
    print "Calculating some basic statistics"
    totalQuestions = sum(1 for post in posts if post['PostTypeId'] == QUESTION)
    totalAnswers = sum(1 for post in posts if post['PostTypeId'] == ANSWER)
    averageQuestionScore = float(sum(post['Score'] for post in posts if post['PostTypeId'] == QUESTION))/float(totalQuestions)
    averageAnswerScore = float(sum(post['Score'] for post in posts if post['PostTypeId'] == ANSWER))/float(totalAnswers)
    numZeroScoreAnswers = sum(1 for post in posts if (post['Score'] == 0) and (post['PostTypeId'] == ANSWER))

    print "Total Questions:", totalQuestions
    print "Total Answers:", totalAnswers
    print "Average Question Score:", averageQuestionScore
    print "Average Answer Score:", averageAnswerScore
    print "Number Zero-Score Answers:", numZeroScoreAnswers