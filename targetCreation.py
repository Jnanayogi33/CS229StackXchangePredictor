import os, collections
import xml.etree.ElementTree as ET
import datetime
import numpy as np
import sklearn.linear_model, math
import pickle

## votesCreatedHist is used to find all vote date for
## an answer, output is a dict with key = postId of the 
## answer, value = a list of vote create date
def votesCreatedHist(rootVotes):
    votesCreatedHist_dict = {}
    for vote in rootVotes:
        PostId = vote.get('PostId')
        VoteTypeId = vote.get('VoteTypeId')
        CreationDateStr = vote.get('CreationDate')
        CreationDate = datetime.datetime.strptime(CreationDateStr, "%Y-%m-%dT%H:%M:%S.%f")
        if VoteTypeId in ('1', '2', '3'):
            if PostId not in votesCreatedHist_dict:
                votesCreatedHist_dict[PostId] = [CreationDate]
            else:
                votesCreatedHist_dict[PostId].append(CreationDate)
    return votesCreatedHist_dict


## questionCreationDate returns a dict with key is post id of question
## and value is question creation date 
def questionCreationDate(root):
    question_dict = {}
    for post1 in root:
        postId = post1.get('Id')
        if post1.get('PostTypeId') == '1':
            AcceptedAnswerId = post1.get('AcceptedAnswerId')
            creationDate_str = post1.get('CreationDate')
            question_dict[postId] = [datetime.datetime.strptime(creationDate_str, "%Y-%m-%dT%H:%M:%S.%f"), \
                                     AcceptedAnswerId]
    return question_dict

## questionViewCount returns a dict with key is post id of question
## and value is question view count
def questionViewCount(root):
    question_dict = {}
    for post in root:
        if post.get('PostTypeId') == '1':
            question_dict[post.get('Id')] = float(post.get('ViewCount'))
    return question_dict

def existingAnswersAtTimeOfPost(root):

    countAnswers = collections.Counter()
    answersPosted = collections.Counter()

    for post in root:
        if post.get('PostTypeId') == '2':
            parentId = post.get('ParentId')
            answersPosted[post.get('Id')] = countAnswers[parentId] + 1.0
            countAnswers[parentId] += 1.0
    return answersPosted


## answerScorePerDay returns an array of adjusted score for each answer,
# time_now is the datestamp when the data was updated on stackexchange
# Adjusted Score = raw score/median(days from answer created date to vote date)
def answerScorePerDay(root, time_now):
    score_per_day_Array = np.array([])
    for post in root:
        postId = post.get('Id')
        if post.get('PostTypeId') == '2':
            ParentId = post.get('ParentId')
            Score = int(post.get('Score'))
            creationDate_str = post.get('CreationDate')
            creationDate = datetime.datetime.strptime(creationDate_str, "%Y-%m-%dT%H:%M:%S.%f")
            if postId in votesCreatedDate_dict:
                voteDateList = votesCreatedHist_dict.get(postId)
            else:
                voteDateList = [time_now]
            total_days = np.median(np.array([(lastVoteDate - creationDate).days for i_date in voteDateList]))
            score_per_day = Score / float(total_days)
            score_per_day_Array = np.append(score_per_day_Array, score_per_day)
            if Score == 0 and (time_now - creationDate).days < 10:
                continue
    return score_per_day_Array

## answerScorePerDay returns an array of adjusted score for each answer,
# timeNow is the datestamp when the data was updated on stackexchange
# Adjusted Score = raw score/median(days from answer created date to vote date)
def answerScorePerViews(root, timeNow):

    scorePerViews = np.array([])
    qCreateDates = questionCreationDate(root)
    qViewCounts = questionViewCount(root)

    for post in root:
        if post.get('PostTypeId') == '2':

            Score = float(post.get('Score'))
            creationDate_str = post.get('CreationDate')
            creationDate = datetime.datetime.strptime(creationDate_str, "%Y-%m-%dT%H:%M:%S.%f")
            postAge = float((timeNow - creationDate).days)
            if Score == 0 and postAge < 10: continue

            parentId = post.get('ParentId')
            parentAge = float((timeNow - qCreateDates[parentId][0]).days)
            parentViewsPerDay = qViewCounts[parentId]/parentAge
            postViews = parentViewsPerDay * postAge
            if postViews < 0: print timeNow, creationDate, qCreateDates[parentId][0]
            scorePerViews = np.append(scorePerViews, Score/postViews)

    return scorePerViews

def getAnswerXY(root, timeNow):

    X = np.array([[0,0,0,0,0,0,0,0]])
    Y = np.array([[0]])
    qCreateDates = questionCreationDate(root)
    qViewCounts = questionViewCount(root)
    aExistingCounts = existingAnswersAtTimeOfPost(root)

    for post in root:
        if post.get('PostTypeId') == '2':

            Score = float(post.get('Score'))
            if Score >= 0: Score = math.log(Score + 1.0)
            else: Score = math.log(math.exp(Score/2.0))
            creationDate_str = post.get('CreationDate')
            creationDate = datetime.datetime.strptime(creationDate_str, "%Y-%m-%dT%H:%M:%S.%f")
            postAge = float((timeNow - creationDate).days)
            parentId = post.get('ParentId')
            parentAge = float((timeNow - qCreateDates[parentId][0]).days)
            parentViews = qViewCounts[parentId]
            existingAnsCount = aExistingCounts[post.get('Id')]
            X = np.append(X,[[postAge, parentAge, parentViews, existingAnsCount,
                              math.log(postAge), math.log(parentAge), math.log(parentViews), math.log(existingAnsCount)]], axis=0)
            Y = np.append(Y, [[Score]], axis=0)

    return X[1:], Y[1:]


def getAverageScorePerAnswerAge(root, timeNow):

    sumScores = collections.Counter()
    countScores = collections.Counter()
    averageScores = collections.Counter()

    for post in root:
        if post.get('PostTypeId') == '2':
            Score = float(post.get('Score'))
            creationDate_str = post.get('CreationDate')
            creationDate = datetime.datetime.strptime(creationDate_str, "%Y-%m-%dT%H:%M:%S.%f")
            postAge = int((timeNow - creationDate).days/100)
            sumScores[postAge] += Score
            countScores[postAge] += 1

    for key in countScores.keys():
        averageScores[key] = sumScores[key]/countScores[key]

    return averageScores

def getAverageScorePerParentAge(root, timeNow):

    sumScores = collections.Counter()
    countScores = collections.Counter()
    averageScores = collections.Counter()
    qCreateDates = questionCreationDate(root)

    for post in root:
        if post.get('PostTypeId') == '2':
            Score = float(post.get('Score'))
            parentId = post.get('ParentId')
            parentAge = int((timeNow - qCreateDates[parentId][0]).days/100)
            sumScores[parentAge] += Score
            countScores[parentAge] += 1

    for key in countScores.keys():
        averageScores[key] = sumScores[key]/countScores[key]

    return averageScores

def getAverageScorePerParentViews(root):

    sumScores = collections.Counter()
    countScores = collections.Counter()
    averageScores = collections.Counter()
    qViewCounts = questionViewCount(root)

    for post in root:
        if post.get('PostTypeId') == '2':
            Score = float(post.get('Score'))
            parentId = post.get('ParentId')
            parentViews = int(qViewCounts[parentId]/1000)
            sumScores[parentViews] += Score
            countScores[parentViews] += 1

    for key in countScores.keys():
        averageScores[key] = sumScores[key]/countScores[key]

    return averageScores

# Return scores with factors associated with
def getAdjustedScores(file):

    root = ET.parse(file).getroot()
    timeNow = datetime.datetime.fromtimestamp(os.path.getctime(file))

    X, Y = getAnswerXY(root, timeNow)
    model = sklearn.linear_model.LinearRegression()
    model = model.fit(X,Y)
    Y_hat = model.predict(X)
    # np.savetxt('Xtemp', X)
    # np.savetxt('Ytemp', Y)
    # np.savetxt('Yhattemp', Y_hat)
    return Y - Y_hat