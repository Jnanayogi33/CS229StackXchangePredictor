import postParseUtils as PU
import postTokenizerUtils as TU
import featureExtractors as FE
import learningModels as LM
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm

#General control variables
nfolds = 5

print "Extract post data from xml"
posts = PU.extractPosts('./stackExchangeData/Apple/PostsSmallSample.xml')
PU.printBasicStats(posts)

print "Parse, tokenize posts"
posts = TU.stripHTML(posts)
posts = TU.stripPunctuation(posts) #except apostrophe
posts = TU.toLower(posts)
posts = TU.tokenize(posts)
posts = TU.stemTokens(posts)

print "Save posts"
PU.savePosts(posts, './stackExchangeData/Apple/posts.pk1')

print "Load posts"
posts = PU.loadPosts('./stackExchangeData/Apple/posts.pk1')

print "Generate features (X)"
unigramCounts = FE.tokenCountByAnswer(posts)
X = FE.vectorizeCounts(unigramCounts)

print "Choose data labels (Y)"
avg = PU.getAverageAnswerScore(posts)
stdev = PU.getAnswerScoreStDev(posts)
scoreSplits = [0, avg+stdev]
Y = FE.labelAnswerByScoreSplits(posts, scoreSplits)

print "Run n-folds validation and output resulting scores"
print LM.nFoldValidation(X,Y,nfolds,MultinomialNB())
print LM.nFoldValidation(X,Y,nfolds,svm.LinearSVC())