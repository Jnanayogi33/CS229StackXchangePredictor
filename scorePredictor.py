import postParseUtils as PU
import postTokenizerUtils as TU
import featureExtractors as FE

from bs4 import BeautifulSoup

#Obtain data
posts = PU.extractPosts('./stackExchangeData/Apple/PostsSmallSample.xml')
PU.printBasicStats(posts)

#Preprocess post data
posts = TU.stripHTML(posts)
posts = TU.stripPunctuation(posts) #except apostrophe
posts = TU.toLower(posts)
posts = TU.tokenize(posts)
posts = TU.stemTokens(posts) #porter stemming

#Save at this point to save time!
PU.savePosts(posts, './stackExchangeData/Apple/posts.pk1')

#Generate features
posts = PU.loadPosts('./stackExchangeData/Apple/posts.pk1')
unigramCounts = FE.tokenCountByPost(posts)
