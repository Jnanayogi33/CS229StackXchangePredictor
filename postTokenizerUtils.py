from bs4 import BeautifulSoup
import re
import nltk

def stripHTML(posts):
    for i in range(len(posts)):
        posts[i]['Body'] = BeautifulSoup(posts[i]['Body']).get_text()
        posts[i]['Title'] = BeautifulSoup(posts[i]['Title']).get_text()
    return posts

def stripPunctuation(posts):
    for i in range(len(posts)):
        posts[i]['Body'] = re.sub("[^a-zA-Z0-9']", " ", posts[i]['Body'])
        posts[i]['Title'] = re.sub("[^a-zA-Z0-9']", " ", posts[i]['Title'])
    return posts

def toLower(posts):
    for i in range(len(posts)):
        posts[i]['Body'] = posts[i]['Body'].lower()
        posts[i]['Title'] = posts[i]['Title'].lower()
    return posts

def tokenize(posts):
    for i in range(len(posts)):
        posts[i]['Body'] = posts[i]['Body'].split()
        posts[i]['Title'] = posts[i]['Title'].split()
    return posts

def stemTokens(posts):
    stemmer = nltk.stem.PorterStemmer()
    for i in range(len(posts)):
        posts[i]['Body'] = [stemmer.stem(word) for word in posts[i]['Body']]
        posts[i]['Title'] = [stemmer.stem(word) for word in posts[i]['Title']]
    return posts
