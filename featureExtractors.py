from collections import Counter

def tokenCountByPost(posts):
    counts = []
    for i in range(len(posts)):
        curr = Counter()
        for word in posts[i]['Title']: curr[word] += 1
        for word in posts[i]['Body']: curr[word] += 1
        counts += [curr]
    return counts

def tokensPerPost(posts):
    counts = []
    for i in range(len(posts)): counts += [len(posts[i]['Title']) + len(posts[i]['Body'])]
    return counts

def totalPerToken(posts):
    counts = Counter()
    for i in range(len(posts)):
        for word in posts[i]['Title']: counts[word] += 1
        for word in posts[i]['Body']: counts[word] += 1
    return counts

