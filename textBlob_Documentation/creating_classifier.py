#!/usr/bin/env python3

from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier


train = [
     ('I love this sandwich.', 'pos'),
     ('this is an amazing place!', 'pos'),
     ('I feel very good about these beers.', 'pos'),
     ('this is my best work.', 'pos'),
     ("what an awesome view", 'pos'),
     ('I do not like this restaurant', 'neg'),
     ('I am tired of this stuff.', 'neg'),
     ("I can't deal with this", 'neg'),
     ('he is my sworn enemy!', 'neg'),
     ('my boss is horrible.', 'neg')
]
test = [
     ('the beer was good.', 'pos'),
     ('I do not enjoy my job', 'neg'),
     ("I ain't feeling dandy today.", 'neg'),
     ("I feel amazing!", 'pos'),
     ('Gary is a friend of mine.', 'pos'),
     ("I can't believe I'm doing this.", 'neg')
]

cl = NaiveBayesClassifier(train)
cl.classify('This is an amazing library')

prob_dist = cl.prob_classify("This one's a doozy")
print(prob_dist.max())
print(round(prob_dist.prob("pos"), 2))
print(round(prob_dist.prob("pos"), 2))

