#!/usr/bin/env python3

from textblob import TextBlob
from nltk import *

blob = TextBlob('Analytics Vidhya is a freat platform to learn Data Science.')

for np in blob.noun_phrases:
    print(np)