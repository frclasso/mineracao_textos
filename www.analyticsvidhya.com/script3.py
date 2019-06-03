#!/usr/bin/env python3

from textblob import TextBlob


blob = TextBlob('Analytics Vidhya is a great platform to learn Data Science.')

for words , tag in blob.tags:
    print(words, tag)