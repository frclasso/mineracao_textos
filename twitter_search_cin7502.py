#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 08:46:18 2019

@author: moises
"""

from TwitterSearch import *
try:
    tso = TwitterSearchOrder() 
    tso.set_keywords(['Brasil', 'Música'])
    tso.set_language('pt')
    tso.set_include_entities(False)

    ts = TwitterSearch(
        consumer_key = 'aaaaaa',
        consumer_secret = 'bbbbbb',
        access_token = 'ccccccccc',
        access_token_secret = 'ddddddd'
     )

    for tweet in ts.search_tweets_iterable(tso):
        print( '@%s tweeted: %s' % ( tweet['user']['screen_name'], tweet['text'] ) )

except TwitterSearchException as e:
    print(e)
