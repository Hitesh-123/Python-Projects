#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 14:22:34 2018

@author: hiteshady
"""

import numpy as np
import pandas as pd
import csv
import tweepy

ckey='8tClQD6IXd336lHgR6iWfSmor'
csecret='6DJN7r0h5CrtWBJU1BeVdxYX3PNkHQig9dGrF2dDPiJMiMJ8nn'
atoken='136671045-I3W7dqHf9OfQ3A0dBAhhoy7u0LWS1OkbaffgduG0'
asecret='c1j5ZXz0J1GgVJBodHfYFka6thjtj3r8dehneUvpCclsT'

auth = tweepy.OAuthHandler(ckey,csecret)
auth.set_access_token(atoken,asecret)

api = tweepy.API(auth)
csvFile = open('paytm.csv', 'a')
csvWriter = csv.writer(csvFile)
for tweet in tweepy.Cursor(api.search,q="@Paytm",count=100,lang="en",since="2018-04-01").items():
    print (tweet.created_at, tweet.text)
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])



