#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 09:36:49 2018

@author: hiteshady
"""

from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json

ckey='8tClQD6IXd336lHgR6iWfSmor'
csecret='6DJN7r0h5CrtWBJU1BeVdxYX3PNkHQig9dGrF2dDPiJMiMJ8nn'
atoken='136671045-I3W7dqHf9OfQ3A0dBAhhoy7u0LWS1OkbaffgduG0'
asecret='c1j5ZXz0J1GgVJBodHfYFka6thjtj3r8dehneUvpCclsT'

class listener(StreamListener):

    def on_data(self, data):
        all_data = json.loads(data)
        
        tweet = all_data["text"]
        
        username = all_data["user"]["screen_name"]
        
        print((username,tweet))
        
        return True

    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["Paytm"])
