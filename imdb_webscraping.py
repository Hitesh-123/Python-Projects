#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 00:56:52 2018

@author: hiteshady
"""
#%%

import requests
import lxml.html

r = requests.get('https://www.imdb.com/title/tt5956100/plotsummary?ref_=tt_stry_pl')
html = r.content

tree = lxml.html.fromstring(html)

fp = open('imdb_tjh.txt','w')

#summary extraction pg wise
sumre = tree.xpath("//ul/li[contains(@id,'summary')]/p")

#delete if not requried
fp.write('SUMMARY:\n')

for s in sumre:
    print(s.text_content())
    fp.write(s.text_content())
    fp.write('\n\n')

synop = tree.xpath("//ul/li[contains(@id,'synopsis')]")

#delete if not requried
fp.write('SYNOPSIS:\n')

#synopsis extraction pg wise
for s in synop[0].itertext():
    print(s)
    fp.write(s)
    fp.write('\n\n')

fp.close()
