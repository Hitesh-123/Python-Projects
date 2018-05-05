#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 02:46:12 2018

@author: hiteshady
"""

import numpy as np
import pandas as pd
import csv
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
train = pd.read_csv('/Users/hiteshady/Documents/MUIT-PGD/Python/Python-Projects/train_coding/train.csv')
chlg = pd.read_csv('/Users/hiteshady/Documents/MUIT-PGD/Python/Python-Projects/train_coding/challenge_data.csv')
test = pd.read_csv('/Users/hiteshady/Documents/MUIT-PGD/Python/Python-Projects/test_coding.csv')
chlg['challenge'] = chlg.challenge_ID
chlg = chlg.drop(['challenge_ID'],axis=1)
c1 = pd.merge(test,chlg,on='challenge',how='left')
c1 = c1.drop(['user_sequence','publish_date'],axis=1)
c1.loc[c1.challenge_series_ID.isnull(),'challenge_series_ID'] = c1.challenge_series_ID.mode()[0]
c1.loc[c1.total_submissions.isnull(),'total_submissions'] = c1.total_submissions.median()
c1.loc[c1.author_ID.isnull(),'author_ID'] = c1.author_ID.mode()[0]
c1.loc[c1.author_gender.isnull(),'author_gender'] = c1.author_gender.mode()[0]
c1.loc[c1.author_org_ID.isnull(),'author_org_ID'] = c1.author_org_ID.mode()[0]
c1.loc[c1.category_id.isnull(),'category_id'] = c1.category_id.mode()[0]
c1.challenge_series_ID = c1.challenge_series_ID.astype('category').cat.codes
c1.author_ID = c1.author_ID.astype('category').cat.codes
c1.author_org_ID = c1.author_org_ID.astype('category').cat.codes
c1.author_gender = c1.author_gender.astype('category').cat.codes
u1 = c1.user_id.unique()
list1 = []
n = 0
for u in u1:
    df1 = c1.loc[c1.user_id==u,:]
    df2 = df1.drop(['user_id','challenge_sequence','challenge'],axis=1)
    df2.index = [x for x in range(10)]
    df3 = df2.T
    tra = np.array(df3.loc[:,0:8],order='C',copy=False)
    tar = np.array(df3.loc[:,9],order='C',copy=False)
    model = GradientBoostingRegressor(random_state=56)
    model = model.fit(tra,tar)
    tes = np.array(df3.loc[:,1:9],order='C')
    pred11 = model.predict(tes)
    df3.loc[:,10] = [int(x) for x in pred11]
    tes1 = np.array(df3.loc[:,2:10],order='C')
    pred12 = model.predict(tes1)
    df3.loc[:,11] = [int(x) for x in pred12]
    tes2 = np.array(df3.loc[:,3:11],order='C')
    pred13 = model.predict(tes2)
    df3.loc[:,12] = [int(x) for x in pred13]
    df4 = df3.loc[:,10:12].T
    list1.append(df4)
    n = n+1
    print('n = ',n)


c2 = pd.merge(train,chlg,on='challenge',how='left')
c2 = c2.drop(['user_sequence','publish_date'],axis=1)
c2.loc[c2.challenge_series_ID.isnull(),'challenge_series_ID'] = c2.challenge_series_ID.mode()[0]
c2.loc[c2.total_submissions.isnull(),'total_submissions'] = c2.total_submissions.median()
c2.loc[c2.author_ID.isnull(),'author_ID'] = c2.author_ID.mode()[0]
c2.loc[c2.author_gender.isnull(),'author_gender'] = c2.author_gender.mode()[0]
c2.loc[c2.author_org_ID.isnull(),'author_org_ID'] = c2.author_org_ID.mode()[0]
c2.loc[c2.category_id.isnull(),'category_id'] = c2.category_id.mode()[0]
c2.challenge_series_ID = c2.challenge_series_ID.astype('category').cat.codes
c2.author_ID = c2.author_ID.astype('category').cat.codes
c2.author_org_ID = c2.author_org_ID.astype('category').cat.codes
c2.author_gender = c2.author_gender.astype('category').cat.codes
target = c2.challenge
c2 = c2.drop(['user_id','challenge_sequence','challenge'],axis=1)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
t1 = le.fit_transform(target)
tra = np.array(c2[:5000],order='C')
tar = np.array(t1[:5000],order='C')

model = RandomForestClassifier(random_state=56,verbose=2)
model = model.fit(tra,tar)
list2 = []
for s in list1:
    tes = np.array(s,order='C')
    pred = model.predict(tes)
    p1 = le.inverse_transform(pred)
    list2.append(p1)

list3 = []
for x in list2:
    list3.append(pd.Series(x))

f1 = pd.concat(list3)
f1.index = [x for x in range(len(f1))]
sub1 = pd.read_csv('/Users/hiteshady/Documents/MUIT-PGD/Python/Python-Projects/sample_submission_J0OjXLi_DDt3uQN.csv')
sub1['challenge'] = f1
sub1.to_csv('/Users/hiteshady/Documents/MUIT-PGD/Python/Python-Projects/Mckinsey.csv',index=False)
