#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 10:31:22 2018

@author: hiteshady
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
df = pd.read_csv('/Users/hiteshady/Documents/MUIT-PGD/Python/Python-Projects/Contriesdata.csv', header = 0)
df.head()
df.isnull().sum().sum()
df = df.fillna(0)
df['Climate']
x = df["Birthrate"]
y = df["Literacy"]
x = x.reshape(-1,1)
y = y.reshape(-1,1)
model = sm.OLS(y,x).fit()
model.summary()
x_new = pd.DataFrame({'Birthrate':[50]})
model.predict(x_new)
x_new
import seaborn as sns
sns.pairplot(df, x_vars=['Birthrate'], y_vars='Literacy', size=7, aspect=0.7, kind='reg')
model.conf_int()
model.pvalues
model.rsquared

#%%%
from sklearn import linear_model
df1 = pd.concat([df[col].dropna().reset_index(drop=True) for col in df], axis=1)
df1
df1 = pd.DataFrame(df1)
df1
target = pd.DataFrame(df1, columns=["GDP per captia USD per annum"])
df1.head()
df1 = df1.fillna(0)
df1.shape
df1
x = df1["Deathrate"]
y = target
x=x.reshape(-1, 1)
lm = linear_model.LinearRegression()
model = lm.fit(y,x)
predictions = lm.predict(y)
print(predictions)[0:5]
#%%%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
df = pd.read_csv('/Users/hiteshady/Documents/MUIT-PGD/Python/Python-Projects/Contriesdata.csv', header = 0)
df.head()
df.describe()
str(df)
df.isnull().sum().sum()
df.isnull().sum()
