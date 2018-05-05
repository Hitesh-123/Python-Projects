#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 11:27:34 2018

@author: hiteshady
"""

import numpy as np
import scipy as sp
import tensorflow as tf
import pandas as pd
a = np.array([1,2,3,4,5])
a
a.fill(5)
a
a[1] = 45
a
type(a)
shape(a)
a.shape
a = np.array([[0,1,2,3],[10,11,12,13]])
a
a.shape
a.size
a.ndim
np.shape(a)
a[1,3]
a[2,3]
a[0,1:]
a[0,3:5]
a[4:,4:]
a[:,2]
a[2::2,::2]
a = np.array([[1,2,3],[4,5,6]],float)
print(a)
print(a.shape)
a.dtype.type
type(a[0,0]) is type(a[1,2])
b = a[:,::2]
b[0,1] = 100
a
c = a[:,::2].copy()
c[1,0] = 500
a
a = np.arange(0,80,10)
y = a[[1,2,-3]]
y
y = np.take(a,[1,2,-3])
y
mask = np.array([0,1,1,0,0,1,0,0],dtype = bool)
y = a[mask]
y
y = np.compress(mask,a)
y
a = np.arange(36).reshape(6,6)
a[(0,1,2,3,4),(1,2,3,4,5)]
a[3:,[0,2,5]]
mask = np.array([1,0,1,0,0,1],dtype = bool)
a[mask,2]
a = np.array([[1,2,3],[4,5,6]],float)
sum(a)
sum(a, axis = 0)
b = np.array([2.,3.,0.,1.])
b.min(axis=0)
np.amin(b,axis=0)
b.argmin(axis=0)
np.argmin(b, axis=0)
np.array([1.5,1.5,1.5])
b.var(axis=0)
b.std(axis=0)
a = np.array([[1,2,3],[4,5,6]],float)
a.clip(3,5)
a = np.array([1.35,2.5,1.5])
a.round()
a.ptp(axis=0)
a.ptp(axis=None)
s = pd.Series([3,7,4,4,0.3],['a','b','c','d','e'])
df = pd.DataFrame(np.arange(9).reshape(3,3),['a','b','c'],['Paris','Berlin','Madrid'])
data = {'Paris':[0,3,6,99999999],'Berlin':[1,4,7],'Madrid':[2,5,8]}
df
s
data
df[:2]
df[2:]
df[:1]
df['Paris']
df[df['Paris']>1]
df.Berlin[df['Berlin']>1] = 0
df
df.ix['a','Berlin']
df.drop('c')
df.drop('Berlin',axis=1)
s
s.drop('a')
s2 = pd.Series([0,1,2],['a','c','f'])
s+s2
s.add(s2,fill_value=0)
s.subtract(s2,fill_value=0)
df2 =pd.DataFrame(np.arange(12).reshape(4,3),['b','e','c','a'],['Paris','Lisbonne','Madrid'])
df2
df+df2
df.add(df2,fill_value=0)
df1 = pd.DataFrame({'data1':[0,1,2,3,4,5,6],'keyleft':['b','b','a','c','a','a','b']})
df1
pd.df2 = pd.DataFrame({'data2':[0,1,2],'key':['a','b','d']})
df2
pd.merge(df1,df2,left_on = 'keyleft',right_on = 'key', how = 'inner')
pd.merge(df1,df2,left_on = 'keyleft',right_on = 'key', how = 'outer')
pd.merge(df1,df2)
left1 = pd.DataFrame({'key':['a','b','a','a','b','c'],'data1':[0,1,2,3,4,5]})
left1
right1 = pd.DataFrame({'data2':np.arange(5),'key':['a','b','a','b','d']})
right1
pd.merge(left1,right1,on = 'key', how = 'left')
s
s.rank
s.rank(method = "first")
s.rank(method = 'max',ascending = False)
s.order()
s.sort_index
df.max()
df.min()
df + df.max()
s.order()
df + df.min()
import math
f = lambda x:math.sqrt(x)
df.applymap(f)
df.Berlin = df['Berlin'].map(f)
df.describe()
df.sum(axis=1)
df.cov()
df.corr()
df.reindex(['c','b','a','g'])
df.reindex(['c','b','a','g'],fill_value = 14)
df.reindex(columns = ['Varsovie','Paris','Madrid'])
getwd()
import os.path
getwd()
os.getcwd()


#%%

a = np.arange(0,80,10)
a
y = a[[1,2,-3]]
y
y = np.take(a,[1,2,-3])
y
ind = [1,2,-3]
y = np.take(a,ind)
y
mask = np.array([0,1,1,0,0,1,0,0], dtype=bool)
y = a[mask]
y
y = np.compress(mask,a)
y
a = np.array([[1,2,3],[4,5,6]],float)
a
sum(a)
np.sum(a)
np.sum(a,axis = 0)
a.sum()
%timeit sum(a)
%timeit a.sum()
%timeit np.sum(a)
b = np.arange(0,80,5).reshape(4,4)
b
np.amin(b, axis = 0)
b.max(axis = None)
b.argmax(axis = None)
b.std(axis=0)
b.var(axis=0)
np.var(b,axis=0)
np.mean(b,axis=0)
b.mean(axis=0)
np.average(b,axis=0)
np.average(a, weights = [1,2],axis=0)
a.clip(3,5)
a.round(decimals=1)
a.flatten()
np.flat(a)
a.ravel()
id(a)
a.swapaxes(0,1)
a.resize(3,2)
a
c = np.array([1,2,3,4,5,6])
c.reshape(2,3)
c.reshape(6,)
c.reshape(1,6)
a.transpose()
a.squeeze()
a.nonzero()
a.round()
a
a.cumsum(axis=None)
a.sort(axis=-1)
a
a.cumprod(axis=None)
a.sum()
a = (2*pi)/10
#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('classic')
import numpy as np
x = np.linspace(0,10,100)
x
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
plt.show()
plt.draw()
from IPython.display import Image
plt.subplot(2,1,1)
plt.plot(x, np.sin(x))
plt.subplot(2,1,2)
plt.plot(x,np.cos(x))
fig, ax = plt.subplots(5)
ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x))
ax[2].plot(x, np.tan(x))
ax[3].plot(x, np.cos(x))
ax[4].plot(x, np.tan(x))
fig = plt.figure()
ax = plt.axes()
x =np.linspace(0,10,1000)
x
ax.plot(x, np.sin(x));
plt.plot(x, np.sin(x - 0), color = 'blue')
plt.plot(x,np.sin(x - 1), color = 'g')
plt.plot(x, np.sin(x - 2), color = '0.75')
plt.plot(x, np.sin(x - 3), color = '#FFDD44')
plt.plot(x, np.sin(x - 4), color = (1.0,0.2,0.3))
plt.plot(x, np.sin(x - 5), color = 'chartreuse')
plt.plot(x, x + 0, linestyle='solid')
plt.plot(x, x + 1, linestyle='dashed')
plt.plot(x, x + 2, linestyle='dashdot')
plt.plot(x, x + 3, linestyle='dotted');
plt.plot(x, x + 4, linestyle='-')  # solid
plt.plot(x, x + 5, linestyle='--') # dashed
plt.plot(x, x + 6, linestyle='-.') # dashdot
plt.plot(x, x + 7, linestyle=':');
plt.plot(x, x + 0, '-g')  # solid green
plt.plot(x, x + 1, '--c') # dashed cyan
plt.plot(x, x + 2, '-.k') # dashdot black
plt.plot(x, x + 3, ':r');  # dotted red

plt.plot(x, np.sin(x))
plt.xlim(-1, 11)
plt.ylim(-1.5, 1.5);
plt.plot(x, np.sin(x))
plt.xlim(10, 0)
plt.ylim(1.2, -1.2);
plt.plot(x, np.sin(x))
plt.axis([-1, 11, -1.5, 1.5]);

plt.plot(x, np.sin(x))
plt.title("A Sine Curve")
plt.xlabel("x")
plt.ylabel("sin(x)");

plt.plot(x, np.sin(x), '-g', label='sin(x)')
plt.plot(x, np.cos(x), ':b', label='cos(x)')
plt.axis('equal')
plt.legend();

from sklearn import datasets

b = 30000000
import math
math.log(b)
#%%%

rollnoL = [109,102,105,106,103,110,101,107,104,111,108]
nameL = ['meena','apoorva','kaustav','shubham','goldie','hitesh','shruti','vijay','achal','lalit','varun']
genderL = ['F','F','M','M','M','M','F','M','M','M','M']
import numpy as np
import pandas as pd
pythonL = np.random.randint(60,90,11)
pythonL
sasL = np.random.randint(60,90,11)
sasL
nameS = pd.Series(nameL,index=rollnoL)
nameS
type(nameS)
n1 = pd.Series(nameL)
n1.index = rollnoL
112 in n1.index
111 in n1.index
n1.index
n1.keys()
112 in n1
nameS.items
nameS.keys
nameS.values
list(nameS,items())
nameS.ix[108] = 'jain'
nameS[108]
nameS
nameS[108] = 'varun'
nameS[nameS == "varun"]
nameS
nameS[0:5]
nameS[101:106]
nameS.iloc[0:5]
nameS[0]
nameS.iloc[0]
nameS.loc[108]
nameS[nameS == 'meena']
nameS == 'meena'
nameS[0:1]
nameS.iloc[4:6]
nameS.loc[103:110]
nameS.ix[108]
nameS
df1 = pd.Series(rollnoL)
names = pd.Series(nameL)
sas = pd.Series(sasL)
python = pd.Series(pythonL)
gender = pd.Series(genderL)
df = pd.concat([df1,names,gender,sas,python], axis=1)
df.columns = ['rollno','name','gender','sas','python'] 
studentDf = pd.DataFrame({'rollno':rollnoL,'name':nameL,'gender':genderL,'python':pythonL,'sas':sasL})
studentDf.index = rollnoL
studentDf
df3 = studentDf.reindex(columns=sorted(studentDf.columns))
df3
studentDf.columns = ('name','rollno','gender','python','sas')
studentDf
studentDf = studentDf[['name','rollno','gender','python','sas']]
studentDf
studentDf2 = studentDf[['name','rollno']]
studentDf2
studentDf.values
studentDf.T
studentDf.values[0]
studentDf.loc[109]
studentDf.iloc[0:1]
studentDf['name']
studentDf(3,2)
studentDf.iloc[:3,:2]
studentDf.loc[:105,:'python']
studentDf.iloc[:5,:2]
studentDf['total'] = studentDf['python'] + studentDf['sas']
studentDf
studentDf.loc[studentDf.total>150,]
studentDf.iloc[0:5]
studentDf.head()
studentDf
course = ['pg','pg','msc','msc','pg','pg','pg','pg','pg','pg','bsc']
hadoop = np.random.randint(71,90,11)
fees = np.random.randint(300000,600000,11)
hostel = [True,False,True,False,False,True,False,True,True,True,False]
course = pd.Series(course)
hadoop = pd.Series(hadoop)
fees = pd.Series(fees)
hostel = pd.Series(hostel)
df6 = pd.concat([course,hadoop,fees,hostel],axis = 1)
df6
df6.columns = (["course","hadoop","fees","hostel"])
df6
df7 = pd.concat([studentDf,df6],axis = 1)
df7
df6.index = rollnoL
df6
df7.to_csv("students2.csv")
df7.groupby('gender').mean()
df7.columns
from numpy import random
classes = ['C1','C2','C3']
sclass = random.choice(classes,11)
sclass
df7['sclass'] = sclass
df7
pd.pivot_table(df7, index = ['name'])
pd.pivot_table(df7, index = ['name','sclass','hostel'])
pd.pivot_table(df7, index = ['sclass','gender'])
pd.pivot_table(df7, index = ['course','sclass'], values = ['total','python'])
pd.pivot_table(df7, index = ['course','sclass'], values = ['total','python'],aggfunc = np.sum)
pd.pivot_table(df7, index = ['course','sclass'], values = ['total','python'],aggfunc = [np.sum,np.mean,len])
#%%%

import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = 2 * x - 5 + rng.randn(50)
x
rng
y
plt.scatter(x,y);
from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
model.fit(x[:, np.newaxis], y)
xfit = np.linspace(0, 10, 1000)
yfit = model.predict(xfit[:, np.newaxis])
plt.scatter(x, y)
plt.plot(xfit, yfit);
rng = np.random.RandomState(1)
X = 10 * rng.rand(100, 3)
y = 0.5 + np.dot(X, [1.5, -2., 1.])
model.fit(X, y)
print(model.intercept_)
print(model.coef_)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
x = np.array([2, 3, 4])
poly = PolynomialFeatures(3, include_bias=False)
poly.fit_transform(x[:, None])
poly_model = make_pipeline(PolynomialFeatures(7),LinearRegression())
poly_model
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = np.sin(x) + 0.1 * rng.randn(50)
poly_model.fit(x[:, np.newaxis], y)
yfit = poly_model.predict(xfit[:, np.newaxis])
plt.scatter(x, y);
plt.plot(xfit, yfit);
x


#%%
import numpy as np
import pandas as pd
rng = np.random.RandomState(42)
rng
marks = pd.Series(rng.randint(50,100,11))
marks
rng = np.random.RandomState(42)
marks1 = pd.Series(rng.randint(50,100,11))
marks == marks1
marks1.sum()
marks.sum()
marks.std()

{'x':1, 'y':[1,2]}
rng = np.random.RandomState(42)
A = pd.Series(rng.randint(0,10,6))
B = pd.Series(rng.randint(0,10,6))
df = pd.DataFrame(A,B)
df
df = pd.concat([A,B],axis = 1)
df
df.columns = ['A','B']
df
df.sum()
df.mean()
df.mean(axis=0)
df.mean(axis=1)
df.mean(axis = 'columns')
df.groupby(['A']).sum()
df1 = pd.DataFrame({'Name':['A','B','C']*2,'data1':range(6),'Value':rng.randint(0,10,6)},columns = ['Name','data1','Value'])
df1
df1.groupby('Name').aggregate(['min','max','median'])
df1.groupby('Name').aggregate([np.median,'median'])# error
df1.groupby('Name').aggregate({'Name':'min','data1':['max','min']})

df1.filter(items = ['data1','Value'])
df1.filter(like = '2', axis = 0)
df1.filter(like = '2', axis = 1)
df1.groupby('Name').std()
df1
df1.filter(items = ['key','Name'])
df1['Value'].mean() > 4
grouped = df1.groupby('Name')
grouped.filter(lambda x:x['Value'].mean()>4)
x = 2
y = 3
product = lambda x,y : x*y
product(x,y)

grouped.filter(lambda x:x['Value'].mean()>4)
grouped.filter(lambda x:x['Value'].std()>4)
grouped.transform(lambda x:x - x.mean())
grouped.apply(lambda x:x['Value'] * 2)
df2 = df1.set_index('Name')
df2
newmap = {'A':'Post Graduate','B':'Master of Science','C':'Bachelor of Science'}
newmap
df2.groupby(newmap).sum()
df2.groupby(str.lower).mean()
df2.groupby([str,str.lower,newmap]).mean()
df2.groupby('Name').sum().unstack()


df = pd.read_csv('~/Documents/MUIT-PGD/Python/Python-Projects/students2.csv', header = 0)
df
del df['Unnamed: 0']
df
df['rollno'].dtype
df.describe()
df.groupby('course')['sas'].describe()
df.groupby('course')['sas'].describe().unstack()
import csv
pd.pivot_table(df, index = 'course', values =['sas','hadoop'],aggfunc = [np.mean,np.median,min,max])
import matplotlib.pyplot as plt
pd.pivot_table(df, index = 'gender', columns ='course',values = 'sas').plot(kind='bar')
from keras.models import Sequential
import tensorflow as tf

##%%
x = [1,2,3]
y = [2,4,1]

plt.plot(x,y)

plt.xlabel('X Axis')
plt.ylabel('Y Axis')


x1 = [1,2,3]
y1 = [2,4,1]
x2 = [1,2,3]
y2 = [4,1,3]


plt.plot(x1,y1, label='Line1')
plt.plot(x2,y2, label='Line2')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.show()

import plotly.plotly as py
x = [10,5,6,7]
y = [4,5,8,9]
width = 1/1.5
plt.bar(x,y,width,color = "blue")
marks = np.random.uniform(30,100,1000)
np.all(marks>=30)
np.all(marks< 100)
range = (20,100)
bins = 10
plt.hist(marks,bins,range,color='green',histtype='bar',rwidth=0.8)
plt.xlabel('Marks')
plt.ylabel('No of Students')
plt.title('Histogram of Marks of Students')
plt.show()


activity = ['sleep','study','eat']
colors = ['red','green','yellow']
plt.pie(y)
plt.pie(y,labels=activity,colors = colors,startangle=90, shadow=True,radius=2,explode=(0.5,0.0,0.0),autopct = '%1.1f%%')



import torch
import torchvision
#%%

a = ['Apple','Samsung','Micromax','Huwaei']
for i in a:
    print(i)
        
x = int(input("Please enter a number:"))
if x < 0:
    x = 0
    print('Negative changed to zero')
else:
    print('zero')
    
    
words = ['cat','window','defenstrate']
for w in words:
    print(w,len(w))
for w in words:
    if len(w) > 6:
        words.insert(0,w)
for i in range(7):
    print(i)
    
for i in range(6,100,5):
    print(i)

for i in range(7,1000,20):
    print(i)
    
for i in range(-1,-100,-1):
    print(i)
    
a = ['Mary', 'had', 'a', 'little', 'lamb']
for i in range(len(a[1])):
    print(i)
print(range(50))


#%%

for n in range(2,10):
    for x in range(2,n):
        if n % x == 0:
            print(n,'equals',x,'*',n//x)
            break
    else:
        print(n,'is a prime number')
        
        
        
for num in range(2,10):
    if num % 2 == 0:
        print("Found an even number", num)
        continue
    print("Found a number", num)
    
    
def fib(n):
    """Print a Fibonacci series up to n."""
    a,b = 0,1
    while a < n:
        print(a, end = ' ')
        a, b = b, a + b
    print()
fib(2000)

def ask_ok(prompt, retries = 4, reminder = 'Please try again!'):
    while True:
        ok = input(prompt)
        if ok in ('y','ye','yes'):
            return True
        if ok in ('n','no','nop','nope'):
            return False
        retries = retries - 1
        if retries < 0:
            raise ValueError('invalid user response')
        print(reminder)
i = 5

def f(arg=i):
    print(arg)
    
i = 6
f()