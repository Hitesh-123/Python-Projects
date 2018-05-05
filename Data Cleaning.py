#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 19:22:40 2018

@author: hiteshady
"""

L = [1,2,'a',True]
sum(L[0:1])
L1 = [1,2,3,4,5]
L2 = [1,2,3,'a',True]
L3 = [i for i in range(5)]
L1
L2
L3
type(L1)
type(L2)
type(L3)
type(L3[4])
for i in range(len(L2)):
    print(type(L2[i]), end = ' ')
L = L1 + L2
print(L)
sum(L)
sum(L[1:4])
L[len(L)-2].upper()
L4 = [1,2,L2]
L4
L4[1]
L4[2]
L4[2][2]
# Multiple levels
L5a = [1,2,3,4,5]
L5b = [6,7,8,9,10]
L5 = [L5a,L5b]
L5
L5[1]
L5[0][3]
L5[1][4]
d1 = {1:'achal',2:'apoorva',3:'hitesh','dean':'dhiraj'}
d1
d1.keys()
d1.values()
d1.items()
d1[1]
d1['dean']
d2 = {'a':L1,'b':L3}
d2
d3 = {'a':L1,'b':L2}
d3['c'] = d3.pop('b')
d3
d3['c'][3] = 'z'
d3
vars(d1)
for key in d2:
    print(key, end = ' ')
    print(d2[key], end = ' ')
    
L1 =[1,2,3,4,5,5]
S1 = set(L1)
type(S1)
S1
a = set([1,2,3,4])
b = set([3,4,5,6])
a|b
c = a&b
a<b
a-b
a^b
S1 = {1,2,3}
type(S1)
S2 = set()
type(S2)
S2.add(4)
S2.add(5)
S2
any(S1)
s1 = set([1,2,4,'apple','banana','Tom',3])
s1.add(9)
s1
s1.discard(1)
s1.pop()
s2 = set([4,5,6,'cat'])
s3 = s1.union(s2)
s3
s1.update(s2)
s3 = s1.intersection(s2)
s3 - s1.difference(s2)
s1.difference_update(s2)
s1.isdisjoint(s2)
import numpy as np
np.zeros(10, dtype = int)
L = [i for i in range(0,100,2)]
L
np.ones((3,5),dtype = float)
np.ones((3,5), dtype=int)
np.full(shape=(3,5),fill_value = 3.14, dtype = int)
np.arange(0,20,2)
np.linspace(0,1,5)
np.random.random((3,3))
np.random.normal(0,1,(3,3))
np.eye(3)
np.empty(7)
np.zeros(10, dtype = 'int16')
np.random.seed(0)
x1 = np.random.randint(10, size=6)
x1
x2 = np.random.randint(10, size=(3,4))
x2
x3 = np.random.randint(10, size=(3,4,5))
x3
print("x3 ndim: ", x3.ndim)
print("x3 shape: ", x3.shape)
print("x3 size: ", x3.size)
print("dtype:",x3.dtype)
print("itemsize:", x3.itemsize, "bytes")
print("itemsize:", x3.nbytes, "bytes")
x2
x2[2,1]
x2[0,0] = 12
x = np.arange(10)
x[:5]
x[5:]
x[4:7]
x[::2]
x[2::]
x[:2:]
x[::-1]
x[5::-2]
x2
x2[:2, :3]
x2[:2, ::2]
x2[::-1, ::-1]
print(x2[:,0])
print(x2[0])
x2_sub = x2[:2,:-2]
x2_sub
x2_sub[0,0] = 99
x2_sub
x2_sub_copy = x2[:2,:2].copy()
x2_sub_copy
import pandas as pd
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
data=pd.Series([0.25,0.5,0.75,1.0],index = ['a','b','c','d'])
data
data['b']
dict1 = {'b':23,'d':45,'h':67}
s1 = pd.Series(dict1)
s1
pd.Series(5,index=[100,200,300])
population = [100,200,300]
area = [10,20,30]
states = pd.DataFrame({'population':population,'area':area})
states
states.columns
rollno = [1,2,3]
names = ['A','B','C']
df1 = pd.DataFrame(rollno,columns=['rollno'])
sdata = pd.DataFrame({'rollno':rollno,'sname':names})
sdata
sdata2 = pd.DataFrame(list(zip(rollno,names)))
sdata2
pd.DataFrame([{'a':1,'b':2},{'b':3,'c':4}])
pd.DataFrame(np.random.rand(3,2),columns=['foo','bar'],index=['a','b','c'])
ind = pd.Index([2,3,5,7,11])
ind
ind[1]
ind[::2]
list(ind[::2])
inda = pd.Index([1,3,5,7,9])
indb = pd.Index([2,3,6,7,8])
inda & indb
data.iloc[1]
data.loc['a']
data
data.area

















