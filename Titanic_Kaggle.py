#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 22:26:23 2018

@author: hiteshady
"""

import numpy as np
import pandas as pd
import keras as ks
import tensorflow as tf
from sklearn.model_selection import train_test_split,GridSearchCV
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
label=LabelEncoder()
%matplotlib inline
train_df = pd.read_csv('/Users/hiteshady/Documents/MUIT-PGD/Python/Python-Projects/titanic_train.csv',header = 0)
test_df = pd.read_csv('/Users/hiteshady/Documents/MUIT-PGD/Python/Python-Projects/titanic_test.csv',header = 0)
train_df.name()
train_df.head()
train_df.isnull().sum()
test_df.isnull().sum()
test_df.Age.fillna(test_df.Age.mean(), inplace=True)
test_df['Cabin'] = test_df['Cabin'].astype('category')
train_df.Age.fillna(train_df.Age.mean(), inplace=True)
train_df['Cabin'] = train_df['Cabin'].astype('category')
data_df = train_df.append(test_df)
passenger_id = test_df['PassengerId']
train_df.drop(['PassengerId'],axis = 1,inplace = True)
test_df.drop(['PassengerId'],axis = 1,inplace = True)
test_df.shape
train_df.isnull().sum()
test_df.isnull().sum()
train_df['Sex'] = train_df.Sex.apply(lambda x: 0 if x == "female" else 1)
test_df['Sex'] = test_df.Sex.apply(lambda x: 0 if x == "female" else 1)
train_df.head()
pd.options.display.max_columns = 99
test_df['Fare'].fillna(test_df['Fare'].mean(),inplace=True)
train_df.describe()
train_df.groupby('Survived').mean()
train_df.groupby('Sex').mean()
train_df.corr()
plt.subplots(figsize = (15,8))
sns.heatmap(train_df.corr(), annot=True,cmap="PiYG")
plt.title("Correlations Among Features", fontsize = 20)
plt.subplots(figsize = (15,8))
sns.barplot(x = "Sex", y = "Survived", data=train_df, edgecolor=(0,0,0), linewidth=2)
plt.title("Survived/Non-Survived Passenger Gender Distribution", fontsize = 25)
labels = ['Female', 'Male']
plt.ylabel("% of passenger survived", fontsize = 15)
plt.xlabel("Gender",fontsize = 15)
plt.xticks(sorted(train_df.Sex.unique()), labels)
sns.set(style='darkgrid')
plt.subplots(figsize = (15,8))
ax=sns.countplot(x='Sex',data=train_df,hue='Survived',edgecolor=(0,0,0),linewidth=2)
train_df.shape
plt.title('Passenger distribution of survived vs not-survived',fontsize=25)
plt.xlabel('Gender',fontsize=15)
plt.ylabel("# of Passenger Survived", fontsize = 15)
labels = ['Female', 'Male']
plt.xticks(sorted(train_df.Survived.unique()),labels)
leg = ax.get_legend()
leg.set_title('Survived')
legs=leg.texts
legs[0].set_text('No')
legs[1].set_text('Yes')
plt.subplots(figsize = (8,8))
ax=sns.countplot(x='Pclass',hue='Survived',data=train_df)
plt.title("Passenger Class Distribution - Survived vs Non-Survived", fontsize = 25)
leg=ax.get_legend()
leg.set_title('Survival')
legs=leg.texts

legs[0].set_text('No')
legs[1].set_text("yes")
plt.subplots(figsize=(10,8))
sns.kdeplot(train_df.loc[(train_df['Survived'] == 0),'Pclass'],shade=True,color='r',label='Not Survived')
ax=sns.kdeplot(train_df.loc[(train_df['Survived'] == 1),'Pclass'],shade=True,color='b',label='Survived' )

labels = ['First', 'Second', 'Third']
plt.xticks(sorted(train_df.Pclass.unique()),labels)
train_df.head()
train_df['family_size'] = train_df.SibSp + train_df.Parch+1
test_df['family_size'] = test_df.SibSp + test_df.Parch+1
def family_group(size):
    a = ''
    if (size <= 1):
        a = 'loner'
    elif (size <= 4):
        a = 'small'
    else:
        a = 'large'
    return a

train_df['family_group'] = train_df['family_size'].map(family_group)
test_df['family_group'] = test_df['family_size'].map(family_group)
train_df['is_alone'] = [1 if i<2 else 0 for i in train_df.family_size]
test_df['is_alone'] = [1 if i<2 else 0 for i in test_df.family_size]
train_df['child'] = [1 if i<16 else 0 for i in train_df.Age]
test_df['child'] = [1 if i<16 else 0 for i in test_df.Age]
train_df.child.value_counts()
train_df.head()
test_df.head()
train_df['calculated_fare'] = train_df.Fare/train_df.family_size
test_df['calculated_fare'] = test_df.Fare/test_df.family_size
train_df.calculated_fare.mean()
train_df.calculated_fare.mode()
def fare_group(fare):
    a= ''
    if fare <= 4:
        a = 'Very_low'
    elif fare <= 10:
        a = 'low'
    elif fare <= 20:
        a = 'mid'
    elif fare <= 45:
        a = 'high'
    else:
        a = "very_high"
    return a
train_df['fare_group'] = train_df['calculated_fare'].map(fare_group)
test_df['fare_group'] = test_df['calculated_fare'].map(fare_group)
train_df = pd.get_dummies(train_df, columns=['Title',"Pclass",'Embarked', 'family_group', 'fare_group'], drop_first=True)
test_df = pd.get_dummies(test_df, columns=['Title',"Pclass",'Embarked', 'family_group', 'fare_group'], drop_first=True)
train_df.drop(['Cabin', 'family_size','Ticket','Name', 'Fare'], axis=1, inplace=True)
test_df.drop(['Ticket','Name','family_size',"Fare",'Cabin'], axis=1, inplace=True)
