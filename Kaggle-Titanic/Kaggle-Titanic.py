# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 15:43:20 2019

@author: jmiller
"""

import pandas as pd
from sklearn import neighbors

# Comment this out if you don't want to print all columns
pd.set_option('display.expand_frame_repr', False)

n_neighbors = 2

# Load data, shuffle rows
df = pd.read_csv('train.csv')
df = df.sample(frac = 1)
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
target = ['Survived']

# Create train_df with 70% of df, test_df with 30%
train_df = df.loc[:int(df.shape[0] * .7), features + target]
valid_df = df.loc[int(df.shape[0] * .7):, features + target]

# Convert passenger sex to integers for classifier. Drop NaN
for each in [train_df, valid_df]:
    each.replace('male', 0, inplace = True)
    each.replace('female', 1, inplace = True)
    each.dropna(inplace = True)

# Features, targets for training
X = train_df[features]
y = train_df['Survived']

for weights in ['uniform', 'distance']:
    
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights = weights)
    clf.fit(X, y)
    
    Z = clf.predict(valid_df[features])
    correct = (Z == valid_df['Survived']).value_counts()[1]
    accuracy = correct / valid_df.shape[0]
    
    print('Accuracy of ' + weights + ': ' +\
          str(round(accuracy * 100, 2)) + '%')