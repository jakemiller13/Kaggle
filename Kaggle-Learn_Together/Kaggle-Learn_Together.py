# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:36:47 2019

@author: jmiller
"""

# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load data and data description description
train = pd.read_csv('./Data/train.csv')
test = pd.read_csv('./Data/test.csv')
with open('./Data/Data_Description.txt') as file:
    desc = file.read()

# No need to carry ID through model
train, test = train.drop('Id', axis = 1), test.drop('Id', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(train[train.columns[:-1]],
                                                    train[train.columns[-1]],
                                                    test_size = .2,
                                                    random_state = 42)

# Standard scaler on columns that are NOT one-hot encoded
std_scaler = StandardScaler()
train[train.columns[:10]] = std_scaler.fit_transform(train[train.columns[:10]])
test[test.columns[:10]] = std_scaler.fit_transform(test[test.columns[:10]])

# Instantiate RandomForestClassifier
forest_cl = RandomForestClassifier()

# GridSearchCV allows us to optimize parameters
param_grid = [{'n_estimators': [2, 5, 10, 25, 50],
               'max_features': [2, 4, 6, 8, 10]},
              {'bootstrap': [False],
               'n_estimators': [2, 5, 10, 25, 50],
               'max_features': [2, 4, 6, 8, 10]}]

# TODO you can mess around here, try different scoring
grid_search = GridSearchCV(forest_cl,
                           param_grid,
                           cv = 5,
                           scoring = 'neg_mean_squared_error')
grid_search.fit(X_train, y_train)
print('--- GridSearchCV Best Parameters ---\n', grid_search.best_params_)