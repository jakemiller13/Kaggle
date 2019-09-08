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
from sklearn.tree import export_graphviz

# Load data and data description description
train = pd.read_csv('./Data/train.csv')
test = pd.read_csv('./Data/test.csv')
with open('./Data/Data_Description.txt') as file:
    desc = file.read()

# No need to carry ID through model
train, test = train.drop('Id', axis = 1), test.drop('Id', axis = 1)

# Standard scaler on columns that are NOT one-hot encoded
std_scaler = StandardScaler()
train[train.columns[:10]] = std_scaler.fit_transform(train[train.columns[:10]])
test[test.columns[:10]] = std_scaler.fit_transform(test[test.columns[:10]])

X_train, X_test, y_train, y_test = train_test_split(train[train.columns[:-1]],
                                                    train[train.columns[-1]],
                                                    test_size = .2,
                                                    random_state = 42)

# Instantiate RandomForestClassifier
forest_cl = RandomForestClassifier()

# GridSearchCV allows us to optimize parameters
param_grid = [{'n_estimators': [100, 500, 1000],
               'max_depth': [6, 8, 10],
               'max_features': [5, 10, 20],
               'max_leaf_nodes': [8, 16, 32]},
              {'bootstrap': [False],
               'n_estimators': [100, 500, 1000],
               'max_depth': [5, 10, 20],
               'max_features': [6, 8, 10],
               'max_leaf_nodes': [8, 16, 32]}]

# Only run if necessary because this takes a while
def run_grid_search(param_grid, verbose = 0):
    '''
    Returns a grid_search result based on "param_grid"
    '''
    # TODO you can mess around here, try different scoring
    grid_search = GridSearchCV(forest_cl,
                               param_grid,
                               cv = 5,
                               scoring = 'neg_log_loss',
                               verbose = 2)
    grid_search.fit(X_train, y_train)
    return grid_search

try:
    grid_search
except NameError:
    grid_search = run_grid_search(param_grid, verbose = 2)

# Get results of grid search
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(round(np.sqrt(-mean_score), 3), params)

print('\n--- GridSearchCV Best Parameters ---\n', grid_search.best_params_)
best = grid_search.best_estimator_

# Export graph of best estimator
export_graphviz(best,
                out_file = 'best_classifier.dot',
                feature_names = X_train.columns,
                class_names = ['1', '2', '3', '4', '5', '6', '7'],
                rounded = True,
                filled = True)