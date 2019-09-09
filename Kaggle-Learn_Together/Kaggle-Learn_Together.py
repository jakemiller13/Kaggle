# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:36:47 2019

@author: jmiller
"""

# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
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

############################
# RANDOM FOREST CLASSIFIER #
############################

# Instantiate Random Forest Classifier
rf_clf = RandomForestClassifier()

# GridSearchCV allows us to optimize parameters
rf_param_grid = [{'n_estimators': [100, 500, 1000],
                  'max_depth': [6, 8, 10],
                  'max_features': [5, 10, 20],
                  'max_leaf_nodes': [8, 16, 32]},
                 {'bootstrap': [False],
                  'n_estimators': [100, 500, 1000],
                  'max_depth': [5, 10, 20],
                  'max_features': [6, 8, 10],
                  'max_leaf_nodes': [8, 16, 32]}]

# Only run if necessary because this takes a while
def random_forest_grid_search(clf, param_grid, verbose = 0):
    '''
    Returns a Random Forest grid_search result based on "param_grid"
    '''
    # TODO you can mess around here, try different scoring
    rf_grid_search = GridSearchCV(clf,
                                  param_grid,
                                  cv = 5,
                                  scoring = 'neg_log_loss',
                                  verbose = 2)
    rf_grid_search.fit(X_train, y_train)
    return rf_grid_search

# Only run first time through the script
try:
    rf_grid_search
except NameError:
    rf_grid_search = random_forest_grid_search(rf_clf,
                                               rf_param_grid,
                                               verbose = 2)

# Get results of grid search
rf_cvres = rf_grid_search.cv_results_
for mean_score, params in zip(rf_cvres['mean_test_score'],
                              rf_cvres['params']):
    print(round(np.sqrt(-mean_score), 3), params)

print('\n--- GridSearchCV Best Parameters ---\n', rf_grid_search.best_params_)
rf_best = rf_grid_search.best_estimator_

# Export graph of a single decision tree just to see what it looks like
export_graphviz(rf_best.estimators_[0],
                out_file = 'rf_best_classifier.dot',
                feature_names = X_train.columns,
                class_names = ['1', '2', '3', '4', '5', '6', '7'],
                rounded = True,
                filled = True)

# Random Forest Accuracy
print('Random Forest Accuracy Accuracy: ' +\
      str(round((rf_best.predict(X_test) == y_test).sum() / len(y_test), 2)))

#######################
# ADABOOST CLASSIFIER #
#######################

# Instantiate AdaBoost Classifier

ada_param_grid = [{'n_estimators': [50, 200, 500],
                   'max_depth': [1, 2, 10]}]

ada_param_grid = [{#'n_estimators' : [100],
                   'base_estimator__n_estimators': [100, 500, 1000],
                   'base_estimator__max_depth': [6, 8, 10],
                   'base_estimator__max_features': [5, 10, 20],
                   'base_estimator__max_leaf_nodes': [8, 16, 32]}]

def adaboost_grid_search(clf, param_grid, verbose = 0):
    '''
    Returns a AdaBoost grid_search result based on "param_grid"
    '''
    # TODO you can mess around here, try different scoring
    
    ada_clf = AdaBoostClassifier(clf)
    
    ada_grid_search = GridSearchCV(ada_clf,
                                   param_grid,
                                   cv = 5,
                                   scoring = 'neg_log_loss',
                                   verbose = 2)
    ada_grid_search.fit(X_train, y_train)
    return ada_grid_search

try:
    ada_grid_search
except NameError:
    ada_grid_search = adaboost_grid_search(rf_clf, 
                                           ada_param_grid,
                                           verbose = 2)

# Get results of grid search
ada_cvres = ada_grid_search.cv_results_
for mean_score, params in zip(ada_cvres['mean_test_score'],
                              ada_cvres['params']):
    print(round(np.sqrt(-mean_score), 3), params)

print('\n--- GridSearchCV Best Parameters ---\n', ada_grid_search.best_params_)
ada_best = ada_grid_search.best_estimator_

# Random Forest Accuracy
print('AdaBoost Accuracy Accuracy: ' +\
      str(round((ada_best.predict(X_test) == y_test).sum() / len(y_test), 2)))