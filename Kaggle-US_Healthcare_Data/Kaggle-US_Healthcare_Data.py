#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 20:19:58 2019

@author: Jake
"""

# https://www.kaggle.com/maheshdadhich/us-healthcare-data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load in dataframe
df = pd.read_csv('Nutritions_US.csv', encoding = 'cp1252')

# Split short descriptions to track only large categories
descriptions = df['Shrt_Desc'].str.split(',')

categories = {}
for row in descriptions:
    if row[0] in categories:
        categories[row[0]] += 1
    else:
        categories[row[0]] = 1

large_categories = [cat for cat in categories if categories[cat] >= 25]

# Create empty dataframe of only large categories
cat = df.columns.tolist()
cat.append('Target')
new_df = pd.DataFrame(columns = cat)

# Add only rows/targets that belong to large categories
print('\n--- Creating new_df ---')
for i in range(df.shape[0]):
    if df.iloc[i]['Shrt_Desc'].split(',')[0] in large_categories:
        new_df = new_df.append(df.iloc[i])
        new_df.iloc[-1, -1] = df.iloc[i]['Shrt_Desc'].split(',')[0]

# Set features, X, y values
features = new_df.columns[2:48]
X = new_df[features]
y = new_df['Target']
        
# Split into train/test sets
print('\n--- Splitting train/test sets ---')
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.25,
                                                    random_state = 42)

# Train classifier
print('\n--- Creating Imputer ---')
imp_mean = SimpleImputer(missing_values = np.nan, strategy = 'mean')
X_train = pd.DataFrame(imp_mean.fit_transform(X_train), columns = features)
X_test = pd.DataFrame(imp_mean.fit_transform(X_test), columns = features)

print('\n--- Training Decision Tree Classifiers ---')
tree_correct = []
for i in range(1, 41):
    tree_clf = DecisionTreeClassifier(max_depth = i)
    tree_clf.fit(X_train, y_train)
    tree_predictions = tree_clf.predict(X_test)
    tree_correct.append(100 * 
                        (tree_predictions == y_test).value_counts()[1] / 
                        len(y_test))

# Plot correct predictions for Decision Tree Classifier
plt.plot(range(1, 41), tree_correct, 0, 100)
plt.xlabel('Depth of Decision Tree')
plt.ylabel('Correct Predictions (%)')
plt.title('Decision Tree Classifier')
plt.show()

# Train Random Forest Classifier
randf_correct = []
print('\n--- Training Random Forest Classifiers ---')
for i in range(1, 41):
    randf_clf = RandomForestClassifier(n_estimators = 100,
                                       max_depth = i,
                                       random_state = 42)
    randf_clf.fit(X_train, y_train)
    randf_predictions = randf_clf.predict(X_test)
    randf_correct.append(100 *
                         (randf_predictions == y_test).value_counts()[1] /
                         len(y_test))

# Plot correct predictions for Random Forest Classifier
plt.plot(range(1, 41), randf_correct, 0, 100)
plt.xlabel('Depth of Decision Tree Used')
plt.ylabel('Correct Predictions (%)')
plt.title('Random Forest Classifier - 100 Trees Used')
plt.show()