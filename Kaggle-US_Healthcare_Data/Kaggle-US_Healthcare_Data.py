#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 20:19:58 2019

@author: Jake
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
for i in range(df.shape[0]):
    if df.iloc[i]['Shrt_Desc'].split(',')[0] in large_categories:
        new_df = new_df.append(df.iloc[i])
        new_df.iloc[-1]['Target'] = df.iloc[i]['Shrt_Desc'].split(',')[0]

'''
new_df.iloc[-1]['Target'] = 'test'
__main__:1: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation:
http://pandas.pydata.org/pandas-docs/stable/indexing.html#
indexing-view-versus-copy
'''