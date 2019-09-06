# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:36:47 2019

@author: jmiller
"""

# Imports
import pandas as pd
import numpy as np
from sklearn import train_test_split

# Load data and data description description
df = pd.read_csv('train.csv')
with open('Data_Description.txt') as file:
    desc = file.read()



X = df[df.columns[:-1]]
y = df[df.columns[-1]]