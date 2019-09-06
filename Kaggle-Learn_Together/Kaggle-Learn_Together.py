# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:36:47 2019

@author: jmiller
"""

# Imports
import pandas as pd
import numpy as np

# Load data and data description description
df = pd.read_csv('train.csv')
with open('Data_Description.txt') as file:
    desc = file.read()