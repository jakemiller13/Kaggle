#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 19:49:30 2019

@author: Jake
"""

'''
This dataset contains Environmental Health Inspection Results for Restaurants 
and Markets in the City of Los Angeles. Los Angeles County Environmental 
Health is responsible for inspections and enforcement activities for all 
unincorporated areas and 85 of the 88 cities in the County. This dataset is 
filtered from County data to include only facilities in the City of 
Los Angeles. The full dataset is available at 
https://data.lacounty.gov/Health/LOS-ANGELES-COUNTY-RESTAURANT-AND-MARKET-
INSPECTIO/6ni6-h5kp
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('restaurant-and-market-health-inspections.csv')

scores, score_counts = np.unique(df['score'], return_counts = True)
grades, grade_counts = np.unique(df['grade'], return_counts = True)
