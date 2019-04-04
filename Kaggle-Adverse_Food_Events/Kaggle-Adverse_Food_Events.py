#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 20:35:03 2019

@author: Jake
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('CAERS_ASCII_2004_2017Q2.csv')

print('--- DataFrame shape ---\n' + str(df.shape))

code, code_counts = np.unique(df['PRI_FDA Industry Code'],
                              return_counts = True)
plt.bar(code, code_counts)
plt.title('Code')
plt.show()

industry, industry_counts = np.unique(df['PRI_FDA Industry Name'],
                                      return_counts = True)
industry_counts_gt_200 = np.where(industry_counts > 200)
industry_gt_200 = industry[industry_counts_gt_200]

plt.bar(industry[industry_counts_gt_200],
        industry_counts[industry_counts_gt_200])
plt.title('Industry with Greater Than 200 Counts')
plt.xticks(rotation = 'vertical')
plt.show()