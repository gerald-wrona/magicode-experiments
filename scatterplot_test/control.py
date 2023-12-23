# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 10:02:15 2023

@author: wrona

"""

import pandas as pd
import seaborn as sns

data = pd.read_csv('C:\\Users\\wrona\\Documents\\upskill\\magicode\\census_data.csv')
ax = sns.scatterplot(data = data, x = "age", y = "income")
ax.set(title = 'age vs income', xlabel='age', ylabel='income')


