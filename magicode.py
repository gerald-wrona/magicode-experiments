# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 10:09:08 2023

@author: wrona
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming the data is in a CSV file named 'census_data.csv'
data = pd.read_csv('C:\\Users\\wrona\\Documents\\upskill\\magicode\\census_data.csv')

# Assuming 'Age' and 'Median Income' are the columns in the data
#age = data['Age']
#income = data['Median Income']

'''
gerald: here i have fixed it
'''

age = data['age']
income = data['income']


# Create a scatter plot
sns.scatterplot(x=age, y=income)

# Add labels and title
plt.xlabel('Age')
plt.ylabel('Median Income')
plt.title('Age vs Median Income')

# Show the plot
plt.show()
