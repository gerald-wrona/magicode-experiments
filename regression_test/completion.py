# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 10:56:09 2023

@author: wrona
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the data
data = pd.read_csv('census_data.csv')

# Drop features with more than 20% missingness
data = data.dropna(thresh=data.shape[0]*0.8, axis=1)

# Split the data into features (X) and target (y)
X = data.drop('income', axis=1)
y = data['income']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the training accuracy
training_accuracy = model.score(X_train, y_train)

# Calculate the R-squared value
r2 = r2_score(y_test, y_pred)

print('Training accuracy:', training_accuracy)
print('R-squared value:', r2)