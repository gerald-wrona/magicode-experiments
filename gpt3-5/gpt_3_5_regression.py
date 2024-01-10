import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load census data
census_data = pd.read_csv('your_census_data.csv')

# Drop features with more than 20% missingness
missing_threshold = 0.2
census_data = census_data.dropna(thresh=(1 - missing_threshold) * census_data.shape[0], axis=1)

# Assuming 'age' and 'income' are relevant columns
selected_features = ['age']
target_variable = 'income'

# Split data into training and testing sets
train_data, test_data = train_test_split(census_data, test_size=0.2, random_state=42)

# Extract features and target variable from training set
X_train = train_data[selected_features]
y_train = train_data[target_variable]

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict income on the training set
y_train_pred = model.predict(X_train)

# Evaluate model performance on the training set
train_accuracy = model.score(X_train, y_train)
r_squared_value = r2_score(y_train, y_train_pred)

# Report results
print(f'Training Accuracy: {train_accuracy}')
print(f'R-squared Value: {r_squared_value}')
