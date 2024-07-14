# Importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the California housing dataset
# Fetches the dataset from the sklearn library and converts it into a pandas DataFrame
california = fetch_california_housing()
data = pd.DataFrame(california.data, columns=california.feature_names)
data['Target'] = california.target

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Exploratory Data Analysis (EDA)
# This section involves plotting KDE plots and generating a correlation matrix

# KDE plots for each feature and the target variable
# KDE plots help visualize the distribution of the data for each feature
for column in data.columns:
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data[column], shade=True)
    plt.title(f'KDE Plot for {column}')
    plt.show()

# Compute the correlation matrix to understand the relationships between features
corr_matrix = data.corr()

# Plotting a heatmap for the correlation matrix
# This helps visualize the strength of correlations between features
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()

# Splitting the dataset into training and testing sets
# Features are stored in X and the target variable in y
X = data.drop('Target', axis=1)
y = data['Target']
# 80% of the data is used for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training a regression model
# A Linear Regression model is instantiated and trained on the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting on the test set
# The trained model is used to make predictions on the test data
y_pred = model.predict(X_test)

# Evaluating the model
# Calculate and print performance metrics: Mean Squared Error, Mean Absolute Error, and R-squared
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')

# Visualizing actual vs predicted values
# A scatter plot to compare actual target values with the predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()

# Display residuals
# Residuals are the differences between actual and predicted values
# Plot the distribution of residuals to check for any patterns or anomalies
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.show()
