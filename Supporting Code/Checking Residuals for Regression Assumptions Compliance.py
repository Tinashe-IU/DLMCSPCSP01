# Importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm

# Load the California Housing dataset
# Fetches the California housing data and stores it in a pandas DataFrame
california_housing = fetch_california_housing()
X = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
y = pd.Series(california_housing.target)

# Split the dataset into training and test sets
# 80% of the data is used for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a linear regression model
# Instantiate the LinearRegression model and fit it to the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the test set results
# Use the trained model to predict the target values for the test data
y_pred = model.predict(X_test)

# Calculate the residuals
# Residuals are the differences between the actual and predicted target values
residuals = y_test - y_pred

# Perform the Jarque-Bera test for normality
# Tests whether the residuals are normally distributed
jb_test_stat, jb_p_value = jarque_bera(residuals)
print(f"Jarque-Bera test statistic: {jb_test_stat}")
print(f"Jarque-Bera test p-value: {jb_p_value}")

# Perform the Breusch-Pagan test for homoscedasticity
# Tests whether the residuals have constant variance (homoscedasticity)
X_test_sm = sm.add_constant(X_test)  # Add a constant term for the intercept
bp_test_stat, bp_p_value, _, _ = het_breuschpagan(residuals, X_test_sm)
print(f"Breusch-Pagan test statistic: {bp_test_stat}")
print(f"Breusch-Pagan test p-value: {bp_p_value}")

# Perform the Durbin-Watson test for serial independence
# Tests whether the residuals are uncorrelated (no autocorrelation)
dw_test_stat = durbin_watson(residuals)
print(f"Durbin-Watson test statistic: {dw_test_stat}")

# Plot the residuals
# Create a 2x2 grid of plots to visualize the residuals
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Histogram of residuals
# Plots the distribution of residuals to check for normality
sns.histplot(residuals, kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Histogram of Residuals')
axes[0, 0].set_xlabel('Residuals')
axes[0, 0].set_ylabel('Frequency')

# Q-Q plot
# Plots the quantiles of residuals against a normal distribution
# This helps to visually check if the residuals are normally distributed
sm.qqplot(residuals, line='s', ax=axes[0, 1])
axes[0, 1].set_title('Q-Q Plot')

# Residuals vs Fitted values
# Plots residuals against fitted values to check for any patterns
# Ideally, residuals should be randomly scattered around 0
sns.scatterplot(x=y_pred, y=residuals, ax=axes[1, 0])
axes[1, 0].axhline(0, ls='--', color='red')
axes[1, 0].set_title('Residuals vs Fitted Values')
axes[1, 0].set_xlabel('Fitted Values')
axes[1, 0].set_ylabel('Residuals')

# Residuals vs Time/Order
# Plots residuals in the order they were observed to check for any time-related patterns
sns.lineplot(x=np.arange(len(residuals)), y=residuals, ax=axes[1, 1])
axes[1, 1].axhline(0, ls='--', color='red')
axes[1, 1].set_title('Residuals vs Order')
axes[1, 1].set_xlabel('Order')
axes[1, 1].set_ylabel('Residuals')

# Adjust layout to prevent overlap and display the plots
plt.tight_layout()
plt.show()
