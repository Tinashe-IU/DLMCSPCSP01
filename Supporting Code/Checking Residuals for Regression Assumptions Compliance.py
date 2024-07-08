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
california_housing = fetch_california_housing()
X = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
y = pd.Series(california_housing.target)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the test set results
y_pred = model.predict(X_test)

# Calculate the residuals
residuals = y_test - y_pred

# Perform the Jarque-Bera test for normality
jb_test_stat, jb_p_value = jarque_bera(residuals)
print(f"Jarque-Bera test statistic: {jb_test_stat}")
print(f"Jarque-Bera test p-value: {jb_p_value}")

# Perform the Breusch-Pagan test for homoscedasticity
X_test_sm = sm.add_constant(X_test)  # Add a constant term for the intercept
bp_test_stat, bp_p_value, _, _ = het_breuschpagan(residuals, X_test_sm)
print(f"Breusch-Pagan test statistic: {bp_test_stat}")
print(f"Breusch-Pagan test p-value: {bp_p_value}")

# Perform the Durbin-Watson test for serial independence
dw_test_stat = durbin_watson(residuals)
print(f"Durbin-Watson test statistic: {dw_test_stat}")

# Plot the residuals
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Histogram of residuals
sns.histplot(residuals, kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Histogram of Residuals')
axes[0, 0].set_xlabel('Residuals')
axes[0, 0].set_ylabel('Frequency')

# Q-Q plot
sm.qqplot(residuals, line='s', ax=axes[0, 1])
axes[0, 1].set_title('Q-Q Plot')

# Residuals vs Fitted values
sns.scatterplot(x=y_pred, y=residuals, ax=axes[1, 0])
axes[1, 0].axhline(0, ls='--', color='red')
axes[1, 0].set_title('Residuals vs Fitted Values')
axes[1, 0].set_xlabel('Fitted Values')
axes[1, 0].set_ylabel('Residuals')

# Residuals vs Time/Order
sns.lineplot(x=np.arange(len(residuals)), y=residuals, ax=axes[1, 1])
axes[1, 1].axhline(0, ls='--', color='red')
axes[1, 1].set_title('Residuals vs Order')
axes[1, 1].set_xlabel('Order')
axes[1, 1].set_ylabel('Residuals')

plt.tight_layout()
plt.show()
