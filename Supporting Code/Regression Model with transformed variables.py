# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import boxcox, normaltest, probplot
from sklearn.datasets import fetch_california_housing
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load the dataset
# Fetch the California housing data and store it in a pandas DataFrame
california_housing = fetch_california_housing()
data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
data['MedHouseVal'] = california_housing.target

# Identifying and Correcting Skewed Features
def correct_skewed_features(df):
    """
    Identifies skewed features in the DataFrame and applies a log transformation
    to correct skewness if necessary. Plots KDE plots before and after transformation.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing features.

    Returns:
    pandas.DataFrame: The DataFrame with transformed features.
    """
    for col in df.columns:
        # Plot original KDE plot
        sns.kdeplot(df[col])
        plt.title(f'KDE Plot for {col}')
        plt.show()
        
        # Check skewness and apply log transformation if necessary
        if df[col].skew() > 0.5:
            try:
                df[col] = np.log1p(df[col])
                print(f'Applied log transformation to {col}')
            except Exception as e:
                print(f'Error transforming {col}: {e}')
                
        # Plot transformed KDE plot
        sns.kdeplot(df[col])
        plt.title(f'KDE Plot for {col} after transformation')
        plt.show()
    
    return df

# Apply skew correction to the dataset
data = correct_skewed_features(data)

# Ensuring Normality for the Target Variable
def ensure_normality(target):
    """
    Ensures the target variable follows a normal distribution by applying a Box-Cox transformation.
    Plots probability plot and prints normality test results.

    Parameters:
    target (pandas.Series): The target variable.

    Returns:
    numpy.ndarray: The transformed target variable.
    """
    try:
        # Apply Box-Cox transformation
        transformed_target, _ = boxcox(target + 1)  # Adding 1 to avoid log(0)
        
        # Perform normality test
        stat, p = normaltest(transformed_target)
        print(f'Statistics={stat}, p-value={p}')
        
        if p > 0.05:
            print("Data is normally distributed")
        else:
            print("Data is not normally distributed")
        
        # Plot probability plot
        probplot(transformed_target, dist="norm", plot=plt)
        plt.show()
        
        return transformed_target
    except Exception as e:
        print(f'Error ensuring normality: {e}')
        return target

# Apply normality correction to the target variable
data['MedHouseVal'] = ensure_normality(data['MedHouseVal'])

# Creating Pair Plots for Features
def create_pair_plots(df):
    """
    Creates pair plots for the features in the DataFrame to visualize relationships
    and distributions.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing features.
    """
    try:
        sns.pairplot(df, diag_kind='kde')
        plt.show()
    except Exception as e:
        print(f'Error creating pair plots: {e}')

# Create pair plots for the dataset
create_pair_plots(data)

# Dimensionality Reduction
def perform_pca(df):
    """
    Performs Principal Component Analysis (PCA) on the DataFrame to reduce its dimensionality.
    Plots the correlation matrix heatmap and prints explained variance ratio.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing features.

    Returns:
    numpy.ndarray: The transformed features after PCA.
    """
    try:
        # Plot correlation matrix
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=True)
        plt.title('Correlation Matrix')
        plt.show()

        # Perform PCA
        pca = PCA(n_components=0.95)  # Keep 95% of variance
        pca_data = pca.fit_transform(df)
        print(f'Explained variance ratio: {pca.explained_variance_ratio_}')
        
        return pca_data
    except Exception as e:
        print(f'Error performing PCA: {e}')
        return df.values

# Split features and target variable
features = data.drop(columns=['MedHouseVal'])
target = data['MedHouseVal']

# Apply PCA to the features
pca_features = perform_pca(features)

# Train and Evaluate the Linear Regression Model
try:
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(pca_features, target, test_size=0.2, random_state=42)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict the target values for the test set
    y_pred = model.predict(X_test)

    # Calculate and print performance metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'MAE: {mae}')
    print(f'MSE: {mse}')
    print(f'R2 Score: {r2}')
except Exception as e:
    print(f'Error in training and evaluating the model: {e}')
