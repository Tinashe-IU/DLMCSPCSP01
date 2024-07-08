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
california_housing = fetch_california_housing()
data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
data['MedHouseVal'] = california_housing.target

# Identifying and Correcting Skewed Features
def correct_skewed_features(df):
    for col in df.columns:
        sns.kdeplot(df[col])
        plt.title(f'KDE Plot for {col}')
        plt.show()

        if df[col].skew() > 0.5:
            df[col] = np.log1p(df[col])
            print(f'Applied log transformation to {col}')
            
        sns.kdeplot(df[col])
        plt.title(f'KDE Plot for {col} after transformation')
        plt.show()
    return df

data = correct_skewed_features(data)

# Ensuring Normality for the Target Variable
def ensure_normality(target):
    transformed_target, _ = boxcox(target + 1)  # Adding 1 to avoid log(0)
    stat, p = normaltest(transformed_target)
    print(f'Statistics={stat}, p-value={p}')
    
    if p > 0.05:
        print("Data is normally distributed")
    else:
        print("Data is not normally distributed")
    
    probplot(transformed_target, dist="norm", plot=plt)
    plt.show()
    
    return transformed_target

data['MedHouseVal'] = ensure_normality(data['MedHouseVal'])

# Creating Pair Plots for Features
def create_pair_plots(df):
    sns.pairplot(df, diag_kind='kde')
    plt.show()

create_pair_plots(data)

# Dimensionality Reduction
def perform_pca(df):
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True)
    plt.title('Correlation Matrix')
    plt.show()

    pca = PCA(n_components=0.95)
    pca_data = pca.fit_transform(df)
    print(f'Explained variance ratio: {pca.explained_variance_ratio_}')
    return pca_data

features = data.drop(columns=['MedHouseVal'])
target = data['MedHouseVal']

pca_features = perform_pca(features)

# Train and Evaluate the Linear Regression Model
X_train, X_test, y_train, y_test = train_test_split(pca_features, target, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'R2 Score: {r2}')
