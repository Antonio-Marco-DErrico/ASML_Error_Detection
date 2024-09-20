# Imports
import ASML_functions as fn
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import Lasso, Ridge, LassoLarsCV, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

import ASML_functions as fn

import pandas as pd

# excel loading function (output: dict of numpy array, headers list)
def load_excel_to_arrays(file_path):
    # Load the Excel file
    xls = pd.ExcelFile(file_path)
    
    # Create an empty dictionary to hold sheet names and arrays
    arrays_dict = {}
    
    # Iterate through each sheet in the Excel file
    for sheet_name in xls.sheet_names:
        # Read the sheet into a DataFrame
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # Get the headers from the first row
        headers = df.columns.tolist()

        # Convert the DataFrame to a NumPy array
        array = df.to_numpy()
        
        # Add the array to the dictionary with sheet name as key
        arrays_dict[sheet_name] = array
    
    return arrays_dict, headers



# Load the data
file_path = 'ASML case data 2024 bias removed.xlsx'
arrays_dict, headers = fn.load_excel_to_arrays(file_path)

# Sample the first wafer
df_reg = arrays_dict['Wafer 1'][:, -4:]
# (OVL, L1_CD, L2_CD, OVL_L2_CD, OVL_2, L2CD_2_L1CD)

OVL_L2CD  = df_reg[:, 1] * df_reg[:, 3]
OVL_2  = df_reg[:, 1] * df_reg[:, 1]
L2CD_2_L1CD = df_reg[:, 3] * df_reg[:, 3] * df_reg[:, 2]

df_reg = np.hstack((df_reg, np.array([OVL_L2CD, OVL_2, L2CD_2_L1CD]).T))

# Load the data
file_path = 'ASML case data 2024.xlsx'
arrays_dict, headers = fn.load_excel_to_arrays(file_path)
# y = df[:, -4]

# Split data into features and target
X = df_reg[:, 1:]
y = df_reg[:, 0]

# function for testing:
# OLS, Ridge/Lasso, Adaptive Lasso, kNN, Decision Tree
def function_tester(cv_size:int = 10, loss_function:str = 'neg_mean_squared_error'):
    # Ordinary Least Squares (OLS) regression
    ols_model = LinearRegression()
    ols_scores = cross_val_score(ols_model, X, y, cv=cv_size, scoring=loss_function)

    # OLS with Ridge regression
    alpha_best = 0
    ridge_model_best = None
    ridge_scores_mean_best = 1e99
    ridge_scores_best = None
    for alpha in range(11):
        alpha_1 = alpha * 0.1
        ridge_model = Ridge(alpha=alpha_1)
        ridge_scores = cross_val_score(ridge_model, X, y, cv=cv_size, scoring=loss_function)
        if np.mean(ridge_scores) < ridge_scores_mean_best:
            alpha_best = alpha_1
            ridge_model_best = ridge_model
            ridge_scores_mean_best = np.mean(ridge_scores)
            ridge_scores_best = ridge_scores

    # OLS with Adaptive Lasso
    lasso_adaptive_model = LassoLarsCV(cv=10)
    lasso_adaptive_scores = cross_val_score(lasso_adaptive_model, X, y, cv=cv_size, scoring=loss_function)

    # k-Nearest Neighbors (KNN)
    n_neighbors_best = 1
    knn_model_best = None
    knn_scores_mean_best = 1e99
    knn_scores_best = None
    for n_neighbors_1 in range(5, 51, 5):
        knn_model = KNeighborsRegressor(n_neighbors=5)  # You can adjust the number of neighbors
        knn_scores = cross_val_score(knn_model, X, y, cv=cv_size, scoring=loss_function)
        if np.mean(knn_scores) < knn_scores_mean_best:
            n_neighbors_best = n_neighbors_1
            knn_model_best = knn_model
            knn_scores_mean_best = np.mean(knn_scores)
            knn_scores_best = knn_scores

    # Decision Tree with pruning
    max_depth_best = 0
    tree_model_best = None
    tree_scores_mean_best = 1e99
    tree_scores_best = None
    for max_depth_1 in range(5, 51, 5):
        tree_model = DecisionTreeRegressor(max_depth=max_depth_1)  # You can adjust the maximum depth
        tree_scores = cross_val_score(tree_model, X, y, cv=cv_size, scoring=loss_function)
        if np.mean(tree_scores) < tree_scores_mean_best:
            max_depth_best = max_depth_1
            tree_model_best = tree_model
            tree_scores_mean_best = np.mean(tree_scores)
            tree_scores_best = tree_scores

    return (ols_scores, ridge_scores_best, lasso_adaptive_scores, knn_scores_best, tree_scores_best)


## Custom loss function
# custom scoring function
def custom_scoring_function(y_true, y_pred, alpha):
    # Penalize values below alpha and favor values above alpha
    score = ((y_pred >= alpha) * y_true).sum()  # penalize values below alpha
    score += ((y_pred < alpha) * (y_true - alpha)).sum()  # reward values above alpha
    return score

# Create the scorer
custom_scorer = make_scorer(custom_scoring_function, alpha=0.9)


## Setting up the parameters
# Set the CV sizes and loss functions
cv_sizes = (10, 20, 50, 100)
loss_functions = ('neg_mean_squared_error', 'neg_mean_absolute_percentage_error')


## Executing the tests
# for cv_size in cv_sizes:
#     for loss_function in loss_functions:
# Obtain the methods' scores5
(ols_scores, ridge_scores_best, lasso_adaptive_scores, knn_scores_best, tree_scores_best) = function_tester(10, custom_scorer)

# Plotting the scores
plt.boxplot([ols_scores, ridge_scores_best, lasso_adaptive_scores, knn_scores_best, tree_scores_best],
            labels=['OLS', 'Ridge', 'Adaptive Lasso', 'KNN', 'Decision Tree'])
plt.ylabel("loss_function")
plt.title('Comparison of Regression Methods for CV size = ' + str(10))
plt.xticks(rotation=45)
plt.savefig('plot_cv' + str(10) + '_loss_function_' + "loss_function" + '.png')