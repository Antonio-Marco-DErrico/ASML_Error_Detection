# getting residuals in double ML using CV and trying all predictive models in all_estimators() from the library sklearn.BaseEstimator
# all_estimators() is a list of tuples. all_estimators()[n][0] is the model name. all_estimators()[n][1] is the model itself
# the timeout sometimes is not respected as you would need multithreading to stop the process model.fit while it is running and not after it finished for one group.
# Also, we should allow users to set the len of the warnings, and avoid repeating the same warning of a model multiple time for each CV
# IMPORTANT. For some reason the model Quantile regression does not work... Idk why, it gets stuck in model_fit it might be that it takes too long and because the 
# model is run once before the timeout the function is not able to escape and gets stuck in a very long model fitting. This is why I esclude it in the asml code.
# if ever you will deal with time series, use sktime instead of sklearn, it also have all_estimators()
# Types of filters for function all_estimators() in sklearn: type_filter{“classifier”, “regressor”, “cluster”, “transformer”}
# estimators = all_estimators(type_filter='regressor')
# also, you could try to fit a very small batch of the model, see how much it takes, and adapt your timeout accordingly, so that it does not try to fit a big model and wait to the time out.

import panda as pd
import numpy as np
import random
import warnings
from sklearn.model_selection import KFold
from sklearn.utils import all_estimators
from concurrent.futures import ThreadPoolExecutor, TimeoutError


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


# Used to fit our model with a timeout that stops the fitting if it takes too long
def fit_model(model, X_train, y_train, timeout): 
    with ThreadPoolExecutor() as executor:
        future = executor.submit(model.fit, X_train, y_train)
        try:
            future.result(timeout=timeout)  # seconds
        except TimeoutError:
            return None
    return model

# Tries several ML models and chooses the best performing in CV R squared. Then fits it to the datasets and returns the residuals, the best model, and the best model name.
def get_residuals(X, Y, models=all_estimators(type_filter='regressor'), cv_splits=5, random_seed=random.randint(1,10000), timeout=60,display_warnings=True, print_results_table = True):
    # Initialize residuals array
    residuals = np.zeros(len(Y))
    
    # Perform cross-validation
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=random_seed)
    
    # Color codes
    BLUE = '\033[38;5;19m'
    DARK_YELLOW = '\033[38;5;3m'
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

    print(f"MODEL ----------------------------- CV R^2")
    highest_R_squared = 0
    best_model = None
    best_model_name = ""
    demeaned_data = Y - np.mean(Y)
    tss = np.sum(demeaned_data ** 2)
    results_table = []

    for name, Model in models:
        ssr_average = 0
        model_warnings = set()
        model_errors = 0

        try:
            model = Model()
        except Exception as e:
            print(f"{RED}{name:<35} {BOLD}Failed instantiating, model is likely a meta estimator:{RESET}{RED} {str(e)}{RESET}")
            results_table.append([name,'FAIL - instantiating'])
            continue

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("once")  # Capture warnings once

            # Execute cross-validation
            for train_index, test_index in kf.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
                
                try:
                    fitted_model = fit_model(model, X_train, y_train, timeout)
                    if fitted_model is None:
                        print(f"{DARK_YELLOW}{name:<35} TIMEOUT{RESET}")
                        model_errors += 1
                        results_table.append([name,'TIMEOUT'])
                        break
                    
                    y_pred = model.predict(X_test)
                    ssr_average += np.sum((y_test - y_pred) ** 2)
                except Exception as e:
                    print(f"{RED}{name:<35} {BOLD}Model fit error:{RESET}{RED} {str(e)}{RESET}")
                    model_errors += 1
                    results_table.append([name,'FAIL - Fitting'])
                    break
            if fitted_model and not model_errors:  # Calculate R^2 only if the model was fitted without errors
                R_squared = 1 - (ssr_average / tss)
                print(f"{BLUE}{name:<35} {R_squared * 100:.2f}%{RESET}")
                results_table.append([name, R_squared])
                if R_squared > highest_R_squared:
                    highest_R_squared = R_squared
                    best_model = model
                    best_model_name = name
         
            # Handle model-specific warnings after displaying R^2
            if display_warnings:
                for warning in w:
                    model_warnings.add(str(warning.message))

                if model_warnings:
                    for warning in model_warnings:
                        print(f"{RESET}Warning: {warning}\n")           
                       
    # Sorts the simplified table of results and eventually prints it (excluding warning messages, and ordering from the highest R^2 to the lowest)
    def sort_key(item):
        model_name, value = item
        if isinstance(value, float):
            # Sort percentages naturally from high to low
            return (0, -float(value))
        elif value == "TIMEOUT":
            return (1,)
        elif value == "FAIL - fitting":
            return (2,)
        elif value == "FAIL - instantiating":
            return (3,)
        else:
            return (4,)
        
    results_table.sort(key=sort_key)
        
    if print_results_table:

        print(f"{RESET}-----------------------------------------------------")
        print('RESULTS TABLE                       CV R^2')
        for model_name, value in results_table:
            
            if isinstance(value, float):
                COL = BLUE
            else:
                if "TIMEOUT" in value:
                    COL = DARK_YELLOW 
                elif "FAIL - instantiating" in value or "FAIL - fitting" in value:
                    COL = RED
            
            if isinstance(value, float):
                print(f"{COL}{model_name:<{35}} {value * 100:.2f}%")
            else:
                print(f"{COL}{model_name:<{35}} {value}")     
        
    # Calculate residuals for the best model outside cross validation
    best_model.fit(X, Y)
    y_pred = best_model.predict(X)
    residuals = Y - y_pred
    best_ssr_average = np.sum(residuals ** 2)
    best_R_squared = 1 - (best_ssr_average / tss)
    
    print(f"{RESET}-----------------------------------------------------")
    print(f"{'BEST MODEL:':<{35}} {GREEN}{BOLD}{best_model_name}{RESET}")
    print(f"{'CV R^2:':<{35}} {GREEN}{highest_R_squared * 100:.4f}%{RESET}")
    print(f"{'R^2:':<{35}} {GREEN}{best_R_squared * 100:.4f}%{RESET}")
    
    return best_model_name, best_model, y_pred, residuals, results_table
