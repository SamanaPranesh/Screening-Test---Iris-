# Screening test for Data Science internship at Dendrite.ai
# Submitted by : Samana Pranesh, PhD Scholar, Department of Applied Mechanics, IIT Madras. Email: psamana97@gmail.com

#########################################################################################################################################

import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression, SGDRegressor, SGDClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_absolute_error, confusion_matrix
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBRegressor, XGBClassifier
import numpy as np
from sklearn.svm import SVC, SVR
import matplotlib.pyplot as plt
import seaborn as sns

# Read the json file and load it into a dictionary
with open('testing.json') as json_file:
    data = json.load(json_file)

# Access the "target" and "type of regression" field
target = data['design_state_data']['target']['target']
target_type = data['design_state_data']['target']['type']
print("Target:",target)
print("Type of Prediction:",target_type)

# Load the CSv data into a dataframe
df = pd.read_csv('iris.csv')
print(df.head())

################################################################################################################################################ 


# Iterate through the features and apply missing value imputation for each one
for feature_name, feature_details in data['design_state_data']['feature_handling'].items():
    if feature_details['is_selected']:
        # Check if the 'missing_values' key exists in feature_details
        if 'missing_values' in feature_details['feature_details']:
            impute_method = feature_details['feature_details']['missing_values']
            impute_value = feature_details['feature_details']['impute_with']

            if impute_method == 'Impute':
                if impute_value == 'Average of values':
                    df[feature_name].fillna(df[feature_name].mean(), inplace=True)
                elif impute_value == 'custom':
                    df[feature_name].fillna(feature_details['feature_details']['impute_value'], inplace=True)

# Now, the DataFrame 'df' contains missing value imputations for selected features
######################################################################################################################################################


# Feature reduction
def reduce_features(df, method, num_features, depth, num_trees, target_column_name='target_column'):
    if method == 'No Reduction':
        # No feature reduction is performed
        pass

    elif method == 'Corr with Target':
        # Convert categorical columns to numerical using cat.codes
        categorical_column = ['species']
        for column in categorical_column:
            df[column] = df[column].astype('category').cat.codes
        # compute feature correlation with the target and select the top 'num_features'
        corr_matrix = df.corr()
        target_corr = corr_matrix[target_column_name].abs().sort_values(ascending=False)
        # 'selected features' - list of names of top 'num_features' with highest absolute correlation to the target variable
        selected_features = target_corr.head(num_features).index.tolist()
        df = df[selected_features + [target_column_name]]

    elif method == 'PCA':
        x = df.drop(columns=[target_column_name,'species']) # exclude target and categorical feature
        scaler = StandardScaler()
        x_standard = scaler.fit_transform(x)
        pca = PCA(num_features)
        principalComponents = pca.fit_transform(x_standard)
        df = pd.DataFrame(data=reduce_features,columns=[f'PCA_{i}' for i in range(num_features)])

    elif method == 'Tree-based':
        categorical_column = ['species']
        for column in categorical_column:
            df[column] = df[column].astype('category').cat.codes
    # Perform tree-based feature selection (e.g., RandomForest)
        X = df.drop(columns=[target_column_name])  # Exclude the target column
        y = df[target_column_name]
    
        model = RandomForestRegressor(n_estimators=num_trees, max_depth=depth, random_state=0)
        model.fit(X, y)
        feature_importances = model.feature_importances_ # gini index
        sorted_indices = feature_importances.argsort()[::-1]
        selected_features = X.columns[sorted_indices[:num_features]].tolist()
        df = df[selected_features + [target_column_name]]  # Include the target column
        
        # Create a DataFrame to store feature names and their importances
        feature_importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': feature_importances})

        # Sort the features by importance in descending order
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        # Create a bar plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, orient='h', palette='viridis')

        # Set labels and title
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature Name')
        plt.title('Feature Importance Plot')

        # Show the plot
        plt.show()

    return df

# Get the selected feature reduction method from the JSON data
feature_reduction_method = data['design_state_data']['feature_reduction']['feature_reduction_method']

# Get the number of features to keep (if applicable)
num_features_to_keep = int(data['design_state_data']['feature_reduction']['num_of_features_to_keep'])

# Get the number of trees and depth of trees for Tree-based method
num_trees = int(data['design_state_data']['feature_reduction']['num_of_trees'])
depth_of_trees = int(data['design_state_data']['feature_reduction']['depth_of_trees'])

# Get the name of the target column from the JSON data
target_column_name = data['design_state_data']['target']['target']

# Apply feature reduction based on the selected method
df = reduce_features(df, feature_reduction_method, num_features_to_keep, depth_of_trees, num_trees, target_column_name)
print(df.head()) 

###################################################################################################################################################


# Making model objects
# Specify the regression and classification algorithms available in scikit-learn
regression_algorithms = {
    "RandomForestRegressor": RandomForestRegressor(),
    "GBTRegressor": GradientBoostingRegressor(),
    "LinearRegression": LinearRegression(),
    "RidgeRegression": Ridge(),
    "LassoRegression": Lasso(),
    "ElasticNetRegression": ElasticNet(),
    "DecisionTreeRegressor": DecisionTreeRegressor(),
    "SVM": SVR(),
    "extra_random_trees": ExtraTreesRegressor(),
    "neural_network" : MLPRegressor(),
    "xg_boost": XGBRegressor(),
    "SGD": SGDRegressor()
}

classification_algorithms = {
    "RandomForestClassifier": RandomForestClassifier(),
    "GBTClassifier": GradientBoostingClassifier(),
    "LogisticRegression": LogisticRegression(),
    "DecisionTreeClassifier" : DecisionTreeClassifier(),
    "KNN" : KNeighborsClassifier(),
    "SVM":SVC(),
    "extra_random_trees": ExtraTreesClassifier(),
    "neural_network" : MLPClassifier(),
    "xg_boost": XGBClassifier(eval_metric='logloss',use_label_encoder=False),
    "SGD": SGDClassifier()
}

# Get the selected regression algorithm(s) based on your JSON
selected_regression_algorithms = []
selected_classification_algorithms = []

for model_name, model_data in data["design_state_data"]["algorithms"].items():
    if model_data.get("is_selected", False):
        if target_type == "regression" and model_name in regression_algorithms:
            selected_regression_algorithms.append((model_name, regression_algorithms.get(model_name)))
        elif target_type == "classification" and model_name in classification_algorithms:
            selected_classification_algorithms.append((model_name, classification_algorithms.get(model_name)))

# Define a parameter grid for hyperparameter tuning for each model
param_grids = {
    "RandomForestRegressor": {
        "n_estimators": [10, 15, 20],
        "max_depth": [20, 25],
        "min_samples_leaf": [5, 10],
    },
    "GradientBoostingRegressor": {
        'n_estimators': [67,89],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 3, 4],
        'min_samples_leaf': [1, 2, 3],
        'min_iter': [30, 50, 100],          # Added min_iter
        'max_iter': [50, 100, 200],         # Added max_iter
        'min_subsample': [1, 2],         # Added min_subsample
        'max_subsample': [2, 3, 4],         # Added max_subsample
        'min_stepsize': [0.01, 0.1, 0.2],  # Added min_stepsize
        'max_stepsize': [0.1, 0.2, 0.3]    # Added max_stepsize
    },
    "LinearRegression": {
        'fit_intercept':[True,False],
        'normalize':[True,False],
        'copy_X':[True,False],
    },

    "RidgeRegression": {
        "alpha": [0.5, 0.6,0.7, 0.8],
        "max_iter" : [30,40,50],
    },
    "LassoRegression": {
        "alpha": [0.5, 0.8],
        "max_iter" : [30,40,50],
    },
    "ElasticNetRegression": {
        "alpha": [0.5, 0.8],
        "l1_ratio": [0.5, 0.8],
        "max_iter" : [30,40,50],
    },
    "DecisionTreeRegressor":{
        "max_depth":[4,5,7],
        "min_samples_split":[2,5],
        "min_samples_leaf":[12,6],
        "criterion":['mse'],
        "splitter":['best'],
        "random_state":[None],
    },
    "SVM":{
        "kernel": ["linear", "rbf", "poly", "sigmoid"],
        "C": [0.1,1,10],
        "gamma":['scale','auto'],
        "epsilon":[0.01,0.1,0.2],
        #"tol": [7],
        #"max_iter": [7]
    },
    "neural_network":{
        'hidden_layer_sizes': [(67, 89)],  
        'activation': ['relu', 'tanh', 'logistic'],  
        'alpha': [0.0001,0.001],  
        'max_iter': [1000], 
        #'tol': [0],  
        'early_stopping': [True],  
        'solver': ['adam'],  
        'shuffle': [True],  
        'learning_rate_init': [0.001],  
        'batch_size': ['auto'],  
        'beta_1': [0.9],  
        'beta_2': [0.999],  
        'epsilon': [1e-8],  
        'power_t': [0.5],  
        'momentum': [0.9],  
        'nesterovs_momentum': [False], 
         
    },
    "extra_random_trees":{
        'n_estimators':[50,100,200],
        'max_features':['auto','sqrt','log2'],
        'max_depth':[5,10,20],
        'min_samples_leaf':[1,2,4],
        'min_samples_split':[2,5,10],
        #'n_jobs':[3],
    },
    "xg_boost":{
        'n_estimators': [50,100],  
        'max_depth': [3,4,5],  
        'learning_rate': [0.01,0.1,0.2],  
        #'reg_alpha': [0, 0.1, 0.5],  
        #'reg_lambda': [0, 0.1, 0.5],  
    },
    "SGD":{
        'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
        'alpha': [0.001,0.01,0.1,1],  
        'penalty': ['l1', 'l2', 'elasticnet'],
        'max_iter': [100],  
        'tol': [56],  
        'l1_ratio': [True],
    }
    
}

# Define parameter grids for classification models
# Define parameter grids for classification models
param_grids_classification = {
    "RandomForestClassifier": {
        "n_estimators": [10, 20, 30],
        "max_depth": [20, 25],
        "min_samples_leaf": [10, 20, 30,40,50],
    },
    "GradientBoostingClassifier": {
        'n_estimators': [67,89],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 3, 4],
        'min_samples_leaf': [1, 2, 3],
        'min_iter': [30, 50, 100],          
        'max_iter': [50, 100, 200],         
        'min_subsample': [1, 2],         
        'max_subsample': [2, 3, 4],         
        'min_stepsize': [0.01, 0.1, 0.2],  
        'max_stepsize': [0.1, 0.2, 0.3]    
    },
    "LogisticRegression": {
        "C": [0.1, 1, 10],
        "penalty": ['l1', 'l2'],
        "max_iter":[30,50],
        "l1_ratio":[0.5,0.8],
    },
    "DecisionTreeClassifier":{
        'max_depth':[4,5,7],
        'min_samples_split':[2,5],
        'min_samples_leaf':[12,6],
        'criterion':["gini","entropy"],
        'splitter':['best','random'],
        'random_state':[None],
    },
    "KNN":{
        "n_neighbors":[78],
        "weights":["uniform","distance"],
        "algorithm":["auto","ball_tree"],
        "p":[1,2], 
    },
    "SVM":{
        "kernel": ["linear", "rbf", "poly", "sigmoid"],
        "C": [0.1,1,10],
        "gamma":['scale','auto'],
        #"epsilon":[0.01,0.1,0.2],
        #"tol": [7],
        #"max_iter": [7]
    },
    "neural_network":{
        'hidden_layer_sizes': [(67, 89)],  
        'activation': ['relu', 'tanh', 'logistic'],  
        'alpha': [0.0001,0.001],  
        'max_iter': [50], 
        #'tol': [0],  
        'early_stopping': [True],  
        'solver': ['adam'],  
        'shuffle': [True],  
        'learning_rate_init': [0.001],  
        'batch_size': ['auto'],  
        'beta_1': [0.9],  
        'beta_2': [0.999],  
        'epsilon': [1e-8],  
        'power_t': [0.5],  
        'momentum': [0.9],  
        'nesterovs_momentum': [False], 
        },
    "extra_random_trees":{
        'n_estimators':[50,100,200],
        'max_features':['auto','sqrt','log2'],
        'max_depth':[5,10,20],
        'min_samples_leaf':[1,2,4],
        'min_samples_split':[2,5,10],
        #'n_jobs':[3],
    },
    "xg_boost":{
        'n_estimators': [50,100],  
        'max_depth': [3,4],  
        'learning_rate': [0.01,0.1,0.2],
        'min_child_weight':[1,2,3],  
        #'reg_alpha': [0, 0.1, 0.5],  
        #'reg_lambda': [0, 0.1, 0.5],  
    },
    "SGD":{
        'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
        'alpha': [0.001,0.01,0.1,1],  
        'penalty': ['l1', 'l2', 'elasticnet'],
        'max_iter': [100],  
        'tol': [56],  
        'l1_ratio': [True],
    }
}

# Load your data 
X = df.drop(columns=[target_column_name])  # Exclude the target column
y = df[target_column_name]
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Split the data into training and testing sets using train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform hyperparameter tuning and predictions for each selected model
for model_name, model in selected_regression_algorithms:
   
    param_grid = param_grids.get(model_name, {})  # Get the parameter grid for the model
   
    # Perform GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(model, param_grid, cv=6, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
   
    # Get the best-tuned model
    best_model = grid_search.best_estimator_
   
    # Fit the best model to the training data
    best_model.fit(X_train, y_train)
   
    # Make predictions using the best model
    y_pred = best_model.predict(X_test)
   
    # Calculate and print metrics (e.g., Mean Squared Error)
    # Regression Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model: {model_name}")
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R-squared: {r2:.4f}\n")

# Perform hyperparameter tuning and predictions for selected classification models
for model_name, model in selected_classification_algorithms:

    param_grid = param_grids_classification.get(model_name, {})  # Get the parameter grid for the model
   
    # Perform GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(model, param_grid, cv=6, scoring='accuracy')
    grid_search.fit(X_train, y_train)
   
    # Get the best-tuned model
    best_model = grid_search.best_estimator_
   
    # Fit the best model to the training data
    best_model.fit(X_train, y_train)
   
    # Make predictions using the best model
    y_pred = best_model.predict(X_test)
   
    # Calculate and print metrics (e.g., Accuracy)
   # Classification Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred,average='weighted')
    recall = recall_score(y_test, y_pred,average='weighted')
    f1 = f1_score(y_test, y_pred,average='weighted')
    confusion = confusion_matrix(y_test, y_pred)

    print(f"Model: {model_name}")
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(confusion)
    print("\n")
