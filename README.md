# Dendrite.ai Data Science Internship Screening Test

Welcome to my submission for the Data Science internship screening test at Dendrite.ai. This repository contains the code and materials related to the test.

## Submission Details

- **Submitted by**: Samana Pranesh
- **Affiliation**: PhD student, IIT Madras

## Repository Contents

- **`iris.csv`**: This file contains the dataset under study for the screening test.

- **`testing.json`**: JSON file containing related information about data cleaning, feature reduction and modelling.

- **`screening_test.py`**: Python code for the screening test. This code is organized to be executed sequentially based on the assignment questions. The data pertaining to the questions is directly read from the JSON file to make the code as general as possible.

- **`requirements.txt`**: A file listing the Python modules and their versions required to run the code. You can use this file to set up the necessary environment.

- **`Screening Test - DS.docx`**: The assignment questions or related documentation for the screening test.

## Usage
Ensure you have the required dataset `iris.csv` and  JSON file `testing.json` available.
Replace `json_path` and `data_path` with the actual file paths on your system. 

## Table of Contents
- [Analysis Steps](#analysis-steps)
- [Note on Hyperparameter Tuning](#note-on-hyperparameter-tuning)


## Analysis Steps

1. **Identifying the Target Feature and Prediction Type:**

   The initial step of the analysis includes identifying the target feature and prediction type from the JSON file.

2. **Feature Reduction:**

   The given JSON file requires feature reduction to be carried out using a tree-based method. However, other methods like PCA, correlation with the target, or no reduction can also be performed by setting `feature_reduction_method` to the desired method name in the JSON file.

3. **Feature Importance:**

   Using the tree-based method, the importance factor of the features is calculated and plotted which is shown in the image below.
   ![Alt Text](https://github.com/SamanaPranesh/Screening-Test---Iris-/blob/main/feature_importance_plot.png?raw=true)
   


5. **Model Training and Evaluation:**

   The data is trained on different regression and classification models. The standard metrics indicating the performance of the model will be displayed on the screen.

   *Note: If the prediction type is regression, the following metrics will be displayed: Mean Absolute Error, Mean Squared Error, R-squared and Root mean squared.*
   ## Regression Metrics (if prediction type is regression):

| Model                 | Mean Squared Error | Root Mean Squared Error | Mean Absolute Error | R-squared |
|-----------------------|---------------------|--------------------------|---------------------|-----------|
| Random Forest Regressor | 0.0338            | 0.1839                   | 0.1467              | 0.9468    |
| GBT Regressor          | 0.0445            | 0.2109                   | 0.1651              | 0.9300    |
| Linear Regression      | 0.0301            | 0.1735                   | 0.1350              | 0.9526    |
| Ridge Regression       | 0.0287            | 0.1695                   | 0.1330    🏆          | 0.9548    |
| Lasso Regression       | 0.3389            | 0.5822                   | 0.5113              | 0.4668    |
| Elastic Net Regression | 0.1473            | 0.3838                   | 0.3337              | 0.7683    |
| XGBoost                | 0.0392            | 0.1980                   | 0.1517              | 0.9383    |
| Decision Tree Regressor | 0.0474          | 0.2178                   | 0.1655              | 0.9254    |
| SVM                   | 0.033              | 0.1818                   | 0.1362              | 0.9480    |
| SGD                   | 0.0312            | 0.1766                   | 0.1474              | 0.9590  🏆  |
| Extra Random Trees     | 0.0284 🏆           | 0.1686   🏆                | 0.1392              | 0.9533    |
| Neural Network         | 0.0578            | 0.2403                   | 0.1599              | 0.9091    |


   *Note: If the prediction type is classification, relevant classification metrics will be displayed such as accuracy, precision, F1-score and Confusion matrix.*


## Note on Hyperparameter Tuning

For hyperparameter tuning, some parameters for the models given in the JSON file resulted in high errors. As a result, specific parameter values have been manually entered in the code to obtain lower errors and optimize model performance. Be aware that these parameter values may differ from default settings.





