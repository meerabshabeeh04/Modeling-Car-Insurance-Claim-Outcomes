import pandas as pd
import numpy as np
from statsmodels.formula.api import logit
data = pd.read_csv('car_insurance.csv')
print(data.head())
print(data.info())
print(data.describe())
# Check for missing values in each column
print(data.isna().sum())
# Fill missing values in 'credit_score' column with the mean value of the column
data['credit_score'].fillna(data['credit_score'].mean(), inplace=True)
# Fill missing values in 'annual_mileage' column with the mean value of the column
data['annual_mileage'].fillna(data['annual_mileage'].mean(), inplace=True)
# Confirm that there are no more missing values
print(data.isna().sum())
# Initialize an empty list to store logistic regression models for each feature
models = []
# Define the feature columns, excluding 'id' and 'outcome' columns
features = data.drop(['id', 'outcome'], axis=1).columns
print(features)
# Loop through each feature, fitting a logistic regression model with 'outcome' as the target variable
for feature in features:
    model = logit(f"{'outcome'} ~ {feature}", data=data).fit()
    models.append(model)
# Print the list of fitted models
print(models)
# Initialize a list to store the accuracy of each model
accuracies = []
# Calculate the accuracy for each model using the confusion matrix
for i in range(len(models)):
    conf_matrix = models[i].pred_table()
    TN = conf_matrix[0, 0]  # True Negatives
    TP = conf_matrix[1, 1]  # True Positives
    FN = conf_matrix[1, 0]  # False Negatives
    FP = conf_matrix[0, 1]  # False Positives
    acc = (TN + TP) / (TN + TP + FN + FP)  # Calculate accuracy
    accuracies.append(acc)
# Find the index of the feature with the highest accuracy
largest_score = accuracies.index(max(accuracies))
print(largest_score)
# Create a DataFrame to display the best feature and its accuracy
best_feature_df = pd.DataFrame({'best_feature': features[largest_score], 'best_accuracy': accuracies[largest_score]}, index=[0])
print(best_feature_df)
