## Telecom Customer Churn Prediction
## Overview

Customer churn is a major challenge for telecom companies. Predicting which customers are likely to leave helps companies take preventive actions and improve customer retention.

This project builds a Machine Learning model to predict telecom customer churn using customer usage, service, and billing data.

The project includes data preprocessing, feature engineering, handling class imbalance with SMOTE, and training an XGBoost model with hyperparameter tuning.

## Problem Statement

Telecom companies lose revenue when customers leave their services.

The goal of this project is to predict whether a customer will churn or stay using historical customer data.

Target variable:

Churn
0 → Customer stays
1 → Customer leaves
## Dataset

Dataset used: Telecom Customer Dataset

The dataset contains multiple customer attributes such as:

Customer demographics

Service usage

Billing information

Call statistics

Network usage

Target column:

Churn
## Technologies Used
Programming Language : Python

Libraries

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

XGBoost

Imbalanced-learn (SMOTE)

## Project Workflow

1️⃣ Data Loading

The dataset is loaded using pandas.

import pandas as pd

df = pd.read_csv("cell2celltrain.csv")

2️⃣ Data Exploration

Basic exploration techniques were used:

df.head()

df.info()

df.describe()

df.shape

This helps understand:

Dataset structure

Missing values

Feature types

3️⃣ Data Cleaning
Removing duplicates
df.drop_duplicates(inplace=True)
Handling missing values

Numerical columns → filled with median

Categorical columns → filled with "unknown"

4️⃣ Feature Engineering

Unnecessary columns were removed.

Example:

CustomerID

Target variable converted:

Yes → 1
No → 0

5️⃣ Encoding Categorical Variables

Categorical features were converted using One-Hot Encoding.

pd.get_dummies()

6️⃣ Feature Importance

A Random Forest model was used to determine the most important features.

Top 12 important features were selected for the final model.

7️⃣ Correlation Analysis

A heatmap was used to visualize feature relationships.

sns.heatmap()

8️⃣ Train-Test Split

Dataset was split into:

Training Data → 80%
Testing Data → 20%

9️⃣ Feature Scaling

StandardScaler was applied to normalize features.

StandardScaler()

🔟 Handling Class Imbalance

The dataset was imbalanced, so SMOTE (Synthetic Minority Oversampling Technique) was used to balance the classes.

from imblearn.over_sampling import SMOTE
## Machine Learning Model
XGBoost Classifier

The main model used for prediction:

XGBClassifier

XGBoost is powerful because it:

Handles tabular data well

Works well with imbalanced datasets

Provides high performance

## Hyperparameter Tuning

To improve model performance,RandomizedSearchCV was used.

Parameters tuned:

n_estimators

max_depth

learning_rate

subsample

colsample_bytree

## Model Evaluation

The model was evaluated using:

Accuracy

Recall

F1 Score

Confusion Matrix

Example metrics:

Accuracy: 58%
Recall: 60%
## Project Pipeline
Dataset
   ->
Data Cleaning
   ->
Feature Engineering
   ->
Encoding
   ->
Feature Selection
   ->
Train-Test Split
   ->
Scaling
   ->
SMOTE Oversampling
   ->
XGBoost Model
   ->
Hyperparameter Tuning
   ->
Evaluation

