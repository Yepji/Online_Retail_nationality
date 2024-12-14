#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 13:29:22 2024

@author: yeji
"""

# =============================================================================
# Choose Description instead of StockCode
# =============================================================================


# =============================================================================
# Import Libraries and Classes
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, classification_report


# =============================================================================
# Import the dataset
# =============================================================================

df = pd.read_excel('/Users/yeji/Library/CloudStorage/OneDrive-성균관대학교/3-2 Tübingen/Data Science Laboratory with Python/Final_Project/Data/Online Retail.xlsx')


# =============================================================================
# EDA, Analysis
# =============================================================================

# Get basic Information of the data set
print("Column Information", df.info())

# Check how does the data set look like through first and last 5 rows
print("\nFrist and last 5 rows")
print(df.head())
print(df.tail())
print()

# Check Null values 
print("\nNumber of rows that contain Null in column 'Description' and 'CustomerID")
print(df['Description'].isnull().sum())
print(df['CustomerID'].isnull().sum())
print()

# Statistical analysis
describe_result = df.describe()
print("\nStatistical Information", describe_result)
print()

# Frequency Analysis
country_counts = df['Country'].value_counts()
print(country_counts)
country_counts_df = country_counts.reset_index() #change series to dataframe
country_counts_df.columns = ['Country', 'Count'] #change the column names

# Print the plot of Country distribution
colors = sns.color_palette('hls',len(country_counts_df['Country']))

plt.figure(figsize=(10,10))
plt.pie(country_counts_df['Count'], labels=country_counts_df['Country'], autopct='%1.1f%%', colors=colors)
plt.title('Country Distribution of Online Retail')
plt.show()

# Correlation Matrix
corr_matrix = df[['Quantity', 'UnitPrice', 'CustomerID']]
plt.figure(figsize=(10,10))
sns.heatmap(corr_matrix.corr(), annot=True, cmap='YlOrRd')
plt.title('Correlation Matrix')
plt.show()


# =============================================================================
# Preprocessing
# =============================================================================

# Processing missing values
df = df.dropna(subset=['Description', 'CustomerID'])

#Divide InvoiceData into year, month, day, hour, minute
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month
df['Day'] = df['InvoiceDate'].dt.day
df['Hour'] = df['InvoiceDate'].dt.hour
df['Minute'] = df['InvoiceDate'].dt.minute

df = df.drop('InvoiceDate', axis=1)

# Label Encoding - Description Column = Stock name
D = df['Description']
le = LabelEncoder()
encoded_D = le.fit_transform(D)

print("\nEncoded result of Description")
print(encoded_D)
print(le.classes_)
print()

df['Description'] = encoded_D


# =============================================================================
# Select features and target , Split train and test data
# =============================================================================

# Select features and target data
X = df[['Description', 'Quantity', 'UnitPrice', 'CustomerID', 'Year', 'Month', 'Day', 'Hour', 'Minute']]
Y = df['Country']

# Data scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split train and test data
(trainX, testX, trainY, testY) = train_test_split(X, Y, random_state=1, test_size=0.2, stratify=Y)


# =============================================================================
# Training Model and Evaluation - KNN
# =============================================================================

# Define the model
knn = KNeighborsClassifier()

# Parameter candidates setting
p_grid = {'n_neighbors': [1,3,5,7,9,11]}
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

# Inner grid-search CV - Finding the optimal hyperparameters
clf = GridSearchCV(estimator=knn, param_grid=p_grid, cv=inner_cv, verbose=1)

# Model training
clf.fit(trainX, trainY)

# Print the best hyperparameter
print("\nBest Estimator")
print(clf.best_estimator_)

# Predict
predictions = clf.predict(testX)

# Evaluate the model
# Accuracy Score
acc = accuracy_score(testY, predictions)
print("\nKNeighbors Accuracy Score: ", acc)


# Average F1 Score
f1_scores = f1_score(testY, predictions, average='weighted')
print(f"\nKNeighbors Weighted average F1 score: ", f1_scores)

# Classification Report
print("\nKNeighbors Classification Report:")
print(classification_report(testY, predictions))


# =============================================================================
# Training Model and Evaluation - Decision Tree
# =============================================================================

# Define the model
dt=DecisionTreeClassifier(random_state=1)

# Parameter candidates setting
p_grid = {'max_depth':[2,3,4,5,10,15,20]}
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

# Inner grid-search CV - Finding the optimal hyperparameters
clf = GridSearchCV(estimator=dt, param_grid=p_grid, cv=inner_cv, verbose=0)


# Model training
clf.fit(trainX, trainY)

# Print the best hyperparameter
print("\nDecisionTree Best Estimator")
print(clf.best_estimator_)

# Predict
predictions = clf.predict(testX)

# Evaluate the model
# Accuracy Score
acc = accuracy_score(testY, predictions)
print("\nDecisionTree Accuracy Score: ", acc)


# Average F1 Score
f1_scores = f1_score(testY, predictions, average='weighted')
print(f"\nDecisionTree Weighted average F1 score: ", f1_scores)

# Classification Report
print("\nDecisionTree Classification Report")
print(classification_report(testY, predictions))


# =============================================================================
# Training Model and Evaluation - Random Forest
# =============================================================================

# Define the model
rf=RandomForestClassifier(random_state=1)

# Parameter candidates setting
p_grid = {"n_estimators": [10,20],'max_depth':[5,10,15,20],'max_features':[1,2,3]}
inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)

# Inner grid-search CV - Finding the optimal hyperparameters
clf = GridSearchCV(estimator=rf, param_grid=p_grid, cv=inner_cv, verbose=1)


# Model training
clf.fit(trainX, trainY)

# Print the best hyperparameter
print("\nRandomForest Best Estimator")
print(clf.best_estimator_)

# Predict
predictions = clf.predict(testX)

# Evaluate the model
# Accuracy Score
acc = accuracy_score(testY, predictions)
print("\nRandomForest Accuracy Score: ", acc)


# Average F1 Score
f1_scores = f1_score(testY, predictions, average='weighted')
print(f"\nRandomForest Weighted average F1 score: {f1_scores}")

# Classification Report
print("\nRandomForest Classification Report:")
print(classification_report(testY, predictions))


# =============================================================================
# Training Model and Evaluation - Gaussian Naive Bayes
# =============================================================================

# Define the model
nb=GaussianNB()

# Model training
nb.fit(trainX,trainY)

# Predict
predictions = nb.predict(testX)

# Evaluate the model
# Accuracy Score
acc = accuracy_score(testY, predictions)
print("\nNaiveBayes Accuracy Score: ", acc)

# Average F1 Score
f1_scores = f1_score(testY, predictions, average='weighted')
print(f"\nNaiveBayes Weighted average F1 score: {f1_scores}")

# Classification Report
print("\nClassification Report:")
print(classification_report(testY, predictions))




