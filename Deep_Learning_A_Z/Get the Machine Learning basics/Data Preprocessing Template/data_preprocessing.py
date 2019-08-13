# Data Preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

# Importing the dataset
dataset = pd.read_csv('Deep_Learning_A_Z/Get the Machine Learning basics/Data Preprocessing Template/Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# handling missing data
imputer = SimpleImputer()
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# encoding categories
lblEncoder = LabelEncoder()
X[:, 0] = lblEncoder.fit_transform(X[:, 0])

oneEncoder = OneHotEncoder(categorical_features = [0])
X = oneEncoder.fit_transform(X).toarray()
y = lblEncoder.fit_transform(y)

# splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y)