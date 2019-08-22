# Data Preprocessing

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# import the CSV data into a DataFrame
dataset = pd.read_csv('Machine Learning basics/Data Preprocessing/Data.csv')
# matriX of independent variables
X = dataset.iloc[:, :-1].values
# array of dependent variable (output)
y = dataset.iloc[:, -1].values

# handling missing data
imputer = SimpleImputer()
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# encoding categories:
# categorical data must be represented as numbers
# because ML models are mathematical
lblEncoder = LabelEncoder()
X[:, 0] = lblEncoder.fit_transform(X[:, 0])

# some categorical data doesn't have order nor hierarchy,
# it just needs a numerical format
oneEncoder = OneHotEncoder(categorical_features = [0])
X = oneEncoder.fit_transform(X).toarray()
y = lblEncoder.fit_transform(y)

# splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

# feature scaling:
# variables must be in the same range
scaler = StandardScaler()
scaler = scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)