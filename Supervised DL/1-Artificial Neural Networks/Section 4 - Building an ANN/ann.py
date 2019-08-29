from pandas import read_csv
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Dense

dataset = read_csv('Supervised DL/1-Artificial Neural Networks/Section 4 - Building an ANN/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, -1].values

# encode countries to numbers
lblEncoder_countries = LabelEncoder()
X[:, 1] = lblEncoder_countries.fit_transform(X[:, 1])
# encode genders to numbers
lblEncoder_genders = LabelEncoder()
X[:, 2] = lblEncoder_genders.fit_transform(X[:, 2])
# encode countries' numbers to dummy variables
oneEncoder = OneHotEncoder(categorical_features = [1])
X = oneEncoder.fit_transform(X).toarray()
X = X[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler()
scaler = scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# initialize the neural network
classifier = Sequential()

# add layers to NN
for i in range(2):
    # create hidden layer:
    # 6 nodes = average of nodes in input and nodes in output
    # activation function: rectifier (relu)
    # uniform method to randomly initialize weights
    hidden_layer = Dense(6, activation = 'relu', kernel_initializer = 'uniform')
    classifier.add(hidden_layer)
# create output layer
# 1 node as it is binary output
# sigmoid because it will predict a probability
out_layer = Dense(1, activation = 'sigmoid', kernel_initializer = 'uniform')
classifier.add(out_layer)

# set parameters to create NN
classifier.compile('adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# fit the model into dataset
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
# predicted values as a probability
y_pred = classifier.predict(X_test)
y_truefalse = (y_pred > 0.5)
# confussion matrix to check accuracy
matrix = confusion_matrix(y_test, y_truefalse)