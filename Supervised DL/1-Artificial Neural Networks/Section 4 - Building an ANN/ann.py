from pandas import read_csv
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
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
# create hidden layer:
# 6 nodes = average of nodes in input + nodes in output
# activation function: rectifier (relu)
# uniform method to randomly initialize weights
hidden_layer = Dense(6, activation = 'relu', kernel_initializer = 'uniform')
classifier.add(hidden_layer)
