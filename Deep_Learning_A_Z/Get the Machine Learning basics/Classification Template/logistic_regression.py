import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

# import the CSV data into a DataFrame
dataset = pd.read_csv('Deep_Learning_A_Z/Get the Machine Learning basics/Classification Template/Social_Network_Ads.csv')
X = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y)

# feature scaling
scaler = StandardScaler()
scaler = scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# fitting logistic regression to training set
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
# predicted values based on X_test
y_predicted = classifier.predict(X_test)
matrix = confusion_matrix(y_test, y_predicted)

def plotPrediction(X_set, y_set):
    age, salary = np.meshgrid(
        np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1, 0.01),
        np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, 0.01)
    )
    arr = np.array([age.ravel(), salary.ravel()]).T
    colors_back = ListedColormap(('red', 'green'))
    colors_points = ListedColormap(('yellow', 'blue'))

    plt.contourf(age, salary, classifier.predict(arr).reshape(age.shape), alpha = 0.75, cmap = colors_back)
    plt.xlim(age.min(), age.max())
    plt.ylim(salary.min(), salary.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = colors_points(i), label = j)
    plt.title('Logistic Regression')
    plt.xlabel('Age')
    plt.ylabel('Estimated salary')
    plt.legend()
    plt.show()

plotPrediction(X_train, y_train)
plotPrediction(X_test, y_test)