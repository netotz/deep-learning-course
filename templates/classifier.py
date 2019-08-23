from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

class DataPreprocessor(object):
    '''Preprocesses necessary objects and variables from a CSV file.
    '''
    def __init__(self, csv_path, independ_vars):
        # import the CSV data into a DataFrame
        self.dataset = read_csv(csv_path)
        self.X = self.dataset.iloc[:, independ_vars].values
        self.y = self.dataset.iloc[:, -1].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)

        # feature scaling
        self.scaler = StandardScaler()
        self.scaler = self.scaler.fit(self.X_train)
        self.X_train = self.scaler.transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        # fitting logistic regression to training set
        self.classifier = LogisticRegression()
        self.classifier.fit(self.X_train, self.y_train)
        # predicted values based on self.X_test
        self.y_predicted = self.classifier.predict(self.X_test)
        self.matrix = confusion_matrix(self.y_test, self.y_predicted)

class Classifier(DataPreprocessor):
    '''Classification template which inherits attributes from DataPreprocessor, and has a function that plots the prediction.
    '''
    def __init__(self, csv_path, independ_vars):
        super().__init__(csv_path, independ_vars)

    def plotPrediction(self, X_set, y_set, title = "Classifier", x_label = "", y_label = ""):
        '''Plots the prediction areas and the real observations.
        '''
        # prepare each pixel
        age, salary = np.meshgrid(
            np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1, 0.01),
            np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, 0.01)
        )
        arr = np.array([age.ravel(), salary.ravel()]).T
        # colors for the background
        colors_back = ListedColormap(('red', 'green'))
        # colors for the points
        colors_points = ListedColormap(('yellow', 'blue'))
        # apply the classifier to every pixel
        plt.contourf(age, salary, self.classifier.predict(arr).reshape(age.shape), alpha = 0.75, cmap = colors_back)
        # axes' limits
        plt.xlim(age.min(), age.max())
        plt.ylim(salary.min(), salary.max())
        # plot data points (real values)
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = colors_points(i), label = j)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.show()