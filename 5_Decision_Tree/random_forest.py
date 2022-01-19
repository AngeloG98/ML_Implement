#Not well performed, waiting for update...

import imp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler

from decision_tree import Decision_Tree

def data_scaler():
    dataset = pd.read_csv('./0_Data_Generation/data/multiclass_data.csv')
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, -1].values

    sc = StandardScaler()
    X = sc.fit_transform(X)
    return X, Y, sc

class Random_Forest:
    def __init__(self, forest_size = 10) -> None:
        self.fs = forest_size

    def bootstrap_sampling(self, data):
        sample = np.array([])
        sample_i = np.random.choice(data.shape[0],size=data.shape[0])
        for i in range(len(sample_i)):
            sample = np.append(sample, data[sample_i[i]])
        return sample.reshape(data.shape)
    
    def fit(self, data):
        self.forest = []
        k = round(np.log2(data[:,:-1].shape[1]))
        for i in range(self.fs):
            sample_data = self.bootstrap_sampling(data)
            tree = Decision_Tree()
            tree.fit(sample_data,1)
            self.forest.append(tree)
        print()

    def predict(self, X):
        y_pred = np.array([])
        for x in X:
            pred_list = []
            for i in range(self.fs):
                pred_list.append(self.forest[i].predict_single(x))
            labels, count = np.unique(pred_list, return_counts= True)
            pred_vote = labels[np.argmax(count)]
            y_pred = np.append(y_pred, pred_vote)
        return y_pred


if __name__ == "__main__":
    X, Y, sc = data_scaler()
    data = np.column_stack((X,Y))
    rf_classifier = Random_Forest()
    rf_classifier.fit(data)
    Y_pred = rf_classifier.predict(X)
    err = Y-Y_pred
    print()

    # Training set
    X_set, y_set = X, Y
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.05),
                        np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.05))
    plt.contourf(X1, X2, rf_classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha = 0.75, cmap = ListedColormap(('red', 'green', "blue")))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green', "blue"))(i), label = j)
    plt.title('Decision Tree Classification (Training set)')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()
