from numpy.lib.shape_base import column_stack
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

def data_split():
    dataset = pd.read_csv('./0_Data_Generation/data/knn_data.csv')
    X = dataset.iloc[ : , : -1 ].values
    Y = dataset.iloc[ : , -1 ].values
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.2, random_state = 1)
    return X_train, X_test, Y_train, Y_test


class knn:
    def __init__(self, train_data) -> None:
        self.train_data = train_data

    def L1dist(self, x):
        dist = []
        for data in self.train_data:
            d = np.sum(abs(x-data[:-1]))
            dist.append(d)
        return np.array(dist)

    def L2dist(self, x):
        dist = []
        for data in self.train_data:
            d = (x-data[:-1]).T.dot(x-data[:-1])
            dist.append(d)
        return np.array(dist)
            
    def predict(self, X_test, k = 5, dist_type = 2):
        if dist_type == 1:
            distfunc = self.L1dist
        elif dist_type == 2:
            distfunc = self.L2dist
        Y_pred = []
        for x in X_test:
            dist_arr = distfunc(x)
            min_i = dist_arr.argsort()
            label_list = []
            for i in range(k):
                label_list.append(self.train_data[:,-1][min_i[i]])
            class_labels, count = np.unique(label_list, return_counts= True)
            Y_pred.append(class_labels[np.argmax(count)])

        return np.array(Y_pred)


if __name__ == "__main__":
    #split
    X_train, X_test, Y_train, Y_test = data_split()
    knn_classifier = knn(np.column_stack((X_train,Y_train)))
    Y_pred = knn_classifier.predict(X_test)
    print(Y_test,Y_pred)

    # Training set
    X_set, y_set = X_train, Y_train
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.1),
                        np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.1))
    plt.contourf(X1, X2, knn_classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha = 0.75, cmap = ListedColormap(('red', 'green', "blue")))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green', "blue"))(i), label = j)
    plt.title('KNN (Training set)')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

    # Test set
    # X_set, y_set = X_test, Y_test
    # X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.05),
    #                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.05))
    # plt.contourf(X1, X2, knn_classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    #              alpha = 0.75, cmap = ListedColormap(('red', 'green', "blue")))
    # plt.xlim(X1.min(), X1.max())
    # plt.ylim(X2.min(), X2.max())
    # for i, j in enumerate(np.unique(y_set)):
    #     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
    #                 c = ListedColormap(('red', 'green', "blue"))(i), label = j)
    # plt.title('KNN (Training set)')
    # plt.xlabel('x1')
    # plt.ylabel('x2')
    # plt.legend()
    # plt.show()