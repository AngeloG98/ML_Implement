from numpy.lib.shape_base import column_stack
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import cvxopt
from cvxopt import matrix, solvers

def data_split():
    dataset = pd.read_csv('./0_Data_Generation/data/nonsep-linear_svm_data.csv')
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, -1].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
    return X_train, X_test, Y_train, Y_test

class SVM:
    def __init__(self, C = None) -> None:
        self.w = []
        self.b = 0
        self.C = C

    def fit(self, X, Y):
        n_samples, n_features = X.shape
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = np.dot(X[i], X[j])
        P = matrix(np.outer(Y, Y)*K)
        q = matrix(-np.ones((n_samples, 1)))

        if self.C is None:
            G = matrix(np.negative(np.eye(n_samples)))
            h = matrix(np.zeros(n_samples))
        else:
            tmp1 = np.negative(np.eye(n_samples))
            tmp2 = np.identity(n_samples)
            G = matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = matrix(np.hstack((tmp1, tmp2)))

        b = matrix(np.zeros(1))
        A = matrix(Y.reshape(1, -1))

        solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        alphas = np.ravel(solution["x"])

        self.w = np.dot((Y * alphas).T, X)

        # Support vectors have non zero lagrange multipliers
        S = alphas > 1e-5
        index = np.arange(len(alphas))[S]
        self.alphas = alphas[S]
        self.sv_x = X[S]
        self.sv_y = Y[S]

        # bias
        for n in range(len(self.alphas)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.alphas * self.sv_y * K[index[n], S])
        self.b /= len(self.alphas)

    def predict(self, X):
        try:
            return np.sign(np.dot(X, self.w) + self.b)
        except:
            print("empty weight!")



if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test = data_split()

    svm_classifier = SVM(C=1)
    svm_classifier.fit(X_train, Y_train)
    
    Y_pred = svm_classifier.predict(X_test)
    cm = confusion_matrix(Y_test, Y_pred)
    print(cm)

    # training set
    X_set, y_set = X_train, Y_train
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                        np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, svm_classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title('SVM (Training set)')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

    #test set
    # X_set, y_set = X_test, y_test
    # X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
    #                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    # plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    #              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    # plt.xlim(X1.min(), X1.max())
    # plt.ylim(X2.min(), X2.max())
    # for i, j in enumerate(np.unique(y_set)):
    #     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
    #                 c = ListedColormap(('red', 'green'))(i), label = j)
    # plt.title('SVM (Test set)')
    # plt.xlabel('x1')
    # plt.ylabel('x2')
    # plt.legend()
    # plt.show()