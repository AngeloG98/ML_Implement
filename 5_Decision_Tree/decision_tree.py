# CART / continuous features / multi-classification

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from random import sample

def data_split():
    dataset = pd.read_csv('./0_Data_Generation/data/multiclass_data.csv')
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, -1].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, Y_train, Y_test


class Decision_Tree:
    def __init__(self, min_data = 2, max_depth = 5) -> None:
        self.feature = []
        self.min_data = min_data
        self.max_depth = max_depth

    def splitsubdata(self, data, idx, threshold):
        left_data = []
        right_data = []
        for i in range(data.shape[0]):
            t = data[i][idx]
            if data[i][idx] <= threshold:
                left_data.append(data[i])
            else:
                right_data.append(data[i])
        return np.array(left_data), np.array(right_data)


    def computegini(self, data):
        gini = 1
        y_cnt = {}
        if data.shape[0] == 0:
            return 0
        else:
            for y in data[:, -1]:
                y_cnt[y] = y_cnt.get(y, 0) + 1
            for y in y_cnt:
                gini -= (y_cnt[y]/data.shape[0])**2
            return gini

    def choose_feature_th(self, data): # if not leaf node --- split
        bestgini = 999.0
        bestfeature = -1
        bestthreshold = 0
        best_left_data = np.array([])
        best_right_data = np.array([])
        for idx in self.feature_list:
            value_list = list(data[:, idx])
            value_list.sort()
            threshold_list = [ (value_list[i]+value_list[i+1])/2 for i in range(len(value_list)-1) ]
            for th in threshold_list:
                gini = 0.0
                left_data, right_data = self.splitsubdata(data, idx, th)
                p_left = left_data.shape[0]/data.shape[0]
                p_right = right_data.shape[0]/data.shape[0]
                gini = p_left*self.computegini(left_data) + p_right*self.computegini(right_data)
                if gini < bestgini:
                    bestgini = gini
                    bestfeature = idx
                    bestthreshold = th
                    best_left_data = left_data
                    best_right_data = right_data
        node = {}
        node['feature'] = bestfeature
        node['threshold'] = bestthreshold
        node['groups'] = best_left_data, best_right_data
        if best_left_data.size == 0 or best_right_data.size == 0:
            return self.leafnode(data), 1
        else:
            return node, 0

    def leafnode(self, data): # if leaf node --- choose max class
        class_labels, count = np.unique(data[:,-1], return_counts= True)
        return class_labels[np.argmax(count)]

    def growCARTtree(self, node, depth):
        #end
        left_data , right_data = node['groups']

        if depth >= self.max_depth:
            node['left'] = self.leafnode(left_data)
            node['right'] = self.leafnode(right_data)
            return

        if len(set(left_data[:,-1])) == 1:
            node["left"] = self.leafnode(left_data)
        elif left_data.shape[0] <= self.min_data:
            node["left"] = self.leafnode(left_data)
        else:
            node["left"], tag = self.choose_feature_th(left_data)
            if tag == 0:
                self.growCARTtree(node["left"],depth+1)
                
        if len(set(right_data[:,-1])) == 1:
                node["right"] = self.leafnode(right_data)
        elif right_data.shape[0] <= self.min_data:
            node["right"] = self.leafnode(right_data)
        else:
            node["right"], tag = self.choose_feature_th(right_data)
            if tag == 0:
                self.growCARTtree(node["right"],depth+1)

    def fit(self, data, k=None):
        if k == None:
            self.feature_list = list(range(data[:,:-1].shape[1]))
        else:
            self.feature_list = sample(range(data[:,:-1].shape[1]), k)
        self.root, tag = self.choose_feature_th(data)
        self.growCARTtree(self.root, 1)
        return self.root

    def searchtree(self, node, xi):
        if xi[node['feature']] < node['threshold']:
            if isinstance(node['left'], dict):
                return self.searchtree(node['left'], xi)
            else:
                return node['left']
        else:
            if isinstance(node['right'],dict):
                return self.searchtree(node['right'],xi)
            else:
                return node['right']

    def predict(self, X):
        y_pred = np.array([])
        for xi in X:
            y_pred = np.append(y_pred,self.searchtree(self.root,xi))
        return y_pred

    def predict_single(self, x):
        y_pred = self.searchtree(self.root,x)
        return y_pred

        
if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test = data_split()
    data = np.column_stack((X_train,Y_train))
    dt_classifier = Decision_Tree()
    dt_classifier.fit(data)
    print(Y_test, dt_classifier.predict(X_test))

    # Training set
    X_set, y_set = X_train, Y_train
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.05),
                        np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.05))
    plt.contourf(X1, X2, dt_classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
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

    # Test set
    # X_set, y_set = X_test, Y_test
    # X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.05),
    #                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.05))
    # plt.contourf(X1, X2, dt_classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    #              alpha = 0.75, cmap = ListedColormap(('red', 'green', "blue")))
    # plt.xlim(X1.min(), X1.max())
    # plt.ylim(X2.min(), X2.max())
    # for i, j in enumerate(np.unique(y_set)):
    #     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
    #                 c = ListedColormap(('red', 'green', "blue"))(i), label = j)
    # plt.title('Decision Tree Classification (Training set)')
    # plt.xlabel('x1')
    # plt.ylabel('x2')
    # plt.legend()
    # plt.show()
