from numpy.lib.shape_base import column_stack
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

def data_split():
    dataset = pd.read_csv('./0_Data_Generation/data/linear_classification_data.csv')
    X = dataset.iloc[ : , : -1 ].values
    Y = dataset.iloc[ : , -1 ].values
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.2, random_state = 1)
    return X_train, X_test, Y_train, Y_test

class GDA: # binary classification
    def __init__(self) -> None:
        self.phi = 0
        self.mu0 = []
        self.mu1 = []
        self.simga = [] # using same covariance matirx
        self.p_y = 0 # p(y=1)

    def fit(self, data):
        self.data = data
        # phi
        class_labels, count = np.unique(data[:,-1], return_counts= True)
        n = data.shape[0]
        self.phi = count[1]/n
        # mu
        sum_0 = np.zeros(data[:,:-1].shape[1])
        sum_1 = np.zeros(data[:,:-1].shape[1])
        for i in range(n):
            a = data[i,:-1]
            if data[i,-1] == 0:
                sum_0 += data[i,:-1]
            else:
                sum_1 += data[i,:-1]
        self.mu0 = sum_0/count[0]
        self.mu1 = sum_1/count[1]
        # sigma
        MU = np.array([])
        for i in range(n):
            if data[i,-1] == 0:
                MU = np.append(MU,self.mu0)
            else:
                MU = np.append(MU,self.mu1)
        X_MU = data[:,:-1]-MU.reshape(data[:,:-1].shape)
        self.simga = X_MU.T.dot(X_MU) / n

        self.p_y = self.phi

    def p_x_y(self, X): # p(x|y=0), p(x|y=1), p(x)
        pxy0 = (1 / np.sqrt(2 * np.pi * np.linalg.det(self.simga))) * np.exp(-0.5 * (np.transpose(X - self.mu0)).dot(np.linalg.inv(self.simga)).dot(X - self.mu0))
        pxy1 = (1 / np.sqrt(2 * np.pi * np.linalg.det(self.simga))) * np.exp(-0.5 * (np.transpose(X - self.mu1)).dot(np.linalg.inv(self.simga)).dot(X - self.mu1))
        px = pxy0*(1-self.p_y) + pxy1*self.p_y
        return pxy0, pxy1, px

    def predict(self, X):
        Y = []
        for x in X:
            pxy0, pxy1, px = self.p_x_y(x)
            py0x = pxy0*(1-self.p_y)/px
            py1x = pxy1*self.p_y/px
            if py0x>py1x:
                Y.append(0)
            else:
                Y.append(1)
        return Y


def plot(mu1, mu2, simga, Xdata, Ydata):
    plt.figure('Output')
    step = 0.1
    x = np.arange(0, 10, step)
    y = np.arange(0, 10, step)
    X, Y = np.meshgrid(x, y)

    for i, j in enumerate(np.unique(Ydata)):
        plt.scatter(Xdata[Ydata == j, 0], Xdata[Ydata == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)

    mean = [mu1, mu2]
    for k in range(2):
        Z = []
        for i in X[0]:
            z = []
            for j in Y[:,0]:
                p = np.array([j, i])
                z.append((1 / np.sqrt(2 * np.pi * np.linalg.det(simga))) * np.exp(-0.5 *
                                                                                (np.transpose(p - mean[k])).dot(np.linalg.inv(simga)).dot(p - mean[k])))
                z_arr = np.array(z)
            Z.append(z_arr)
        plt.contour(X, Y, np.array(Z).reshape(X.shape))
    plt.show()
        


if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test = data_split()
    data = np.column_stack((X_train,Y_train))
    gda_model = GDA()
    gda_model.fit(data)
    Y_pred = gda_model.predict(X_test)
    print(Y_test,Y_pred)
    plot(gda_model.mu0, gda_model.mu1, gda_model.simga, X_train, Y_train)
