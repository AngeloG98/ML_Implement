from numpy.lib.shape_base import column_stack
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def data_split():
    dataset = pd.read_csv('./0_Data_Generation/data/linear_classification_data.csv')
    X = dataset.iloc[ : , : -1 ].values
    Y = dataset.iloc[ : , -1 ].values
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.2, random_state = 1)
    return X_train, X_test, Y_train, Y_test

class Logistic_Regression:
    def __init__(self, learn_rate = 0.05, iter = 10000, epochs = 1000, batchsize = 50, gradient_descent = "bgd") -> None:
        self.lr = learn_rate
        self.gd = gradient_descent
        self.iter = iter
        self.epochs = epochs
        self.batchsize = batchsize
        self.theta = []
        self.J = []

    def compute_cost(self, X, Y, theta): # maximize log likelihood
        h_x = (1/(1 + np.exp(-X.dot(theta))))
        ones = np.ones(X.shape[0])
        return ( np.log2(h_x).T.dot(Y) + np.log2(ones-h_x).T.dot(ones-Y) ) / X.shape[0]

    def fit_model(self, X, Y):
        self.J = []
        X = np.insert(X, 0, np.ones(X.shape[0]), 1)
        self.theta = [0.001] * X.shape[1]
        # sigmoid
        if self.gd == "bgd":
            for i in range(self.iter):
                self.J.append(self.compute_cost(X, Y, self.theta))
                h_x = (1/(1 + np.exp(-X.dot(self.theta))))
                self.theta += self.lr * X.T.dot(Y - h_x) / X.shape[0]
        elif self.gd == "mgd": # mini-batch gradient ascent / stochastic gradient ascent
            XYdata = np.column_stack((X,Y))
            for j in range(self.epochs):
                np.random.shuffle(XYdata)
                mini_batches = [ XYdata[k:k+self.batchsize] for k in range(0, XYdata.shape[0], self.batchsize)] # batchsize may > datasize, but ok
                for data in mini_batches:
                    self.J.append(self.compute_cost(data[:, :-1], data[:,-1], self.theta))
                    h_x = (1/(1 + np.exp(-data[:, :-1].dot(self.theta))))
                    self.theta += self.lr * data[:, :-1].T.dot(data[:,-1] - h_x) / data.shape[0]
    
    def predict(self, X):
        X = np.insert(X, 0, np.ones(X.shape[0]), 1)
        Y = (1/(1 + np.exp(-X.dot(self.theta))))
        Y = np.where(Y>0.5, 1, 0)
        return Y

            
if __name__ == "__main__":
    #split
    X_train, X_test, Y_train, Y_test = data_split()
    
    #train
    LR = Logistic_Regression(gradient_descent = "bgd")
    LR.fit_model(X_train, Y_train)

    #predict
    Y_pred = LR.predict(X_test)
    cm = confusion_matrix(Y_test, Y_pred)
    print(cm)

    # plot
    plt.figure("linear_regression")
    ax1 = plt.subplot(211)
    ax1.axis([0,10,0,10])
    XY = column_stack((X_train,Y_train))
    for i in range(XY.shape[0]):
        if XY[i,-1] == 1:
            ax1.scatter(XY[i,0] , XY[i,1], color = 'red', marker= 'x')
        else:
            ax1.scatter(XY[i,0] , XY[i,1], color = 'blue', marker= 'o')
    ax1.plot(X_train[:,0], -X_train[:,0].dot(LR.theta[1]/LR.theta[2])-LR.theta[0]/LR.theta[2],'-',color = 'black')
    ax2 = plt.subplot(212)
    ax2.plot(LR.J)
    print(LR.J[-1])
    plt.show()