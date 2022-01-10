import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def data_split():
    dataset = pd.read_csv('./0_Data_Generation/data/linear_data.csv')
    X = dataset.iloc[ : , : 1 ].values
    Y = dataset.iloc[ : , 1 ].values
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.2, random_state = 0)
    return X_train, X_test, Y_train, Y_test

class Linear_Regression:
    def __init__(self, learn_rate = 0.05, iter = 10000, epochs = 1000, batchsize = 50, gradient_descent = "bgd") -> None:
        self.lr = learn_rate
        self.gd = gradient_descent
        self.iter = iter
        self.epochs = epochs
        self.batchsize = batchsize
        self.theta = []
        self.J = []

    def compute_cost(self, X, Y, theta):
        return 0.5 * np.sum(np.square(Y - X.dot(theta))) / X.shape[0]

    def fit_model(self, X, Y):
        self.J = []
        X = np.insert(X, 0, np.ones(X.shape[0]), 1)
        self.theta = [0.001] * X.shape[1]
        if self.gd == "bgd":
            for i in range(self.iter):
                self.J.append(self.compute_cost(X, Y, self.theta))
                self.theta += self.lr * X.T.dot(Y - X.dot(self.theta)) / X.shape[0]
        elif self.gd == "mgd": # mini-batch gradient ascent / stochastic gradient ascent
            XYdata = np.column_stack((X,Y))
            for j in range(self.epochs):
                np.random.shuffle(XYdata)
                mini_batches = [ XYdata[k:k+self.batchsize] for k in range(0, XYdata.shape[0], self.batchsize)] # batchsize may > datasize, but ok
                for data in mini_batches:
                    self.J.append(self.compute_cost(data[:, :-1], data[:,-1], self.theta))
                    self.theta += self.lr * data[:, :-1].T.dot(data[:,-1] - data[:, :-1].dot(self.theta)) / data.shape[0]
    
    def predict(self, X):
        X = np.insert(X, 0, np.ones(X.shape[0]), 1)
        Y = X.dot(self.theta)
        return Y

            

if __name__ == "__main__":
    #split
    X_train, X_test, Y_train, Y_test = data_split()
    
    #train
    LR = Linear_Regression(gradient_descent = "bgd")
    LR.fit_model(X_train, Y_train)

    #predict
    Y_pred = LR.predict(X_test)

    # plot
    plt.figure("linear_regression")
    ax1 = plt.subplot(211)
    ax1.axis([0,10,0,10])
    ax1.scatter(X_train , Y_train, color = 'red', marker= 'x')
    ax1.plot(X_train, X_train.dot(LR.theta[1])+LR.theta[0],'-',color = 'black')
    ax1.scatter(X_test , Y_test, color = 'blue', marker= 'x')
    ax1.scatter(X_test , Y_pred, color = 'green', marker= 'x')
    ax2 = plt.subplot(212)
    ax2.plot(LR.J)
    print(LR.J[-1])
    plt.show()

