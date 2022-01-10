import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def data_split():
    dataset = pd.read_csv('./0_Data_Generation/data/nonlinear_data.csv')
    X = dataset.iloc[ : , : 1 ].values
    Y = dataset.iloc[ : , 1 ].values
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.2, random_state = 0)
    return X_train, X_test, Y_train, Y_test

class Linear_Regression:
    def __init__(self, learn_rate = 0.05, iter = 10000, epochs = 1000, batchsize = 50, if_weight = False, weight_x = 0, gradient_descent = "bgd") -> None:
        self.lr = learn_rate
        self.gd = gradient_descent
        self.iter = iter
        self.epochs = epochs
        self.batchsize = batchsize
        self.theta = []
        self.J = []
        self.if_weight = if_weight
        self.weight_x = weight_x
        self.weight = []

    def compute_cost(self, X, Y, theta, weight):
        return 0.5 * weight.dot(np.square(Y - X.dot(theta))) / X.shape[0]

    def compute_weight(self, X):
        if self.if_weight == False:
            self.weight = np.ones(X.shape[0])
        else:
            sigma = 0.8
            X_vec = X.reshape(-1)
            self.weight = np.exp( -np.square(X_vec - self.weight_x*np.ones(X.shape[0])) / (2 * sigma**2) )

    def fit_model(self, X, Y):
        self.compute_weight(X)
        self.J = []
        X = np.insert(X, 0, np.ones(X.shape[0]), 1)
        self.theta = [0.001] * X.shape[1]
        
        if self.gd == "bgd":
            weight_matrix = np.diagflat(self.weight)
            for i in range(self.iter):
                self.J.append(self.compute_cost(X, Y, self.theta, self.weight))
                self.theta += self.lr * X.T.dot(weight_matrix.dot(Y - X.dot(self.theta))) / X.shape[0]
        elif self.gd == "mgd": # mini-batch gradient ascent / stochastic gradient ascent
            XYWdata = np.column_stack(((X,Y,self.weight)))
            for j in range(self.epochs):
                np.random.shuffle(XYWdata)
                mini_batches = [ XYWdata[k:k+self.batchsize] for k in range(0, XYWdata.shape[0], self.batchsize)] # batchsize may > datasize, but ok
                for data in mini_batches:
                    weight_matrix = np.diagflat(data[:,-1])
                    self.J.append(self.compute_cost(data[:, :-2], data[:,-2], self.theta, data[:,-1]))
                    self.theta += self.lr * data[:, :-2].T.dot(weight_matrix.dot(data[:,-2] - data[:, :-2].dot(self.theta))) / data.shape[0]
    
    def predict(self, X):
        X = np.insert(X, 0, np.ones(X.shape[0]), 1)
        Y = X.dot(self.theta)
        return Y

            

if __name__ == "__main__":
    #split
    X_train, X_test, Y_train, Y_test = data_split()
    
    #train
    LR = Linear_Regression(gradient_descent = "mgd", if_weight=True, weight_x=0.5)
    LR.fit_model(X_train, Y_train)

    #predict
    Y_pred = LR.predict(X_test)

    # plot
    plt.figure("locally_weighted_linear_regression")
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

