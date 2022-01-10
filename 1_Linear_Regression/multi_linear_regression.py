import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

def data_process():
    dataset = pd.read_csv('./0_Data_Generation/data/50_Startups.csv')
    X = dataset.iloc[ : , :-1 ].values
    Y = dataset.iloc[ : , -1 ].values
    labelencoder = LabelEncoder()
    X[: , 3] = labelencoder.fit_transform(X[ : , 3])

    ct = ColumnTransformer([("City",OneHotEncoder(),[3])], remainder='passthrough')
    X = ct.fit_transform(X)
    X = X.astype(np.float64) # change dtype from object to float64
    X = X[: , 1:]

    X = np.column_stack((X[:,0:2],featureNormalize(X)[0]))
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.2, random_state = 0)
    return X_train, X_test, Y_train, Y_test

def featureNormalize(X):
    X_norm = X
    mu = np.zeros((1,X.shape[1]))
    sigma = np.zeros((1,X.shape[1]))
    
    mu = np.mean(X,axis=0) # mean
    sigma = np.std(X,axis=0) # std
    X_norm = (X - mu) / sigma
    
    return X_norm, mu, sigma 

class Linear_Regression:
    def __init__(self, learn_rate = 0.05, iter = 1000, epochs = 1000, batchsize = 10, gradient_descent = "bgd") -> None:
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
    # split
    X_train, X_test, Y_train, Y_test = data_process()
    
    # train
    LR = Linear_Regression(gradient_descent = "mgd")
    LR.fit_model(X_train, Y_train)

    # predict
    Y_pred = LR.predict(X_test)
    # comparw predict with test
    print(np.column_stack((Y_pred,Y_test)))

    # plot
    plt.figure("mutli_linear_regression_cost")
    plt.plot(LR.J)
    plt.show()


