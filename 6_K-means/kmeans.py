from numpy.random.mtrand import sample
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from random import sample


def plot(X, centroids):
    plt.figure("k_means")
    plt.xlim(X[:,0].min()-1, X[:,0].max()+1)
    plt.ylim(X[:,1].min()-1, X[:,1].max()+1)
    plt.scatter(X[:,0] , X[:,1], color = 'blue')
    centroids_1 = centroids[:,:2]
    centroids_2 = centroids[:,2:4]
    centroids_3 = centroids[:,4:]
    plt.plot(centroids_1[:,0] , centroids_1[:,1], '--', color = 'red')
    plt.scatter(centroids_1[:,0] , centroids_1[:,1], marker='x', color = 'red')
    plt.plot(centroids_2[:,0] , centroids_2[:,1], '--', color = 'yellow')
    plt.scatter(centroids_2[:,0] , centroids_2[:,1], marker='x', color = 'yellow')
    plt.plot(centroids_3[:,0] , centroids_3[:,1], '--',  color = 'green')
    plt.scatter(centroids_3[:,0] , centroids_3[:,1], marker='x', color = 'green')
    plt.show()

class kmeans():
    def __init__(self, k = 3, iter = 10) -> None:
        self.k = k
        self.iter = iter
        self.centroids_list = np.array([])

    def random_centroids(self, X):
        centroids = []
        x_c = [[0 for col in range(self.k)] for row in range(X.shape[1])]
        for i in range(X.shape[1]):
            x_c[i] = sample(range(round(X[:,1].min()-1), round(X[:,0].max()+1)), self.k)
        for j in range(self.k):
            for i in range(X.shape[1]):
                centroids.append(x_c[i][j])
        return np.array(centroids)

    def find_closest_centroids(self, X, centroids):
        idxs = []
        centroids = centroids.reshape(self.k,X.shape[1])
        for x in X:
            mindist = 999999
            idx = 0
            for i in range(centroids.shape[0]):
                dist = np.sum(np.square(x-centroids[i]))
                if dist<mindist:
                    mindist = dist
                    idx = i
            idxs.append(idx)
        return idxs

    def compute_centroids(self, X, idxs):
        centroids = []
        subXX = []
        for idx in range(self.k):
            subX = []
            subX.append(np.array([X[j] for j in range(X.shape[0]) if idxs[j] == idx]))
            if subX[0].size != 0:
                mu = np.array([np.mean(thisX,axis=0) for thisX in subX])
            else:
                mu = np.zeros((1,X.shape[1]))
            centroids.append(mu)
            subXX.append(subX)
        return np.array(centroids).reshape(self.k * X.shape[1])

    def fit(self, X):
        self.centroids_list = np.append(self.centroids_list, self.random_centroids(X))
        self.centroids_list = np.vstack((self.centroids_list, self.centroids_list))
        for i in range(self.iter):
            self.idxs = self.find_closest_centroids(X, self.centroids_list[-1])
            self.centroids_list = np.vstack((self.centroids_list, self.compute_centroids(X, self.idxs)))
        print()


if __name__ == "__main__":
    data = scio.loadmat('./0_Data_Generation/matdata/ex7data2.mat')
    X = data["X"]
    kmeans_classifier = kmeans()
    kmeans_classifier.fit(X)
    plot(X,kmeans_classifier.centroids_list)