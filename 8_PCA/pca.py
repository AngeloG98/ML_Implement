import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt

class PCA:
    def __init__(self) -> None:
        pass

    def featureNormalize(self, X):
        self.means = np.mean(X,axis=0)
        self.X_norm = X - self.means
        self.stds  = np.std(self.X_norm,axis=0)
        self.X_norm = self.X_norm / self.stds

    def getUSV(self):
        cov_matrix = self.X_norm.T.dot(self.X_norm)/self.X_norm.shape[0] # XX^T
        self.U, self.S, self.V = np.linalg.svd(cov_matrix, full_matrices = True, compute_uv = True)


    def projectData(self, X, K):
        Ureduced = self.U[:,:K]
        z = X.dot(Ureduced)
        return z

    def reduction(self,X):
        self.featureNormalize(X)
        self.getUSV()
        return self.projectData(self.X_norm, 1)
        


if __name__ == "__main__":
    data = scio.loadmat('./0_Data_Generation/matdata/ex7data1.mat')
    X = data["X"]
    pca_reduction = PCA()
    Z = pca_reduction.reduction(X)
    print(Z)
