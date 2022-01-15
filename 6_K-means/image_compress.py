import numpy as np
import matplotlib.pyplot as plt
from kmeans import kmeans
import skimage.io as io

if __name__ == "__main__":

    img = io.imread('./0_Data_Generation/matdata/bird_small.png')

    X = img.reshape(-1, 3)
    K = 16
    kmeans_compress = kmeans(k = K)
    kmeans_compress.fit(X)

    cen = kmeans_compress.centroids_list[-1].reshape(K,X.shape[1])
    compressed_image = np.zeros((len(kmeans_compress.idxs),3))
    for x in range(compressed_image.shape[0]):
        compressed_image[x] = cen[int(kmeans_compress.idxs[x])].astype(int)

    plt.imshow(compressed_image.reshape(img.shape).astype(int))
    plt.show()