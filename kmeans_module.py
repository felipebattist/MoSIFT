import sift_module as sm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

data = sm.gen_data_set()
data_array = np.asarray(data)

def clustering_and_ploting():
    kmeans = KMeans(n_clusters=10).fit(data)
    plt.scatter(data_array[:,1], data_array[:,2], s = 5, c = kmeans.labels_)
    plt.show()
