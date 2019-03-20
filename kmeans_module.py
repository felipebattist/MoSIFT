import sift_module as sm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

def clustering(data,number_of_clusters):
    kmeans = KMeans(n_clusters=number_of_clusters).fit(data)
    return kmeans

def gen_feature_vector(points_list, clusters):
    feature_vector = np.zeros(len(clusters.cluster_centers_))
    cluster_result =  clusters.predict(points_list)
    for i in cluster_result:
        feature_vector[i] += 1.0
    return feature_vector
