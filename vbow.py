import extraction as e
#from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans 
import matplotlib.pyplot as plt
import os
import numpy as np

def clustering(data,number_of_clusters):
    kmeans = MiniBatchKMeans(n_clusters=number_of_clusters, batch_size=50).fit(data)
    return kmeans

def gen_feature_vector(points_list, clusters):
    feature_vector = np.zeros(len(clusters.cluster_centers_))
    cluster_result =  clusters.predict(points_list)
    for i in cluster_result:
        feature_vector[i] += 1.0
    return feature_vector
   

def gen_train(path,kmeans):
    listing = os.listdir(path)
    all_feature_vectors = []
    for video in listing:
        video = path+video
        all_kp, all_dsc = e.gen_mosift_features(video)
        feature_vector = gen_feature_vector(all_dsc, kmeans)
        #feature_vector = np.append(feature_vector, label)
        all_feature_vectors.append(feature_vector)
    
    return all_feature_vectors
