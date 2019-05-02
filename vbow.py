import extraction as e
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

def gen_train(path,kmeans,label):
    listing = os.listdir(path)
    all_feature_vectors = []
    
    for video in listing:
        data = e.gen_video_3d_points(video)
        feature_vector = e.gen_feature_vector(data, kmeans)
        feature_vector.append(label)
        all_feature_vectors.append(feature_vector)
        
    return all_feature_vectors
