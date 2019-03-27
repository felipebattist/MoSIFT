import kmeans_module as km
import sift_module as sm
import numpy as np
import pandas as pd
import os

def gen_train(path,kmeans,label):
    listing = os.listdir(path)
    all_feature_vectors = []
    
    for video in listing:
        data = gen_video_3d_points(video)
        feature_vector = gen_feature_vector(data, kmeans)
        feature_vector.append(label)
        all_feature_vectors.append(feature_vector)
        
    return all_feature_vectors

data_dict = sm.gen_data_set()

dfdata_dict = pd.DataFrame(data_dict)
dfdata_dict.to_csv('data_dict.csv',index=False)

kmeans = km.clustering(data_dict,600)

features_violence = gen_train(r'C:\Users\Arnaldo\Desktop\MoSIFT\train\violence',kmeans,1)
features_non_violence = gen_train(r'C:\Users\Arnaldo\Desktop\MoSIFT\train\non violence',kmeans,0)

train = []
train.append(features_violence)
train.append(features_non_violence)

dftrain = pd.DataFrame(train)
dftrain.to_csv('train.csv',index=False)



