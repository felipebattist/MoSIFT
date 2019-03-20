import kmeans_module as km
import sift_module as sm
import numpy as np
import pandas as pd

data = sm.gen_data_set()
video = sm.gen_video_3d_points(r'C:\Users\Arnaldo\Desktop\MoSIFT\test\002.wmv')

dftrain = pd.DataFrame(data)
dftrain.to_csv('train.csv',index=False)

dftest = pd.DataFrame(video)
dftest.to_csv('test.csv',index=False)

kmeans = km.clustering(data,2)

histogram_ds = km.gen_feature_vector(data,kmeans)
histogram_video = km.gen_feature_vector(video,kmeans)

