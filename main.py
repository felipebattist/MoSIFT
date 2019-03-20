import kmeans_module as km
import sift_module as sm
import numpy as np

data = sm.gen_data_set()
data_array = np.asarray(data)

video = sm.gen_video_3d_points(r'C:\Users\Arnaldo\Desktop\MoSIFT\test\002.wmv')
video_array = np.asarray(video)

kmeans = km.clustering(data_array,2)

#testes
histogram = km.gen_feature_vector(data_array,kmeans)
histogram2 = km.gen_feature_vector(video_array,kmeans)
