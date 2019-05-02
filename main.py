import vbow as vb
import extraction as e
import numpy as np
import pandas as pd
import os

all_kp, all_dsc, all_3d_points = e.gen_mosift_features(r"C:/Users/Arnaldo/Desktop\MoSIFT/dict/009.wmv")
dfdata_dict = pd.DataFrame(all_3d_points)
dfdata_dict.to_csv('data_dict.csv',mode='a',index=False)

##e.gen_data_set()

data_dict = pd.read_csv('data_dict.csv')
kmeans = vb.clustering(data_dict,600)

##features_violence = vb.gen_train(r'C:\Users\Arnaldo\Desktop\MoSIFT\train\violence',kmeans,1)
##features_non_violence = vb.gen_train(r'C:\Users\Arnaldo\Desktop\MoSIFT\train\non violence',kmeans,0)

##train = []
##train.append(features_violence)
##train.append(features_non_violence)

##dftrain = pd.DataFrame(train)
##dftrain.to_csv('train.csv',index=False)


