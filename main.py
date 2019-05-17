import vbow as vb
import extraction as e
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import os
import matplotlib.pyplot as plt


kp1, dsc_v = e.gen_mosift_features(r"C:/Users/ADM/Desktop/MoSIFT/test/violence/005.wmv")
kp2, dsc_n = e.gen_mosift_features(r"C:/Users/ADM/Desktop/MoSIFT/test/non violence/004.wmv")

dsc_v_e =TSNE ( n_components = 2, random_state=0) .fit_transform ( dsc_v )
dsc_n_e = TSNE ( n_components = 2, random_state=0) .fit_transform ( dsc_n )

data = (dsc_v_e, dsc_n_e)

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter([x[0] for x in dsc_v_e], [x[1] for x in dsc_v_e], c='r', label='violence')
ax1.scatter([x[0] for x in dsc_n_e], [x[1] for x in dsc_n_e], c='b', label='non violence')
plt.show()    


#all_kp, all_dsc = e.gen_mosift_features(r"C:/Users/Arnaldo/Desktop\MoSIFT/dict/009.wmv")
#dfdata_dict = pd.DataFrame(all_3d_points)
#dfdata_dict.to_csv('data_dict.csv',mode='a',index=False)

#e.gen_data_set()

#data_dict = pd.read_csv("data_dict.csv")

#dict_embedded = TSNE ( n_components = 2, random_state=0) .fit_transform ( data_dict )

##features_violence = vb.gen_train(r'C:\Users\Arnaldo\Desktop\MoSIFT\train\violence',kmeans,1)
##features_non_violence = vb.gen_train(r'C:\Users\Arnaldo\Desktop\MoSIFT\train\non violence',kmeans,0)

##train = []
##train.append(features_violence)
##train.append(features_non_violence)

##dftrain = pd.DataFrame(train)
##dftrain.to_csv('train.csv',index=False)


