import vbow as vb
import extraction as e
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import os
import matplotlib.pyplot as plt

##e.gen_data_set()
##carregar dicionário em csv e criar as palavras
data_dict = pd.read_csv("data_dict.csv", chunksize=2000,low_memory = False)
data = pd.concat(data_dict, ignore_index=True)
kmeans = vb.clustering(data, 600)
print("CLUSTERIZOU")
##
train_clapping = vb.gen_train(r"C:/Users/ADM/Desktop/MoSIFT/train/handclapping/",kmeans)

df_train_clapping = pd.DataFrame(train_clapping)
df_train_clapping.to_csv('train_clapping.csv',mode='a', index=False)
df_train_clapping = []
##
train_boxing = vb.gen_train(r"C:/Users/ADM/Desktop/MoSIFT/train/boxing/",kmeans)

df_train_boxing = pd.DataFrame(train_boxing)
df_train_boxing.to_csv('train_boxing.csv',mode='a', index=False)
df_train_boxing = []

test_clapping= vb.gen_train(r"C:/Users/ADM/Desktop/MoSIFT/test/handclapping/",kmeans)
##
df_test_clapping = pd.DataFrame(test_clapping)
df_test_clapping.to_csv('test_clapping.csv',mode='a', index=False)
df_test_clapping = []
##
test_boxing = vb.gen_train(r"C:/Users/ADM/Desktop/MoSIFT/test/boxing/",kmeans)

df_test_boxing = pd.DataFrame(test_boxing)
df_test_boxing.to_csv('test_boxing.csv',mode='a', index=False)
df_test_boxing = []
