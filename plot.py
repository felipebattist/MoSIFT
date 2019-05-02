from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import random
import kmeans_module as km
import sift_module as sm
import pandas as pd


points = sm.gen_video_3d_points(r'C:\Users\Arnaldo\Desktop\MoSIFT\train\violence\004.wmv')
dfpoints = pd.DataFrame(points)
dfpoints.to_csv('v1.csv',index=True)

v1 = pd.read_csv('v1.csv')

points = sm.gen_video_3d_points(r'C:\Users\Arnaldo\Desktop\MoSIFT\train\non violence\003.wmv')
dfpoints = pd.DataFrame(points)
dfpoints.to_csv('v2.csv',index=True)

v2 = pd.read_csv('v2.csv')

data = (v1, v2)
colors = ("red","blue")

fig = pyplot.figure()
ax = Axes3D(fig)
          
for data, color in zip(data, colors):
    x = data['0'].to_list()
    y = data['1'].to_list()
    z = data['2'].to_list()
    ax.scatter(x, y, z,c=color)
          
pyplot.show()
