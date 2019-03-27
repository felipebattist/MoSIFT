from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import random
import kmeans_module as km
import sift_module as sm
import pandas as pd


points = sm.gen_video_3d_points(r'C:\Users\Arnaldo\Desktop\MoSIFT\train\violence\004.wmv')
dfpoints = pd.DataFrame(points)

dfpoints.to_csv('video.csv',index=True)

df_gearME = pd.read_csv('video.csv')
x = df_gearME['0'].to_list()
y = df_gearME['1'].to_list()
z = df_gearME['2'].to_list()

fig = pyplot.figure()
ax = Axes3D(fig)

sequence_containing_x_vals = (x)
sequence_containing_y_vals = (y)
sequence_containing_z_vals = (z)

ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)
pyplot.show()
