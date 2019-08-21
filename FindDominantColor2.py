import cv2
from sklearn.cluster import KMeans
import numpy as np 
import time

W = 480.
img = cv2.imread("kri17.jpg")
height, width, depth = img.shape
imgScale = W/width
newX,newY = img.shape[1]*imgScale, img.shape[0]*imgScale
img = cv2.resize(img,(int(newX),int(newY)))
# img = cv2.blur(img, (13,13))
img = cv2.blur(img, (5,5))
img2show = img.copy()
n_clusters = 3
img2process = img
img2process = np.reshape(img2process, (img2process.shape[0] * img2process.shape[1], 3))
kmeans = KMeans(n_clusters)
kmeans.fit(img2process)
colors = kmeans.cluster_centers_
colors = colors.astype(int)

print(colors)

