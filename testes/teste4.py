import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import open3d as o3d
import os

img = cv.imread('testes/19644.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
corners = cv.goodFeaturesToTrack(gray,50,0.01,10)
corners = np.intp(corners)

for i in corners:
    x,y = i.ravel()
    cv.circle(img,(x,y),3,255,-1)


img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.show()