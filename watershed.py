import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import label2rgb

imagePath = ""
img = cv2.imread(imagePath)
ret,thld_img = cv2.threshold(img[:,:,0],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

plt.figure()
plt.title("Image")
plt.imshow(thld_img,cmap = 'gray')
kernel_open = np.ones((5,5),np.uint8)

opening = cv2.morphologyEx(thld_img,cv2.MORPH_OPEN,kernel_open,iterations=2) #to remove stray pixels

kernel_dilate = np.ones((3,3),np.uint8)
sure_bg = cv2.dilate(opening,kernel_dilate,iterations=15)

plt.figure()
plt.title("Sure BG")
plt.imshow(sure_bg, cmap = 'gray')
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)

ret,sure_fg = cv2.threshold(dist_transform,0.6*dist_transform.max(),255,0)
sure_fg = np.uint8(sure_fg)

plt.figure()
plt.title("Sure FG")
plt.imshow(sure_fg, cmap = 'gray')

roi = np.subtract(sure_bg,sure_fg)

plt.figure()
plt.title("ROI")
plt.imshow(roi, cmap = 'gray')

ret2, markers = cv2.connectedComponents(sure_fg)

markers = markers + 10

markers[roi==255] = 0

markers = cv2.watershed(img,markers)

colored_overlay = label2rgb(markers,bg_label =10)

plt.imshow(colored_overlay)

