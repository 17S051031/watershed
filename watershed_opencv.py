import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('images/pills_02.png')
shifted = cv.pyrMeanShiftFiltering(img, 21, 51)
gray = cv.cvtColor(shifted,cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
cv.imshow("thresh",thresh)

# noise removal
kernel = np.ones((3,3),np.uint8)
# opening=thresh
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
cv.imshow("opening",opening)
# sure background area
sure_bg = cv.dilate(opening,kernel,iterations=3)
sure_bg=255-sure_bg
cv.imshow("sure_bg",sure_bg)
# Finding sure foreground area
dist_transform = cv.distanceTransform(opening,cv.DIST_C,cv.DIST_MASK_PRECISE)
print(0.7*dist_transform.max())
# ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
ret, sure_fg = cv.threshold(dist_transform,0.6*dist_transform.max(),255,0)
cv.imshow("sure_fg",sure_fg)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
allwhite=np.ones(sure_bg.shape,dtype=np.uint8)*255
unknown = cv.subtract(allwhite,sure_bg)
unknown = cv.subtract(unknown,sure_fg)
cv.imshow("unknown",unknown)

# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv.watershed(img,markers)
print("[INFO] {} unique segments found".format(len(np.unique(markers)) - 1))
img[markers == -1] = [255,255,0]
cv.imshow("img",img)
cv.waitKey(0)