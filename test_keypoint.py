import cv2 as cv
import numpy as np
import constant as CONST

for i in range(44):
# for i in range(16):
    img = cv.imread(CONST.CONNECTED_LINE_PATH+'six/dice-6-line-'+str(i)+'.jpg',1)
    # img = cv.imread(CONST.CONNECTED_LINE_PATH+'five/dice-5-line-'+str(i)+'.jpg',1)
    
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    sift = cv.xfeatures2d.SIFT_create()
    kp = sift.detect(gray,None)
    cv.drawKeypoints(gray,kp,img)
    cv.imshow('sift_keypoints1',img)
    cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imshow('sift_keypoints2',img)
    cv.waitKey(-1)