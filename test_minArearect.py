import cv2 as cv
from lib import *
import numpy as np

if __name__=='__main__':
    img = cv.imread(CONST.ABS_PATH+'mask_cir_screenshot_24.06.2018.png',1)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    _,th = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(gray)
    print('------------------------------------------')
    print(min_val, max_val, min_loc, max_loc)
    print('------------------------------------------')
    _,cnts,_ = cv.findContours(th,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    ct = 0
    x_min = 100000
    x_max = -1
    y_min = 100000
    y_max = -1

    for cnt in cnts[:5]:
        (x,y),(w,h),angle = rect = cv.minAreaRect(cnt)
        x,y,w,h = cv.boundingRect(cnt)
        a, b, c, d = int(x), int(y), int(x + w), int(y + h)
        if a < x_min:
            x_min = a
        if b < y_min:
            y_min = b
        if c > x_max:
            x_max = c
        if d > y_max:
            y_max = d
        print(a,b,c,d)
        print(x_min,x_max,y_min,y_max)
        box = cv.boxPoints(rect)
    
        box = np.int0(box)
        img = cv.drawContours(img,[box],0,(0,0,255),1)
        img = cv.drawContours(img,cnt,-1,(0,0,255),1)
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img,str(ct),(int(x),int(y-15)), font, 0.25,(255,255,255),1,cv.LINE_AA)
        print(ct,rect)
        ct += 1
    roi = gray[y_min:y_max,x_min:x_max]
    rows, cols = roi.shape
    M = cv.getRotationMatrix2D((cols/2,rows/2),10,1)
    dst = cv.warpAffine(roi,M,(cols,rows))
    cv.imshow('dst',dst)
    cv.imshow('roi',roi)
    cv.imshow('img',img)
    cv.waitKey(-1)