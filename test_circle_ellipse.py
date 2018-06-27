import cv2 as cv
from lib import *
import numpy as np
from dice_detection import *


if __name__=='__main__':
    cap = cv.VideoCapture(CONST.VDO_PATH + 'dice_01.mp4')

    while True:
        ret, image = cap.read()
        if image is None:
            continue
        # image = cv.resize(image,(0,0),fx=0.5,fy=0.5)
        image = pre_processing(image)
        mask_th = find_mask_threshold(image)
        img = mask_th.copy()
        img.fill(0)
        
        _,cnts,hierachy = cv.findContours(mask_th,cv.RETR_CCOMP,cv.CHAIN_APPROX_NONE)
        ct = 0
        x_min = 100000
        x_max = -1
        y_min = 100000
        y_max = -1

        for (cnt,hh) in zip(cnts,hierachy[0]):
            if len(cnt) < 5:
                continue
            (x,y),(w,h),angle = ellipse = cv.fitEllipse(cnt)
            x,y,_,_ = cv.boundingRect(cnt)
           
            area = cv.contourArea(cnt)
            area_ellipse = math.pi * (w/2.0) * (h/2.0)
            hull = cv.convexHull(cnt)
            hull_area = cv.contourArea(hull)
            solidity = float(area)/hull_area
            print(ct,w,h,w/h, solidity, hh)
            ct += 1
            # print()
       
            if not (list(hh[2:]) == [-1,-1]):
                continue
            if not (w >= 8 and h>=8):
                continue 
            if not 0.35 <= float(w)/h < 1.2:
                continue
            if not solidity >= 0.925 or not area/area_ellipse >= 0.8:
                continue
            if area > 10000:
                continue
            box = cv.boxPoints(ellipse)
        
            box = np.int0(box)
            cv.ellipse(img,ellipse,(255),-1)
            x,y,w,h = cv.boundingRect(cnt)
            dice_size = max(h/2.0,w/2.0) * 9
           
            # cv.rectangle(img,(int(x-(w*0.5)),int(y-(h*0.5))),(int(x+(w*4.5)),int(y+(h*4.5))),(155),1)
            cv.rectangle(img,(int(x-(w*2)),int(y-(h*2))),(int(x+(w*2.75)),int(y+(h*2.75))),(155),1)
            # cv.rectangle(img,(int(x+(w*0.5)),int(y+(h*0.5))),(int(x-(w*4.5)),int(y-(h*4.5))),(155),1)
            cv.rectangle(img,(int(x),int(y)),(int(x+w),int(y+h)),(155),1)
            # img = cv.drawContours(img,[box],0,(0,0,255),1)
            # img = cv.drawContours(img,cnt,-1,(0,0,255),1)
        cv.imshow('img',img)
        cv.imshow('image',image)
        k = cv.waitKey(-1) & 0xff
        if k == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()