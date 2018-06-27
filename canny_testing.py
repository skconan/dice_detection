import cv2 as cv
import numpy as np
import constant as CONST
def nothing(x):
    pass

def canny():
    cap = cv.VideoCapture(CONST.VDO_PATH+'dice_03.mp4')
    cv.namedWindow('image')
    cv.createTrackbar('min', 'image', 0, 1000, nothing)
    cv.createTrackbar('max', 'image', 0, 1000, nothing)
    cv.setTrackbarPos('min','image',100)
    cv.setTrackbarPos('max','image',300)
    while cap.isOpened():
        ret, img = cap.read()
        if img is None:
            continue
        img = cv.resize(img,(0,0),fx=0.5,fy=0.5)
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)
        
        min_val = cv.getTrackbarPos('min', 'image')
        max_val = cv.getTrackbarPos('max', 'image')
        canny = cv.Canny(gray,min_val,max_val)
        
        cv.imshow('image_default',img)
        cv.imshow('image',canny)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cv.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    canny()