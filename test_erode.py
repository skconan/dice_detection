import cv2 as cv
from lib import *
import numpy as np
import constant as CONST

if __name__ == '__main__':
    im = cv.imread(CONST.DATA_SET_PATH+'point/2/dice_2_0.jpg',0)
    _,th = cv.threshold(im,127,255,cv.THRESH_BINARY)
    erode = cv.erode(th,get_kernel('rect',(7,7)))
    dilate = cv.dilate(erode,get_kernel('rect',(7,7)))
    cv.imshow('erode',erode)
    cv.imshow('th',th)
    cv.imshow('dilate',dilate)
    cv.waitKey(-1)