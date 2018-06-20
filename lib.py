'''
    File name: lib.py
    Author: zeabus2018
    Date created: 05/27/2018
    Python Version: 3.6.1
'''
import math
import numpy as np
import cv2 as cv
import constant as CONST

'''
    Get index of point on dice.
     ___ ___ ___
    |_1_|_2_|_3_|
    |_4_|_5_|_6_|
    |_7_|_8_|_9_|

'''


def what_is_index(r, c):
    count = 1
    step = start = int(CONST.DICE_SIZE / 3)
    stop = CONST.DICE_SIZE + 1
    for i in range(start, stop, step):
        for j in range(start, stop, step):
            if r <= i and c <= j:
                return count
            count += 1
    return None


'''
    Get point from dice(image or roi) 60 x 60 px
'''


def get_point(mask):
    result = [0] * 10
    _, contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # mask_result = mask.copy()
    for cnt in contours:
        (x, y), radius = cv.minEnclosingCircle(cnt)
        area_real = math.pi * radius * radius
        area_cnt = cv.contourArea(cnt)
        area_ratio = area_cnt / area_real
        if (not CONST.POINT_AREA_LOWER <= area_cnt <= CONST.POINT_AREA_UPPER or
                area_ratio < CONST.EXPECTED_AREA_RATIO):
            continue
        x, y, radius = int(x), int(y), int(radius)
        # cv.circle(mask_result, (x, y), radius, (100, 100, 100), -1)
        index = what_is_index(y, x)
        result[index - 1] = 1
    # cv.imshow('mask_point', mask_result)
    # cv.waitKey(-1)

    return result

def get_kernel(shape='rect', ksize=(5, 5)):
    if shape == 'rect':
        return cv.getStructuringElement(cv.MORPH_RECT, ksize)
    elif shape == 'ellipse':
        return cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize)
    elif shape == 'plus':
        return cv.getStructuringElement(cv.MORPH_CROSS, ksize)
    else:
        return None
        
def clahe(imgBGR):
    lab = cv.cvtColor(imgBGR, cv.COLOR_BGR2Lab)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv.merge((l, a, b))
    resBGR = cv.cvtColor(lab, cv.COLOR_Lab2BGR)
    return resBGR

def adjust_gamma(imgBGR=None, gamma=1):
    if imgBGR is None:
        print('given value to imgBGR argument\n' +
              'adjust_gamma_by_value(imgBGR, gamma)')

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) *
                      255 for i in np.arange(0, 256)]).astype("uint8")

    return cv.LUT(imgBGR, table)

def color_mapping(mat):
    '''
        cv.COLORMAP_AUTUMN,
        cv.COLORMAP_BONE,
        cv.COLORMAP_JET,
        cv.COLORMAP_WINTER,
        cv.COLORMAP_RAINBOW,
        cv.COLORMAP_OCEAN,
        cv.COLORMAP_SUMMER,
        cv.COLORMAP_SPRING,
        cv.COLORMAP_COOL,
        cv.COLORMAP_HSV,
        cv.COLORMAP_PINK,
        cv.COLORMAP_HOT
    '''
    norm = None
    norm = cv.normalize(src=mat, dst=norm, alpha=0, beta=255,
                        norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC3)
    return cv.applyColorMap(norm, cv.COLORMAP_HSV)

def inRangeBGR(bgr,lowerb,upperb):
    r,c,_ = bgr.shape  
    bgr = cv.split(bgr)
    result = np.zeros((r,c),np.uint8)
    result.fill(255)
    for i in range(3):
        result[bgr[i]<lowerb[i]] = 0
        result[bgr[i]>upperb[i]] = 0
        
    return  result