'''
    File name: lib.py
    Author: zeabus2018
    Date created: 05/27/2018
    Python Version: 3.6.1
'''
import math
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
            print(i, j)
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
    mask_result = mask.copy()
    for cnt in contours:
        (x, y), radius = cv.minEnclosingCircle(cnt)
        area_real = math.pi * radius * radius
        area_cnt = cv.contourArea(cnt)
        area_ratio = area_cnt / area_real
        if (not CONST.POINT_AREA_LOWER <= area_cnt <= CONST.POINT_AREA_UPPER or
                area_ratio < CONST.EXPECTED_AREA_RATIO):
            continue
        x, y, radius = int(x), int(y), int(radius)
        cv.circle(mask_result, (x, y), radius, (100, 100, 100), -1)
        index = what_is_index(y, x)
        result[index - 1] = 1
    cv.imshow('mask_point', mask_result)
    cv.waitKey(-1)

    return result

