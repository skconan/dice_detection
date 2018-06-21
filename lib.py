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


'''
    Rewrite code: old code
'''
# def what_is_index(r, c):
#     count = 1
#     step = start = int(CONST.DICE_SIZE / 3)
#     stop = CONST.DICE_SIZE + 1
#     for i in range(start, stop, step):
#         for j in range(start, stop, step):
#             if r <= i and c <= j:
#                 return count
#             count += 1
#     return None


'''
    Get point from dice(image or roi) 60 x 60 px
    old code
'''


# def get_point(mask):
#     result = [0] * 10
#     mask_result = mask.copy()
#     mask_result.fill(0)
#     _, mask = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
#     _, contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
#     res = []
#     for cnt in contours:
#         (x, y), radius = cv.minEnclosingCircle(cnt)
#         area_real = math.pi * radius * radius
#         area_cnt = cv.contourArea(cnt)
#         area_ratio = area_cnt / area_real
#         if (not CONST.POINT_AREA_LOWER <= area_cnt <= CONST.POINT_AREA_UPPER or
#                 area_ratio < CONST.EXPECTED_AREA_RATIO):
#             continue
#         x, y, radius = int(x), int(y), int(radius)
#         cv.circle(mask_result, (x, y), radius, (255, 255, 255), -1)
#         index = what_is_index(y, x)
#         result[index - 1] = 1
#         res.append([x, y])
#     for i in res:
#         for j in res:
#             pts = np.array([i, j], np.int32)
#             pts = pts.reshape((-1, 1, 2))
#             cv.polylines(mask_result, [pts], True, (255, 255, 255), 2)

#     _, mask = cv.threshold(mask_result, 127, 255, cv.THRESH_BINARY)
#     _, contours, hierarchy = cv.findContours(
#         mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
#     print(hierarchy)
#     hierarchy = hierarchy[0]
#     print(len(contours))
#     x, y, width, height = 0,0,0,0
#     if len(contours) > 0:
#         for (cnt, h) in zip(contours, hierarchy):
#             if h[3] == -1:
#                 x, y, width, height = cv.boundingRect(cnt)
#                 cv.drawContours(mask_result, cnt, -1, (155, 155, 155), 2)
#                 # box = cv.boxPoints(rect)
#                 # box = np.int0(box)
#                 # cv.drawContours(mask_result,[box],0,(100,100,100),2)
#     rows, cols = mask.shape
#     if x + y + width + height > 0:
#         pts1 = np.float32(
#             [[x, y], [x + width, y], [x, y + height], [x + width, y + height]])
#         pts2 = np.float32([[0, 0], [CONST.DICE_SIZE, 0], [0, CONST.DICE_SIZE], [
#                           CONST.DICE_SIZE, CONST.DICE_SIZE]])
#         M = cv.getAffineTransform(pts1[:-1],pts2[:-1])
#         mask = cv.warpAffine(mask,M,(cols,rows))
#         # M = cv.getPerspectiveTransform(pts1, pts2)
#         # mask = cv.warpPerspective(mask, M, (cols, rows))


#     cv.imshow('mask_point', mask)
#     cv.imshow('mask_result1', mask_result)
#     cv.waitKey(-1)

#     # cv.waitKey(-1)
#     print(result)

#     return result


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


def inRangeBGR(bgr, lowerb, upperb):
    r, c, _ = bgr.shape
    bgr = cv.split(bgr)
    result = np.zeros((r, c), np.uint8)
    result.fill(255)
    for i in range(3):
        result[bgr[i] < lowerb[i]] = 0
        result[bgr[i] > upperb[i]] = 0

    return result
