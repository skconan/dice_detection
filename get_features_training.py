'''
    File name: get_features.py
    Author: zeabus2018
    Date created: 2018/05/24
    Python Version: 3.6.1
'''

import cv2 as cv
import math
from operator import itemgetter
from sklearn import linear_model
import csv
from lib import *
import constant as CONST

mk = []
combination_result = []
permutation_result = []


def combination(n, r, k, res, ct):
    global combination_result
    if ct > 0:
        res += ' ' + str(k)
    if ct == r:
        combination_result.append(res)
    else:
        for i in range(k + 1, n + 1):
            combination(n, r, i, res, ct + 1)


def permutation(n, r, k, res, ct):
    global permutation_result, mk
    if ct > 0:
        res += ' ' + str(k)
    if ct == r:
        permutation_result.append(res)
    else:
        for i in range(1, n + 1):
            if not i in mk:
                mk.append(i)
                permutation(n, r, i, res, ct + 1)
                mk.remove(i)


def get_point(img_bgr):
    result = []
    if len(img_bgr.shape) == 3:
        gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    else:
        gray = img_bgr
    _, mask = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    _, contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    mask_result = mask.copy()
    for cnt in contours:
        (x, y), radius = cv.minEnclosingCircle(cnt)
        area_real = math.pi * radius * radius
        area_cnt = cv.contourArea(cnt)
        area_ratio = area_cnt / area_real
        # if (not CONST.POINT_AREA_LOWER <= area_cnt <= CONST.POINT_AREA_UPPER or
        #         area_ratio < CONST.EXPECTED_AREA_RATIO):
        #     continue
        x, y, radius = int(x), int(y), int(radius)
        cv.drawContours(mask_result, cnt, -1, (0, 255, 0), 1)
        result.append([x, y, radius])
        index = what_is_index(y, x)
        # result[index - 1] = 1
    cv.imshow('mask_point', mask_result)
    cv.waitKey(-1)

    return result


def get_index(n, r):
    global combination_result
    combination_result = []
    mk = []
    combination(n, r, 0, '', 0)
    combination_result = list(set(combination_result))

    res = []
    for i in combination_result:
        tmp = [int(k) for k in i.split(' ')[1:]]
        res.append(tmp)

    return sorted(res)


def generate_data_training():
    circles = []
    for i in [2, 5, 6]:
        prefix = 'dice-' + str(i) + '-'
        for j in range(44):
            img = cv.imread(CONST.IMG_PATH + prefix + str(j) + '.jpg')
            if img is None:
                break
            img = cv.resize(img, (60, 60))
            circle = get_point(img)
            circle['y'] = i
            circles.append(circle)

    fcsv = CONST.ABS_PATH + 'dataset.csv'
    with open(fcsv, 'w', newline='\n') as csvfile:
        fieldnames = ['x1', 'x2', 'x3', 'x4',
                      'x5', 'x6', 'x7', 'x8', 'x9', 'y']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for c in circles:

            writer.writerow(c)


def generate_images_training(img, point=5, ct=0):
    prefix = 'dice-' + str(point) + '-'
    circles = get_point(img)
    list_of_index = get_index(point, point - 0) + \
        get_index(point, point - 1) + get_index(point, point - 2)
    ct = ct
    for index in list_of_index:
        img_save = img.copy()
        img_save.fill(0)
        res = []
        print(index)
        for i in index:
            x, y, radius = circles[i - 1]
            cv.circle(img_save, (x, y), radius, (255, 255, 255), -1)
            res.append([x, y])

        for i in res:
            for j in res:
                pts = np.array([i, j], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv.polylines(img_save, [pts], True, (255, 255, 255), 2)
        cv.imwrite(CONST.IMG_PATH + prefix + 'line-' +
                   str(ct) + '.jpg', img_save)
        ct += 1


def generate_images_training_all():
    img = cv.imread(CONST.IMG_PATH + 'data_set\dice_5_0.jpg', 1)
    generate_images_training(img, 5)
    img = cv.imread(CONST.IMG_PATH + 'data_set\dice_6_0.jpg', 1)
    generate_images_training(img, 6)
    img = cv.imread(CONST.IMG_PATH + 'data_set\dice_6_1.jpg', 1)
    generate_images_training(img, 6, 22)


def main():
    generate_images_training_all()
    # generate_data_training()


if __name__ == '__main__':
    main()
