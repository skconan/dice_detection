'''
    File name: dice_detection.py
    Author: zeabus2018
    Date created: 05/25/2018
    Python Version: 3.6.1
'''
import cv2 as cv
import csv
import numpy as np
from lib import *
from operator import itemgetter
pattern = []
pattern_predict = []


def load_pattern():
    global pattern, pattern_predict
    with open(CONST.ABS_PATH + 'dataset.csv', 'r') as csvfile:
        csvfile.readline()
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            pattern.append([int(v) for v in row[:-1]])
            pattern_predict.append(int(row[-1]))


def matching(var):
    for (p, res) in zip(pattern, pattern_predict):
        if var == p:
            return res
    return None


def get_circle_in_frame(mask):
    area_ratio_expected = 0.6
    circles = []
    radius_list = []

    _, contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    for cnt in contours:
        (x, y), radius = cv.minEnclosingCircle(cnt)
        area_real = math.pi * radius * radius
        area_cnt = cv.contourArea(cnt)
        area_ratio = area_cnt / area_real
        if area_cnt <= 20 or area_ratio < area_ratio_expected:
            continue
        x, y, radius = int(x), int(y), int(radius)
        circles.append([x, y, radius])
        radius_list.append(radius)
    radius_avg = np.average(radius_list)

    return circles, radius_avg


def may_be_dice(mask):
    result = get_point(mask)
    is_one = np.count_nonzero(result)
    if is_one >= 2:
        return True, result
    else:
        return False, None


def get_region(dice_size, x, y, radius):
    a,b = 4,14
    region = []
    top = int(y - radius - int(0.5 * radius)) - a
    left = int(x - radius - int(0.5 * radius)) - a
    bottom = top + dice_size + b
    right = left + dice_size + b
    region.append([top, left, bottom, right])

    bottom = int(y + radius + int(0.5 * radius)) + a
    right = int(x + radius + int(0.5 * radius)) + a
    top = bottom - dice_size - b
    left = right - dice_size - b
    region.append([top, left, bottom, right])

    return region


def get_dice_position(mask, circles, radius_avg):
    dice_size = int(radius_avg * CONST.SIDE_PER_RADIUS)
    data_list = []

    for cir in circles:
        x, y, radius = cir
        region_list = get_region(dice_size, x, y, radius)
        x, y, radius = int(x), int(y), int(radius)
        for region in region_list:
            top, left, bottom, right = region
            roi = mask.copy()[top:bottom, left:right]
            if not len(roi):
                continue
            roi = cv.resize(roi, (CONST.DICE_SIZE, CONST.DICE_SIZE))
            is_dice, point = may_be_dice(roi)
            if not is_dice:
                continue
            count = np.count_nonzero(point)
            dice = matching(point[:-1])
            if dice is None:
                continue
            data_list.append([x, y, radius, dice, point, count])
    data_dict = remove_redundant_dice(data_list)
    return data_dict


def remove_redundant_dice(data):
    result = {'2': None, '5': None, '6': None}
    data = sorted(data, key=itemgetter(3))
    for d in data:
        if result[str(d[3])] is None or result[str(d[3])][5] < d[5]:
            result[str(d[3])] = d

    return result


def mask_dice(img, dict):
    color = {'2': (255, 0, 0), '5': (0, 255, 0), '6': (0, 0, 255)}
    for d in dict.keys():
        if dict[d] is None:
            continue
        x, y, radius, dice, point, count = dict[d]
        cv.circle(img, (x, y), radius, color[d], -1)

    # cv.imshow('roi', img)
    # cv.waitKey(-1)
    # cv.imshow('image_result', img_result)
    # cv.imshow('equ', equ)
    # cv.imshow('gray', gray)
    cv.imshow('img', img)
    cv.waitKey(-1)


def main():
    load_pattern()
    img = cv.imread(CONST.IMG_PATH + 'dice.jpg', 1)
    img = cv.resize(img, (0, 0), fx=0.5, fy=0.5)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    equ = cv.equalizeHist(gray)
    _, mask = cv.threshold(equ, 30, 255, cv.THRESH_BINARY_INV)
    circles, radius_avg = get_circle_in_frame(mask)
    dice_dict = get_dice_position(mask, circles, radius_avg)
    mask_dice(img, dice_dict)


if __name__ == '__main__':
    main()
    # matching()
