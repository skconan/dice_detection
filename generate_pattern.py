'''
    File name: generate_pattern.py
    Author: zeabus2018
    Date created: 2018/05/24
    Python Version: 3.6.1
'''

import csv
import math
import cv2 as cv
from lib import *
import numpy as np
import constant as CONST
from operator import itemgetter
from sklearn import linear_model

mk = []
combination_result = []
permutation_result = []
position = []


def create_position():
    start = CONST.DICE_SIZE / 6.
    step = start * 2
    start = int(start)
    step = int(step)
    ct = 0
    for r in range(start, CONST.DICE_SIZE, step):
        position.append([])
        for c in range(start, CONST.DICE_SIZE, step):
            position[ct].append([r, c])
        print(position[ct])
        ct += 1
        print()


def create_table_position():
    table_position = []
    for i in range(60):
        table_position.append([])
        for j in range(60):
            table_position[i].append(
                int(i >= 0) + 3 * int(i >= 20) + 3 *
                int(i >= 40) + int(j >= 20) + int(j >= 40) - 1
            )
            if 0 <= i <= 1 or 18 <= i <= 21 or 38 <= i <= 41 or i>= 59:
                table_position[i][j] = -1
            if 0 <= j <= 1 or 18 <= j <= 21 or 38 <= j <= 41 or j>= 59:
                table_position[i][j] = -1

    return table_position


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
            if i in mk:
                continue
            mk.append(i)
            permutation(n, r, i, res, ct + 1)
            mk.remove(i)


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


def generate_pattern(point=5, ct=0, append=True):
    global position
    print('Generate_images_training_dice: ', str(point))

    p = []
    if point == 5:
        p = [position[0][0], position[0][2], position[1]
             [1], position[2][0], position[2][2]]
    elif point == 61:
        p = [position[0][0], position[0][2], position[1][0],
             position[1][2], position[2][0], position[2][2]]
    elif point == 62:
        p = position[0] + position[2]
    if point > 9:
        point = int(point / 10)
    prefix = 'dice-point-' + str(point) + '-'
    list_of_index = get_index(point, point - 0) + \
        get_index(point, point - 1) + get_index(point, point - 2)
    ct = ct

    index_pattern = []

    position_flat = position[0] + position[1] + position[2]
    print(position_flat)
    print(p)
    for index in list_of_index:
        img_save = np.zeros((CONST.DICE_SIZE, CONST.DICE_SIZE))
        res = []

        '''
            Generate points pattern
        '''

        radius = int(CONST.POINT_RADIUS)
        index_dict = {'x1': 0, 'x2': 0, 'x3': 0, 'x4': 0,
                      'x5': 0, 'x6': 0, 'x7': 0, 'x8': 0, 'x9': 0, 'y': point}
        for i in index:
            x, y = p[i - 1]

            a, b, c, d = int(x - radius / 2.0), int(y - radius /
                                                    2.0), int(x + radius / 2.0), int(y + radius / 2.0)

            cv.circle(img_save, (x, y), radius, (255, 255, 255), -1)
            index_dict['x' + str(position_flat.index([x, y]) + 1)] = 1
        cv.imshow('image save', img_save)
        k = cv.waitKey(-1) & 0xff
        print('Press Y or y to save the pattern (this image)')
        if not (k == ord('y') or k == ord('Y')):
            continue
        index_pattern.append(index_dict)
        cv.imwrite(CONST.POINT_PATH + str(point) + '/' +
                   prefix + str(ct) + '.jpg', img_save)

        ct += 1
    fcsv = CONST.ABS_PATH + 'dataset.csv'
    mode = 'w'
    if append:
        mode = 'a'
    with open(fcsv, mode, newline='\n') as csvfile:
        fieldnames = ['x1', 'x2', 'x3', 'x4',
                      'x5', 'x6', 'x7', 'x8', 'x9', 'y']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not append:
            writer.writeheader()
        for i in index_pattern:
            writer.writerow(i)
    print('Finish generate_images_training')

    # cv.rectangle(img_save,(x_min,y_min),(x_max,y_max),(255,255,255),1)

    # for i in res:
    #     for j in res:
    #         pts = np.array([i, j], np.int32)
    #         pts = pts.reshape((-1, 1, 2))
    #         cv.polylines(img_save, [pts], True, (155, 155, 155), 1)

    # cv.imwrite(CONST.CONNECTED_LINE_PATH + str(point) + '/' + prefix + 'line-' +
    #            str(ct) + '.jpg', img_save)


def main():
    global position

    create_position()
    generate_pattern(5, 0, False)
    generate_pattern(61, 0, True)
    generate_pattern(62, 21, True)


if __name__ == '__main__':
    main()
