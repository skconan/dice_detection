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

mk = []
combination_result = []
permutation_result = []
abs_path = 'C:/Users/skconan/Desktop/Workspace/dice/'
img_path = abs_path + 'images/'


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


def what_is_index(r,c):
    count = 1
    for i in range(20, 61, 20):
        for j in range(20, 61, 20):
            if r <= i and c <= j:
                return count
            count += 1

def get_point(img_bgr):
    result = {'x1': 0, 'x2': 0, 'x3': 0,'x4': 0, 'x5': 0, 'x6': 0, 'x7': 0, 'x8': 0, 'x9': 0, 'y': 0}
    img_result = img_bgr.copy()
    area_ratio_expected = 0.8
    if len(img_bgr.shape) == 3:
        gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    else :
        gray = img_bgr
    _, mask = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)
    _, contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        (x, y), radius = cv.minEnclosingCircle(cnt)
        area_real = math.pi * radius * radius
        area_cnt = cv.contourArea(cnt)
        if area_cnt <= 100:
            continue
        area_ratio = area_cnt / area_real
        if area_ratio < area_ratio_expected:
            continue
        cv.drawContours(img_result, cnt, -1, (0, 255, 0), 1)
        # result.append([int(x), int(y), int(radius)])
        index = what_is_index(y,x)
        result['x'+str(index)] = 1

    # result = sorted(result, key=itemgetter(2))
    # result_last = []
    # for res in result:
    #     result_last.append(res[:-1])
    cv.imshow('image_result', img_result)
    cv.waitKey(-1)

    # return result_last
    print(result)
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
            img = cv.imread(img_path + prefix + str(j) + '.jpg')
            if img is None:
                break
            img = cv.resize(img,(60,60))
            circle = get_point(img)
            circle['y'] = i
            circles.append(circle)

    fcsv = abs_path + 'dataset.csv'
    with open(fcsv, 'w', newline='\n') as csvfile:
        fieldnames = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'y']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for c in circles:
           
            writer.writerow(c)


def generate_images_training(img, point=5, ct=0):
    prefix = 'dice-' + str(point) + '-'
    circles = get_point(img)
    list_of_index = get_index(point, 0) + \
        get_index(point, 1) + get_index(point, 2)
    ct = ct
    for index in list_of_index:
        img_save = img.copy()
        for i in index:
            x, y, radius = circles[i - 1]
            cv.circle(img_save, (x, y), radius + 2, (0, 0, 0), -1)
        cv.imwrite(img_path + prefix + str(ct) + '.jpg', img_save)
        ct += 1


def generate_images_training_all():
    img = cv.imread(img_path + 'dice_5_0.jpg', 1)
    generate_images_training(img, 5)
    img = cv.imread(img_path + 'dice_6_0.jpg', 1)
    generate_images_training(img, 6)
    img = cv.imread(img_path + 'dice_6_1.jpg', 1)
    generate_images_training(img, 6, 22)


def main():
    # generate_images_training_all()
    generate_data_training()


if __name__ == '__main__':
    main()
