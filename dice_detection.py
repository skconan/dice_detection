'''
    File name: dice_detection.py
    Author: zeabus2018
    Date created: 05/25/2018
    Python Version: 3.6.1
'''
import os
import csv
import cv2 as cv
from lib import *
import numpy as np
from operator import itemgetter
from generate_pattern import *
from matplotlib import pyplot as plt
import time


pattern_list = []
pattern_predict = []
img_result = None
mask_extend_size = 80
table_position = []
position = [
    [10, 10], [10, 30], [10, 50],
    [30, 10], [30, 30], [30, 50],
    [50, 10], [50, 30], [50, 50]
]

'''
    > Load Pattern
    load pattern from .csv file into list
'''


def load_pattern():
    global pattern_list, pattern_predict

    print('Load Pattern Process...')
    with open(CONST.ABS_PATH + 'dataset.csv', 'r') as csvfile:
        csvfile.readline()
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            pattern_list.append([int(v) for v in row[:-2]])
            pattern_predict.append([int(row[-2]), int(row[-1])])
    print('Load Pattern END')


'''
    > Matching
    Check pattern in pattern list
'''


def matching(pattern):
    global pattern_list, pattern_predict

    print('MATCHING Process...')
    not_match = False, 0, 0
    if np.count_nonzero(pattern) < 3:
        return not_match
    for (p, res) in zip(pattern_list, pattern_predict):
        if pattern == p:
            return True, res[0], res[1]
    return not_match
    print('MATCHING END')


'''
    > Get region
    Create region point 3 regions
        - left top
        - right bottom
        - center middle
'''


def get_region(x, y, w, h):
    global img_result

    print('GET REGION PROCESS...')

    dice_w, dice_h = int(
        h / 2.0 * CONST.SIDE_PER_RADIUS), int(h / 2.0 * CONST.SIDE_PER_RADIUS)
    a, b = 1, 1
    region = []
    top = int(y - int(0.5 * h)) - a
    left = int(x - int(0.5 * w)) - a
    bottom = top + dice_h + b
    right = left + dice_w + b
    region.append([top, left, bottom, right])
    # cv.rectangle(img_result, (left - 40, top - 40),
    #              (right - 40, bottom - 40), (255, 0, 0), 1)

    bottom = int(y + int(0.5 * h)) + b
    right = int(x + int(0.5 * w)) + b
    top = bottom - dice_h - a
    left = right - dice_w - a
    region.append([top, left, bottom, right])
    # cv.rectangle(img_result, (left - 40, top - 40),
    #              (right - 40, bottom - 40), (0, 255, 0), 1)

    # bottom = int(y + dice_h / 2.) + b
    # right = int(x + dice_w / 2.) + b
    # top = int(y - dice_h / 2.) - a
    # left = int(x - dice_w / 2.) - a
    # region.append([top, left, bottom, right])
    # cv.rectangle(img_result, (left - 40, top - 40),
    #              (right - 40, bottom - 40), (0, 0, 255), 1)

    print('GET REGION END')

    return region


'''
    > What is pattern
    Find pattern from region of image
'''


def what_is_pattern(mask):
    global table_position, img_result
    print('What is point process...')
    result = [0] * 9
    mask = cv.resize(mask, (60, 60))
    _, mask = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
    _, contours, hierarchy = cv.findContours(
        mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    res = []
    radius_min = 100000
    radius_max = -1
    if len(contours) > 6 or len(contours) <= 2:
        return result
    hierarchy = hierarchy[0]
    for (cnt, hh) in zip(contours, hierarchy):
        (x, y), radius = cv.minEnclosingCircle(cnt)
        area_real = math.pi * radius * radius
        area_cnt = cv.contourArea(cnt)
        area_ratio = area_cnt / area_real
        approx = cv.approxPolyDP(cnt, 0.01 * cv.arcLength(cnt, True), True)
        len_approx = len(approx)
        x, y, radius = int(x), int(y), radius
        print(result)
        index = table_position[int(x)][int(y)]
        if not is_point(cnt, hh):
            continue
        if radius > radius_max:
            radius_max = radius
        if radius < radius_min:
            radius_min = radius
        result[index] += 1
        print(x, y)

    print('RADIUS RATIO:', radius_min, radius_max,
          radius_min / float(radius_max))
    if float(radius_max) / radius_min > 2:
        return [0] * 9
    print('What is point process end')

    return result


'''
    > Find dice
    Find information of dice 5 and 6
'''


def find_dice(mask, circles):
    global mask_extend_size, img_result
    data_list = []

    for cir in circles:
        # Interger
        x, y, w, h = cir
        region_list = get_region(x, y, w, h)
        res_mask = mask.copy()
        for region in region_list:
            top, left, bottom, right = region
            roi = mask.copy()[top:bottom, left:right]
            try:
                roi = cv.resize(roi, (CONST.DICE_SIZE, CONST.DICE_SIZE))
            except:
                continue
            t = time.time()

            point = what_is_pattern(roi)
            is_match, dice, no = matching(point)

            if is_match == False:
                continue
            cv.rectangle(res_mask, (left, top),(right, bottom), (255), 2)

            extend = mask_extend_size/2.0 
            top, left, bottom, right = top - extend, left - extend, bottom - extend, right - extend
            top, left, bottom, right = int(top), int(left), int(bottom), int(right)
            center = (int((left + right) / 2.0) ,int((top + bottom) / 2.0))

            cv.rectangle(img_result, (left, top),(right, bottom), (0, 255, 255), 1)

            data_list.append([center, int(max(w,h)/2.0), dice, np.count_nonzero(point)])

            ########### SAVE ROI THAT IS MAYBE DICE #########
            prefix = 'dice-point-' + str(dice) + '-'
            print(dice, no)
            img_no = cv.imread(CONST.POINT_PATH + str(dice) +
                               '/' + prefix + str(no) + '.jpg', 0)
            roi = cv.resize(roi, (90, 90))
            cv.imwrite(CONST.IMG_PATH + 'roi/dice_' + str(t) +
                       '.jpg', np.concatenate((roi, img_no), axis=1))
            #########################################################
            cv.imshow('mask',res_mask)
            cv.waitKey(1)
    data_dict = remove_redundant_dice(data_list)
    return data_dict


def remove_redundant_dice(data):
    result = {'5': None, '6': None}
    data = sorted(data, key=itemgetter(2))
    for d in data:
        index_dict = str(d[2])
        if result[index_dict] is None or (result[index_dict][3] < d[3] or (result[index_dict][3] == d[3] and result[index_dict][3] > d[3])):
            result[index_dict] = d

    return result


def is_point(contour, hierarchy):
    '''
        <--------------------- Condition --------------------->
        - Length contour is point in contour
        - POINT_AREA_UPPER*0.5 > Area contour > POINT_AREA_LOWER*0.5
            Because reduce image size to half size
        - Convex is True
        - Area ratio > Area ratio expected
        - Length approximation is number of line that approximates a contour
            Length approx > Length approx expected
        - Hierarchy is relationship between each contour
            [Next, Previous, First_Child, Parent]
            [ don't care   ,     -1     ,  -1  ]
            That mean don't have child contour and parent contour 
        - Solidity is the ratio of contour area to its convex hull area.
    '''
    ############################ Expected Value ##################################
    area_upper = CONST.POINT_AREA_UPPER
    area_lower = CONST.POINT_AREA_LOWER * 0.1
    hierarchy_expected = [-1, -1]
    area_ratio_expected = 0.6
    solidity_expected = 0.65
    len_approx_expected = 6
    len_cnt_expected = 5

    ###########################################################################
    len_cnt = len(contour)
    if not len_cnt > len_cnt_expected:
        return False
    ###########################################################################
    hierarchy = list(hierarchy[2:])
    ###########################################################################
    (x, y), (w, h), angle = ellipse = cv.fitEllipse(contour)
    area_ellipse = math.pi * w * 0.5 * h * 0.5
    if area_ellipse == 0:
        return False
    ###########################################################################
    area_cnt = cv.contourArea(contour)
    area_ratio = area_cnt / area_ellipse
    ###########################################################################
    '''
        perimeter * 0.02 because can define convex or concave
        boolean is True that only closed contour
    '''
    perimeter = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, 0.02 * perimeter, True)
    len_approx = len(approx)
    is_convex = cv.isContourConvex(approx)
    ###########################################################################
    hull = cv.convexHull(approx)
    hull_area = cv.contourArea(hull)
    if hull_area == 0:
        return False
    solidity = float(area_cnt) / hull_area

    ################################ CONDITION ##################################
    if not area_lower < area_cnt < area_upper:
        print('BREAK BY AREA RANGE')
        return False
    if not len_approx > len_approx_expected:
        print('BREAK BY LENGTH APPROX')
        return False
    if not area_ratio > area_ratio_expected:
        print('BREAK BY AREA RATIO')
        return False
    if not solidity > solidity_expected:
        print('BREAK BY SOLIDITY')
        return False
    if not hierarchy == hierarchy_expected:
        print('BREAK BY HIERARCHY')
        return False
    if not is_convex:
        return False

    return True


def find_point(img_bin):
    global img_result, mask_extend_size
    print('Find POint')
    r, c = img_bin.shape
    result = np.zeros((r, c), np.uint8)
    result_data = []
    _, cnts, hierarchy = cv.findContours(
        img_bin, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    hierarchy = hierarchy[0]

    for (cnt, hh) in zip(cnts, hierarchy):
        if not is_point(cnt, hh):
            continue
        (x, y), (w, h), angle = ellipse = cv.fitEllipse(cnt)
        result_data.append([x, y, w, h])
        cv.ellipse(result, ellipse, (255, 255, 255), -1)

        ################ DISPLAY ###########################
        x -= int(mask_extend_size / 2.)
        y -= int(mask_extend_size / 2.)
        ellipse = (x, y), (w, h), angle
        cv.ellipse(img_result, ellipse, (255, 255, 255), -1)
        ####################################################

    return result, result_data


def pre_processing(img_bgr):
    result = cv.GaussianBlur(img_bgr, (5, 5), 0)
    return result


def find_mask_threshold(img_bgr):
    global mask_extend_size, img_result
    print('FIND MASK THRESH PROCESS...')
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    r, c = gray.shape

    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    v_avg = np.average(v)

    _, mask = cv.threshold(gray, v_avg / 1.75, 255, cv.THRESH_BINARY_INV)
    erode = cv.erode(mask, get_kernel('rect', (3, 3)))
    mask = cv.dilate(erode, get_kernel('rect', (3, 3)))

    img_result[mask < 127] = 100

    tmp = np.zeros((r + mask_extend_size, c + mask_extend_size), np.uint8)
    extend = int(mask_extend_size / 2.0)
    print(tmp.shape)
    print(mask.shape)
    tmp[extend:-extend, extend:-extend] = mask
    print(tmp.shape)

    print('FIND MASK THRESH END')

    return tmp


def mask_dice(dict):
    global img_result, mask_extend_size
    color = {'2': (255, 0, 0), '5': (0, 255, 0), '6': (0, 0, 255)}
    for d in dict.keys():
        if dict[d] is None:
            continue
        center, radius, dice, accuracy = dict[d]
        cv.circle(img_result, center, radius, color[d], -1)
        cv.circle(img_result, center, radius * 9, color[d], 2)
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img_result, str(dice), center,
                   font, 1, color[d], 2, cv.LINE_AA)
        # cv.imshow('image_result', img_result)


def run(img):
    global img_result, table_position
    '''
        Enhancement BGR image in pre_processing() before do anything.
        mask_th is the result of grayscale that have color value less than threshold and extend the frame.
        mask_circles is the binary image that (only) have a circle(s).
        circles is x, y, w, h, and angle of each circle(s) in mask_circle(s) (extend) 
    '''
    img = pre_processing(img)
    mask_th = find_mask_threshold(img)
    mask_circles, circles = find_point(mask_th)
    
    '''
        mask_circles
    '''

    dice_dict = find_dice(mask_circles, circles)
    mask_dice(dice_dict)
    cv.imshow('result', img_result)
    cv.waitKey(1)


def main():
    global img_result, table_position
    load_pattern()
    table_position = create_table_position()

    cap = cv.VideoCapture(CONST.VDO_PATH + 'dice_11.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            continue

        frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
        img_result = frame.copy()
        run(frame)
        k = cv.waitKey(1) & 0xff
        if k == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
