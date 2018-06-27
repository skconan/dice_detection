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

'''
    Rewrite Code: old variable
'''
pattern = []
pattern_predict = []
img_result = None
mask_extend_size = 80
keypoints = []
des_list = []
table_position = []
position = [
    [10, 10], [10, 30], [10, 50],
    [30, 10], [30, 30], [30, 50],
    [50, 10], [50, 30], [50, 50]
]

'''
    Rewrite Code: old code
'''


def load_pattern():
    global pattern, pattern_predict
    with open(CONST.ABS_PATH + 'dataset.csv', 'r') as csvfile:
        csvfile.readline()
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            pattern.append([int(v) for v in row[:-2]])
            pattern_predict.append([int(row[-2]),int(row[-1])])


def matching(var):
    count = np.count_nonzero(var)
    if count < 3:
        return False, False
    for (p, res) in zip(pattern, pattern_predict):
        if var == p:
            return res
    return False,False


def load_keypoint():
    global keypoints, des_list
    img_list = []
    five = os.listdir(CONST.CONNECTED_LINE_PATH + '5/')
    six = os.listdir(CONST.CONNECTED_LINE_PATH + '6/')

    for name in five:
        img = cv.imread(CONST.CONNECTED_LINE_PATH + '5/' + name, 1)
        img = cv.resize(img, (60, 60))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        sift = cv.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        des_list.append([5, des, kp, gray])

    for name in six:
        img = cv.imread(CONST.CONNECTED_LINE_PATH + '6/' + name, 1)
        img = cv.resize(img, (60, 60))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        sift = cv.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        des_list.append([6, des, kp, gray])


def sift_matching(mask):
    global des_list

    print('SIFT MATCHING PROCESS...')

    accuracy_max = 0.0
    predict_dice = 0
    print('============', len(des_list), '============')

    for (dice, des, kp, img) in des_list:
        sift = cv.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(mask, None)
        kp2, des2 = sift.detectAndCompute(img, None)

        bf = cv.BFMatcher()

        matches = bf.knnMatch(des1, des2, k=2)
        if len(matches) == 0 or len(matches[0]) < 2:
            continue
        good = []
        print(matches)
        for m, n in matches:
            print(m.distance, n.distance)
            if m.distance < 0.7 * n.distance:
                good.append([m])
        print(len(good), len(matches))
        accuracy = float(len(good)) / len(matches)
        if accuracy >= accuracy_max:
            accuracy_max = accuracy
            predict_dice = dice
            img3 = mask.copy()
            img3 = cv.drawMatchesKnn(mask, kp1, img, kp2, good, img3, flags=4)
            cv.imshow('mask sift', mask)
            cv.imshow('im', img3)

            print(predict_dice, accuracy_max)
    
    print('SIFT MATCHING END')

    return predict_dice, accuracy_max


def get_region(x, y, w, h):
    global img_result

    print('GET REGION PROCESS...')

    dice_w, dice_h = int(
        w / 2 * CONST.SIDE_PER_RADIUS), int(h / 2 * CONST.SIDE_PER_RADIUS)
    a, b = 3, 3
    region = []
    top = int(y - int(0.5 * h)) - a
    left = int(x - int(0.5 * w)) - a
    bottom = top + dice_h + b
    right = left + dice_w + b
    region.append([top, left, bottom, right])
    cv.rectangle(img_result, (left-40,top-40), (right-40,bottom-40), (255,0,0), 1)

    bottom = int(y + int(0.5 * h)) + a
    right = int(x + int(0.5 * w)) + a
    top = bottom - dice_h - b
    left = right - dice_w - b
    region.append([top, left, bottom, right])
    cv.rectangle(img_result, (left-40,top-40), (right-40,bottom-40), (0,255,0), 1)


    bottom = int(y + dice_h / 2.) + a
    right = int(x + dice_w / 2.) + a
    top = int(y - dice_h / 2.) - b
    left = int(x - dice_w / 2.) - b
    region.append([top, left, bottom, right])
    cv.rectangle(img_result, (left-40,top-40), (right-40,bottom-40), (0,0,255), 1)

    print('GET REGION END')

    return region


def what_is_point(mask):
    global table_position, img_result
    print('What is point process...')
    result = [0] * 9
    mask = cv.resize(mask,(60,60))
    mask_result = mask.copy()
    mask_result.fill(0)
    _, mask = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
    _, contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    res = []
    x_min = 100000
    x_max = -1
    y_min = 100000
    y_max = -1
    radius_min = 100000
    radius_max = -1
    if len(contours) > 6 or len(contours) <= 2:
        return result
    hierarchy = hierarchy[0]
    for (cnt,hh) in zip(contours,hierarchy):
        (x, y), radius = cv.minEnclosingCircle(cnt)
        area_real = math.pi * radius * radius
        area_cnt = cv.contourArea(cnt)
        area_ratio = area_cnt / area_real
        approx = cv.approxPolyDP(cnt, 0.01 * cv.arcLength(cnt, True), True)
        len_approx = len(approx)
        # if (not CONST.POINT_AREA_LOWER <= area_cnt <= CONST.POINT_AREA_UPPER or
        #         area_ratio < CONST.EXPECTED_AREA_RATIO):
        #     continue
        x, y, radius = int(x), int(y), radius
        print(result)
        index = table_position[int(x)][int(y)]
        # x, y = position[table_position[int(x/2)][int(y/2)]]
        # print(index)  
        if not is_circle(area_ratio,len_approx,hh):
            continue
        if radius > radius_max :
            radius_max = radius
        if radius < radius_min:
            radius_min = radius
        result[index] += 1
        print(x, y)
        # radius = 6
        # a, b, c, d = int(x - radius / 2.0), int(y - radius /
        #                                         2.0), int(x + radius / 2.0), int(y + radius / 2.0)
        # cv.circle(img_result, (x, y), radius, (0, 255, 255), -1)
    print('RADIUS RATIO:',radius_min,radius_max,radius_min / float(radius_max))
    if   float(radius_max) / radius_min > 2:
        return [0] * 9
    # dice = matching(result)
    # print(dice)
    # if not dice == False:
    #     pass
        # cv.imwrite(CONST.IMG_PATH + 'roi/dice_' + str(dice)+'_'+ str(time.time())+'.jpg', mask_result)
    print('What is point process end')
    
    return result


def find_dice(mask, circles):
    global mask_extend_size,img_result
    data_list = []

    for cir in circles:
        # Interger
        x, y, w, h = cir
        region_list = get_region(x, y, w, h)

        for region in region_list:
            top, left, bottom, right = region
            roi = mask.copy()[top:bottom, left:right]
            try:
                roi = cv.resize(roi, (CONST.DICE_SIZE, CONST.DICE_SIZE))
            except:
                continue
            # roi = mask.copy()[top:bottom, left:right]
            t = time.time()

            point = what_is_point(roi)
            dice, no = matching(point)
 
            if dice == False:
                continue
            prefix = 'dice-point-' + str(dice) + '-'
            print(dice,no)
            img_no = cv.imread(CONST.POINT_PATH + str(dice) + '/' + prefix + str(no) + '.jpg', 0)
            roi = cv.resize(roi, (90,90))
            
            cv.imwrite(CONST.IMG_PATH + 'roi/dice_' + str(t) + '.jpg', np.concatenate((roi,img_no),axis=1))
            center = (int((left + right) / 2) - int(mask_extend_size / 2),
                      int((top + bottom) / 2) - int(mask_extend_size / 2))
            cv.rectangle(img_result,(left-40,top-40),(right-40,bottom-40),(0,255,255),1)
            data_list.append([center,0, dice, np.count_nonzero(point)])
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



def is_circle(area_ratio, len_approx, hierarchy):
    area_ratio_expected = 0.65
    len_approx_expected = 7
    hierarchy_expected = [-1, -1]
    hierarchy = list(hierarchy[2:])
    print(hierarchy_expected,hierarchy)
    if (area_ratio < area_ratio_expected or
        len_approx < len_approx_expected or
            not (hierarchy == hierarchy_expected)):
        print('false')
        return False
    print('true')
    return True


def find_point(img_bin):
    global img_result, mask_extend_size
    r, c = img_bin.shape
    result = np.zeros((r, c), np.uint8)
    result_data = []
    _, cnts, hierarchy = cv.findContours(
        img_bin, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    hierarchy = hierarchy[0]

    for (cnt, hh) in zip(cnts, hierarchy):
        area_cnt = cv.contourArea(cnt)

        # From area of img about 1,000,000 and big point about 60,000
        if len(cnt) < 5 :#or area_cnt > (r * c) * 0.05:
            continue
        print('state 1')
        approx = cv.approxPolyDP(cnt, 0.01 * cv.arcLength(cnt, True), True)
        len_approx = len(approx)

        (x,y),(w,h),angle = ellipse = cv.fitEllipse(cnt)
        area_ellipse = math.pi * (w/2.0) * (h/2.0)

        (x, y), radius = cv.minEnclosingCircle(cnt)
        center = (int(x), int(y))

        area_real = math.pi * radius * radius

        area_ratio = area_cnt / area_real
        area = cv.contourArea(cnt)
        hull = cv.convexHull(cnt)
        hull_area = cv.contourArea(hull)
        solidity = float(area) / hull_area
        if area_cnt > 10000:
                continue
        if not (list(hh[2:]) == [-1,-1]):
            continue
        print('solidity \t area_ratio')
        print(solidity, '\t', area_ratio)
        # if not is_circle(area_ratio, len_approx, hh):
        #     continue
        if not 0.35 <= float(w)/h < 1.2:
            continue
        if not solidity >= 0.9 or not area_cnt/area_ellipse >= 0.8:
            continue
        print('state 2')
        x, y = int(x), int(y)
        # radius = int(radius)
        (x, y), (w, h), angle = ellipse
        result_data.append([x, y, w, h])
        cv.ellipse(result, ellipse, (255, 255, 255), -1)
        (x,y),(w,h),angle = ellipse
        x -=  int(mask_extend_size/2)
        y -=  int(mask_extend_size/2)
        ellipse = (x,y),(w,h),angle
        cv.ellipse(img_result, ellipse, (255, 255, 255), -1)
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

    tmp = np.zeros((r + 80, c + 80), np.uint8)
    tmp[40:-40, 40:-40] = mask
    mask = tmp
    print('FIND MASK THRESH END')

    return mask


def mask_dice(dict):
    global img_result, mask_extend_size
    color = {'2': (255, 0, 0), '5': (0, 255, 0), '6': (0, 0, 255)}
    for d in dict.keys():
        if dict[d] is None:
            continue
        center, radius, dice, accuracy = dict[d]
        x, y = center
        x -= int(mask_extend_size / 2)
        y -= int(mask_extend_size / 2)
        cv.circle(img_result, center, radius, color[d], -1)
        cv.circle(img_result, center, radius * 9, color[d], 2)
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img_result, str(dice), center,
                   font, 1, color[d], 2, cv.LINE_AA)
        cv.imshow('image_result', img_result)


def run(img):
    global img_result, table_position
    img = pre_processing(img)
    mask_th = find_mask_threshold(img)
    mask_circles, circles = find_point(mask_th)
    # cv.imshow('mask_th', mask_th)
    dice_dict = find_dice(mask_circles, circles)
    # cv.imshow('mask_cir', mask_circles)
    mask_dice(dice_dict)
    cv.imshow('result', img_result)
    cv.waitKey(1)


def main():
    global img_result, table_position
    load_pattern()
    table_position = create_table_position()
    # load_keypoint()
    # for i in range(1,10):
    cap = cv.VideoCapture(CONST.VDO_PATH + 'dice_01.mp4')
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
